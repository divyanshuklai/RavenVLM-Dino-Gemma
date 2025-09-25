import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel


class GemmaDinoImageCaptioner(nn.Module):
    def __init__(self,
                 gemma_id="google/gemma-3-270m",
                 vit_id="facebook/dinov3-vits16plus-pretrain-lvd1689m",
                 prompt="In this Image: ",
                 max_caption_length=128, 
                 include_cls=True,
                 include_registers=False, 
                 include_patches=False,
                 freeze_gemma=True):
        """
        Gemma-270M + DinoV3-ViT-S+ Image Captioner. 
        DinoV3 ViT outputs (B, 1xCLS + 4xREG + PT, 384).
    
        :param include_cls: include the cls token of ViT which summarizes the image in the gemma input, default to true
        :param include_registers: include the 4x register tokens of ViT for the gemma input, default to false
        :param include_patches: include the HxW/16^2 patch tokens of image in the gemma input, default to false

        """
        super().__init__()

        self.gemma_id = gemma_id
        self.vit_id = vit_id

        self.include_cls = include_cls
        self.include_registers = include_registers
        self.include_patches = include_patches

        self.prompt = prompt
        self.max_caption_length = max_caption_length
        

        self.gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_id)

        self.gemma = AutoModelForCausalLM.from_pretrained(
            gemma_id,
            dtype=torch.float32,
            attn_implementation = "eager",
        )

        self.vit = AutoModel.from_pretrained(
            vit_id,
            dtype=torch.float32,
            attn_implementation = "eager",
        )

        gemma_embed_size =  self.gemma.get_input_embeddings().weight.shape[1]
        vit_embed_size = self.vit.config.hidden_size
        

        self.adapter = nn.Sequential(
            nn.LayerNorm(vit_embed_size),
            nn.Linear(vit_embed_size, gemma_embed_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(gemma_embed_size, gemma_embed_size)
        ) # from internVL3

        for parameter in self.vit.parameters():
            parameter.requires_grad = False
        for parameter in self.gemma.parameters():
            parameter.requires_grad = not freeze_gemma
        for parameter in self.adapter.parameters():
            parameter.requires_grad = True                             

        self.register_buffer("_embedded_prompt", None, persistent=False)
        self.register_buffer("_boi_embed", None, persistent=False)
        self.register_buffer("_eoi_embed", None, persistent=False)
        self.register_buffer("_bos_embed", None, persistent=False)

    def _select_vit_tokens(self, vit_out):
        """
        select the specific tokens to feed into LM
        """
        num_patches = vit_out.shape[1] - 5
        
        assert num_patches > 0, "vit output is not correct, check images input"

        vit_output_mask = ([self.include_cls] + 
                           [self.include_registers] * 4 + 
                           [self.include_patches] * num_patches)
        vit_output_mask = torch.tensor(vit_output_mask, dtype=torch.bool, device=vit_out.device)
        
        return vit_out[:, vit_output_mask, :]

    def forward(self, images, captions):
        """
        :param images: Tensors outputted by AutoImageProcessor
        :param captions: list[str] accepeted by AutoTokenizer
        
        :returns outputs: outputs of the model
        """
        #set device
        device = next(self.adapter.parameters()).device

        if self._boi_embed is None:
            self._boi_embed = (self.gemma.get_input_embeddings()(torch.tensor([self.gemma_tokenizer.boi_token_id], device=device, dtype=torch.long))).unsqueeze(0)
        if self._eoi_embed is None:
            self._eoi_embed = (self.gemma.get_input_embeddings()(torch.tensor([self.gemma_tokenizer.eoi_token_id], device=device, dtype=torch.long))).unsqueeze(0)
        if self._bos_embed is None:
            self._bos_embed = (self.gemma.get_input_embeddings()(torch.tensor([self.gemma_tokenizer.bos_token_id], device=device, dtype=torch.long))).unsqueeze(0)
        if self._embedded_prompt is None:
            self._embedded_prompt = (self.gemma.get_input_embeddings()(self.gemma_tokenizer(self.prompt, return_tensors="pt").to(device).input_ids))


        with torch.no_grad():
            vit_out = self.vit(images).last_hidden_state
        
        vit_selected = self._select_vit_tokens(vit_out)

        #convert for Language model
        image_embed = self.adapter(vit_selected)

        boi = self._boi_embed.expand(image_embed.shape[0], 1, image_embed.shape[2])
        eoi = self._eoi_embed.expand(image_embed.shape[0], 1, image_embed.shape[2])
        image_embed = torch.cat([boi, image_embed, eoi], dim=1)

        # append <bos> <<prompt>> to <boi> <img>...<img> <eoi> 
        bos_embed = self._bos_embed.expand(image_embed.shape[0], 1, image_embed.shape[2])
        prompt_embed = self._embedded_prompt.expand(image_embed.shape[0], -1, image_embed.shape[2])

        image_embed = torch.cat([image_embed, bos_embed, prompt_embed], dim=1)

        #tokenize captions
        caps = self.gemma_tokenizer(captions, 
                                    return_tensors="pt", 
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.max_caption_length)
        caption_ids = caps["input_ids"].to(device)
        attn_mask = caps["attention_mask"].to(device)

        caption_embeds = self.gemma.get_input_embeddings()(caption_ids)

        new_embeds = torch.cat([image_embed, caption_embeds], dim = 1)
        new_mask = torch.cat([torch.ones(image_embed.shape[0], image_embed.shape[1], device = device, dtype=torch.long), attn_mask], dim = 1)

        labels = torch.full((new_embeds.shape[0], new_embeds.shape[1]), -100, dtype=torch.long, device=device)
        labels[:, image_embed.shape[1]:] = caption_ids
        if (labels != -100).sum().item() == 0:
            raise RuntimeError("All targets are -100 (no supervision) — check tokenizer/padding.")


        #return outputs
        outputs = self.gemma(inputs_embeds=new_embeds, attention_mask=new_mask, labels=labels)

        return outputs

    def inference_generate(self, images, max_new_tokens=30, temperature = 0.0):
        """
        generate captions for a batch of images.

        :param images: tensor generated by AutoImageProcessor

        :returns gen: generated text 
        """
        device = next(self.adapter.parameters()).device

        if self._boi_embed is None:
            self._boi_embed = (self.gemma.get_input_embeddings()(torch.tensor([self.gemma_tokenizer.boi_token_id], device=device, dtype=torch.long))).unsqueeze(0)
        if self._eoi_embed is None:
            self._eoi_embed = (self.gemma.get_input_embeddings()(torch.tensor([self.gemma_tokenizer.eoi_token_id], device=device, dtype=torch.long))).unsqueeze(0)
        if self._bos_embed is None:
            self._bos_embed = (self.gemma.get_input_embeddings()(torch.tensor([self.gemma_tokenizer.bos_token_id], device=device, dtype=torch.long))).unsqueeze(0)
        if self._embedded_prompt is None:
            self._embedded_prompt = (self.gemma.get_input_embeddings()(self.gemma_tokenizer(self.prompt, return_tensors="pt").to(device).input_ids))


        with torch.no_grad():
            vit_out = self.vit(images).last_hidden_state

        vit_selected = self._select_vit_tokens(vit_out)

        #convert for Language model
        image_embed = self.adapter(vit_selected)

        boi = self._boi_embed.expand(image_embed.shape[0], 1, image_embed.shape[2])
        eoi = self._eoi_embed.expand(image_embed.shape[0], 1, image_embed.shape[2])
        
        image_embed = torch.cat([boi, image_embed, eoi], dim=1)

        bos_embed = self._bos_embed.expand(image_embed.shape[0], 1, image_embed.shape[2])
        prompt_embed = self._embedded_prompt.expand(image_embed.shape[0], 1, image_embed.shape[2])

        seq_embeds = torch.cat([image_embed, bos_embed, prompt_embed], dim=1)
        attention_mask = torch.ones(seq_embeds.shape[0], seq_embeds.shape[1], dtype=torch.long, device=device)

        gen = self.gemma.generate(
            inputs_embeds=seq_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature>0),
            temperature=temperature,
            eos_token_id=self.gemma_tokenizer.eos_token_id,
            pad_token_id=self.gemma_tokenizer.pad_token_id,
        )

        gen = gen[:, attention_mask.shape[1]:]

        result = self.gemma_tokenizer.batch_decode(gen, skip_special_tokens=True)

        return result








        