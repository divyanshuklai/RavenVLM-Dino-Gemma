import torch
import torch.nn as nn
import torch.nn.functional as F

from .qformer import QFormer

class QFormerForPretraining(nn.Module):
    """
    A wrapper class for the Q-Former model to handle the three pre-training objectives
    (ITC, ITM, ITG) as described in the BLIP-2 paper.

    This class encapsulates the multi-pass logic required for pre-training and is
    intended to be used only in Stage 1. After training, the internal `qformer`
    can be extracted and used for downstream tasks.
    """
    def __init__(self, qformer: QFormer, gemma_vocab_size: int, projection_dim: int = 256):
        super().__init__()
        self.qformer = qformer
        self.qformer_hidden_dim = self.qformer.embed_dim
        
        # --- Heads for Loss Calculation ---
        self.vision_proj = nn.Linear(self.qformer_hidden_dim, projection_dim, bias=False)
        self.text_proj = nn.Linear(self.qformer_hidden_dim, projection_dim, bias=False)
        self.itm_head = nn.Linear(self.qformer_hidden_dim, 2)
        self.itg_head = nn.Linear(self.qformer_hidden_dim, gemma_vocab_size, bias=False)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # ln(1/0.07)

    def forward(self, vit_embeds: torch.Tensor, text_input_ids: torch.LongTensor, text_attention_mask: torch.LongTensor, patch_attention_mask: torch.Tensor):
        """
        Performs the forward passes required for all three pre-training losses.

        Args:
            vit_embeds (torch.Tensor): Pre-computed ViT patch embeddings.
            text_input_ids (torch.LongTensor): Text tokens for the batch.
            text_attention_mask (torch.LongTensor): Attention mask for the text tokens.
            patch_attention_mask (torch.Tensor): Boolean attention mask for the ViT patch embeddings.

        Returns:
            dict: A dictionary containing the 'itc_loss', 'itm_loss', and 'itg_loss'.
        """
        device = vit_embeds.device
        
        # --- 1. ITC (Image-Text Contrastive) Loss ---
        # Get image features from a unimodal pass (no text).
        # The low-level QFormer API returns a tuple, so we unpack it.
        image_feats, = self.qformer(vit_embeds, use_text=False, patch_attn_mask=patch_attention_mask)
        
        # Get text features from a separate multimodal pass.
        # We use a bidirectional mask here to get a global text representation.
        _, text_feats_itc = self.qformer(vit_embeds, use_text=True, text_tok=text_input_ids, mask_type="bidirectional-multimodal", patch_attn_mask=patch_attention_mask)

        image_embeds = F.normalize(self.vision_proj(image_feats[:, 0, :]), dim=-1)
        text_embeds = F.normalize(self.text_proj(text_feats_itc[:, 0, :]), dim=-1)
        
        sim_q2t = torch.matmul(image_embeds.float(), text_embeds.t().float()) * self.logit_scale.exp()
        sim_t2q = torch.matmul(text_embeds.float(), image_embeds.t().float()) * self.logit_scale.exp()
        
        labels = torch.arange(image_embeds.size(0), device=device)
        loss_itc = (F.cross_entropy(sim_q2t, labels) + F.cross_entropy(sim_t2q, labels)) / 2

        # --- 2. ITM (Image-Text Matching) Loss ---
        # Create negative pairs by rolling the tensors.
        with torch.no_grad():
            batch_size = vit_embeds.size(0)
            text_input_ids_neg = torch.roll(text_input_ids, shifts=1, dims=0)
            text_attention_mask_neg = torch.roll(text_attention_mask, shifts=1, dims=0)

        # Prepare a combined batch for one forward pass.
        vit_embeds_itm = torch.cat([vit_embeds, vit_embeds], dim=0)
        patch_attention_mask_itm = torch.cat([patch_attention_mask, patch_attention_mask], dim=0)
        text_ids_itm = torch.cat([text_input_ids, text_input_ids_neg], dim=0)
        
        # Run a single multimodal pass for both positive and negative pairs.
        multimodal_feats_itm, _ = self.qformer(vit_embeds_itm, use_text=True, text_tok=text_ids_itm, mask_type="bidirectional-multimodal", patch_attn_mask=patch_attention_mask_itm)
        
        # Use the first query token as the representation for matching.
        itm_query_feats = multimodal_feats_itm[:, 0, :]
        itm_logits = self.itm_head(itm_query_feats)
        
        # Create labels: 1 for positive pairs, 0 for negative pairs.
        itm_labels = torch.cat([
            torch.ones(batch_size, dtype=torch.long),
            torch.zeros(batch_size, dtype=torch.long)
        ], dim=0).to(device)
        
        loss_itm = F.cross_entropy(itm_logits.float(), itm_labels)

        # --- 3. ITG (Image-Text Generation) Loss ---
        # This is a masked language modeling task conditioned on the image.
        # We use a causal mask to prevent queries from attending to future text tokens.
        _, text_feats_itg = self.qformer(vit_embeds, use_text=True, text_tok=text_input_ids, mask_type="causal-multimodal", patch_attn_mask=patch_attention_mask)
        
        lm_logits = self.itg_head(text_feats_itg)
        
        loss_itg = F.cross_entropy(
            lm_logits.view(-1, self.qformer.lm_vocab_size).float(),
            text_input_ids.view(-1),
            ignore_index=0  # Assuming 0 is the padding token ID
        )
        
        return {
            'itc_loss': loss_itc,
            'itm_loss': loss_itm,
            'itg_loss': loss_itg,
        }