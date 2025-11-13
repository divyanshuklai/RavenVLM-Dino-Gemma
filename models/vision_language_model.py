# vision_language_model.py
import torch
import torch.nn as nn
from typing import Optional, List

from .qformer import QFormer
from .language_model import Gemma3Model

class VisualPrefixAdapter(nn.Module):
    """Project QFormer queries -> Gemma embedding dim (simple linear adapter)."""
    def __init__(self, qformer_dim: int, gemma_emb_dim: int, use_ln: bool = True):
        super().__init__()
        self.proj = nn.Linear(qformer_dim, gemma_emb_dim, bias=False)
        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(gemma_emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        if self.use_ln:
            out = self.ln(out)
        return out

class VLMQFormer(nn.Module):
    """
    Two-stage VLM wrapper.
    """
    def __init__(
        self,
        qformer: QFormer,
        gemma: Optional[Gemma3Model] = None,
        adapter: Optional[VisualPrefixAdapter] = None,
        freeze_gemma: bool = True,
        stage: str = "stage1",
    ):
        super().__init__()
        assert stage in ("stage1", "stage2")
        self.qformer = qformer
        self.gemma = gemma
        self.vit = None
        self.stage = stage

        q_dim = qformer.embed_dim
        gemma_dim = gemma.cfg["emb_dim"] if gemma is not None else None

        if adapter is None and gemma is not None:
            adapter = VisualPrefixAdapter(q_dim, gemma_dim, use_ln=True)
        self.adapter = adapter

        if self.gemma is not None:
            for p in self.gemma.parameters():
                p.requires_grad = not freeze_gemma

    def save_stage1(self, path: str) -> None:
        payload = {
            "qformer": self.qformer.state_dict(),
            "adapter": self.adapter.state_dict() if self.adapter is not None else None,
        }
        torch.save(payload, path)
        print(f"Saved stage1 checkpoint -> {path}")

    def load_stage1(self, path: str, strict: bool = True) -> None:
        data = torch.load(path, map_location="cpu")
        if "qformer" in data:
            self.qformer.load_state_dict(data["qformer"], strict=strict)
        if "adapter" in data and data["adapter"] is not None:
            if self.adapter is None:
                if self.gemma is None:
                    raise RuntimeError("Cannot instantiate adapter because Gemma not attached yet.")
                self.adapter = VisualPrefixAdapter(self.qformer.embed_dim, self.gemma.cfg["emb_dim"])
            self.adapter.load_state_dict(data["adapter"], strict=strict)
        print(f"Loaded stage1 checkpoint <- {path}")

    def attach_vit(self, vit_model: nn.Module, freeze_vit: bool = True) -> None:
        self.vit = vit_model
        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False
        print("ViT attached; freeze_vit=", freeze_vit)

    def forward(
        self,
        vis_emb: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        patch_attention_mask: Optional[torch.Tensor] = None,
        mode: str = "stage2",
    ):
        if vis_emb is None:
            raise ValueError("vis_emb must be provided to forward().")

        queries_tuple = self.qformer(vis_emb, use_text=False, patch_attn_mask=patch_attention_mask)
        queries = queries_tuple[0]

        if self.gemma is None:
            raise RuntimeError("Gemma LM not attached.")
        if input_ids is None:
            raise ValueError("stage2 requires input_ids (text tokens)")

        prefix_emb = self.adapter(queries)
        scale = (self.gemma.cfg["emb_dim"] ** 0.5)
        prefix_emb = prefix_emb * scale

        tok_emb = self.gemma.tok_emb(input_ids) * scale
        x = torch.cat([prefix_emb, tok_emb], dim=1)

        cur_len = x.shape[1]
        device = x.device
        mask_global, mask_local = self.gemma._create_masks(cur_len, device, pos_start=0, pos_end=cur_len)

        for i, block in enumerate(self.gemma.blocks):
            x, _ = block(x, mask_global=mask_global, mask_local=mask_local, cos_global=self.gemma.cos_global, sin_global=self.gemma.sin_global, cos_local=self.gemma.cos_local, sin_local=self.gemma.sin_local, start_pos=0, cache=None)

        x = self.gemma.final_norm(x)
        logits = self.gemma.out_head(x.to(self.gemma.cfg["dtype"]))
        return logits

    @torch.no_grad()
    def generate(self,
                 image: Optional[torch.Tensor] = None,
                 vis_emb: Optional[torch.Tensor] = None,
                 patch_attention_mask: Optional[torch.Tensor] = None,
                 tokenizer = None,
                 max_new_tokens: int = 32,
                 eos_token_id: Optional[int] = None,
                 device: Optional[str] = None,
    ) -> List[str]:
        assert (image is not None) or (vis_emb is not None), "Provide image or precomputed vis_emb."
        assert tokenizer is not None, "Please pass a tokenizer instance."

        if device is None:
            device = next(self.parameters()).device

        if vis_emb is None:
            assert self.vit is not None, "No ViT attached."
            self.vit.eval()
            vis_emb = self.vit(image.to(device))

        self.qformer.eval()
        self.adapter.eval()
        self.gemma.eval()

        q_tuple = self.qformer(vis_emb.to(device), use_text=False, patch_attn_mask=patch_attention_mask)
        queries = q_tuple[0]
        prefix_emb = self.adapter(queries) * (self.gemma.cfg["emb_dim"] ** 0.5)
        
        batch_size = prefix_emb.shape[0]
        
        input_ids = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        for _ in range(max_new_tokens):
            tok_emb = self.gemma.tok_emb(input_ids) * (self.gemma.cfg["emb_dim"] ** 0.5)
            x = torch.cat([prefix_emb, tok_emb], dim=1)
            
            cur_len = x.shape[1]
            mask_global, mask_local = self.gemma._create_masks(cur_len, device, pos_start=0, pos_end=cur_len)

            for i, block in enumerate(self.gemma.blocks):
                x, _ = block(x, mask_global=mask_global, mask_local=mask_local, cos_global=self.gemma.cos_global, sin_global=self.gemma.sin_global, cos_local=self.gemma.cos_local, sin_local=self.gemma.sin_local, start_pos=0, cache=None)

            x = self.gemma.final_norm(x)
            logits = self.gemma.out_head(x.to(self.gemma.cfg["dtype"]))
            
            next_logits = logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if eos_token_id is not None:
                newly_finished = (next_token.view(-1) == eos_token_id)
                unfinished_sequences.mul_((~newly_finished).long())

            if unfinished_sequences.max() == 0:
                break
        
        generated_texts = tokenizer.batch_decode(input_ids.cpu().tolist(), skip_special_tokens=True)
        
        return generated_texts