import torch
import torch.nn as nn

class MLPAdapter(nn.Module):
    def __init__(self, vit_embed_size, lm_embed_size):
        super().__init__()

        self.vit_embed_size = vit_embed_size
        self.lm_embed_size = lm_embed_size

class InternVL3Adapter(MLPAdapter):
    def __init__(self, vit_embed_size, lm_embed_size):
        super().__init__(vit_embed_size, lm_embed_size)

        self.adapter  = nn.Sequential(
            nn.LayerNorm(vit_embed_size),
            nn.Linear(vit_embed_size, lm_embed_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(lm_embed_size, lm_embed_size),
        )

    def forward(self, x):
        return self.adapter(x)
        