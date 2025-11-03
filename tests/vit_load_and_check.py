import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from models.vision_encoder import build_vit_and_transform

model, preprocessor = build_vit_and_transform(vit_type="DINOV3_ViT_S_16_PLUS", resize_size=224)

import torch

img = torch.rand(3,3,224,224)

t_img = preprocessor(img)

outputs : torch.Tensor = model(t_img)

print(outputs.shape, outputs.type(), sep=" || ")