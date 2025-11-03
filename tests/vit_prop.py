import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


from models.vision_encoder import build_vit_and_transform

vit, preprocessor = build_vit_and_transform(resize_size=224, device="cuda")

import torchinfo

torchinfo.summary(vit)

print("-" * 150)

import torch

image = torch.rand(1, 3, 512, 512)

print(f"original batched image shape: {image.shape}")

t_image = preprocessor(image).cuda()

print(f"transformed batched image shape: {t_image.shape}")

outputs = vit(t_image, is_training=True)

print(f"image embedding output VIT:")
print(f"CLS : \n{outputs["x_norm_clstoken"].shape}")
print(f"REG : \n{outputs["x_storage_tokens"].shape}")
print(f"PATCH: \n{outputs["x_norm_patchtokens"].shape}")