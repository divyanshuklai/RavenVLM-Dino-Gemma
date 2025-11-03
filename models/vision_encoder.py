import torch
import torch.nn as nn
from torchvision.transforms import v2

from models.dino_model_paths import MODELS #DinoV3 URLs, use your own.

def build_vit_and_transform(
    vit_type = "DINOV3_ViT_S_16_PLUS",
    cache_file = None,
    do_resize = False,
    resize_size = 224,
    device = "cpu",
):
    """
    Build a DinoV3 and its preprocessing transform.

    loads from cache_path if given.
    resize_size is passed on to ViT transform.
        
    Uses github repo for model definition https://github.com/facebookresearch/dinov3
    """
    import os

    if vit_type not in MODELS:
        raise ValueError(f"Unsupported model type : {vit_type}")
    
    if cache_file is None:
        weights = MODELS[vit_type]["weights"]
    elif os.path.exists(cache_file):
        weights = cache_file
    else:
        raise ValueError("Weights file does not exist!")
    
    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3:main",
        model=MODELS[vit_type]["model"],
        weights=weights
    )

    model.eval()
    model = model.to(device)
    transform = make_transform(do_resize, resize_size)

    return model, transform


def make_transform(do_resize : bool, resize_size : int):
    to_tensor = v2.ToTensor()
    resize = v2.Resize((resize_size, resize_size), antialias=True) if do_resize else v2.Identity()
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])
    