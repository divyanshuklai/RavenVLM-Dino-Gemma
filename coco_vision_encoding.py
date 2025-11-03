"""
This script converts the COCO datasets to vision encodings using DINOV3-VITS+/16 on modal
1. downloads and saves datasets from hf to modal volume cache ./datasets/COCO-Captions-raw
2. uses the train, test, and validation datasets under COCO-Captions-raw to build datasets for each split under ./datasets/COCO-captions-vits16plus-embed/
"""
import os
import modal


CACHE_VOL = modal.Volume.from_name("cache")


image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install(
        "torch>=2.8.0",
        "huggingface-hub[cli]>=0.34.4",
        "datasets>=4.0.0",
        "torchvision>=0.23.0"
    )
)

project_root = os.path.dirname(__file__)

image.add_local_dir(
    local_path = f"{project_root}/models",
    remote_path = "/models"
)

image.add_local_dir(
    local_path= f"{project_root}/data",
    remote_path= "/data",
)

app = modal.App("VISION-DATASET-ENCODER")

@app.function(
    image = image,
    volumes={"/cache" : CACHE_VOL},
    timeout = 60 * 60 * 10,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_dataset_and_encode(
    split : str,
):
    # init model
    from models.vision_encoder import build_vit_and_transform
    vision_encoder, transform = build_vit_and_transform(
        vit_type="DINOV3_ViT_S_16_PLUS",
        cache_file="/cache/models/models--facebookresearch--dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        do_resize=False,
    )
    # init dataloader
    from data.COCOCaptions import make_coco_raw_dataloader
    dl = make_coco_raw_dataloader(
        split=split,
        transform=transform,
        all_captions=True,
        cache_dir="/cache/datasets/COCO-Captions-raw",
        batch_size=1,
        shuffle=False,
    ) 
    # init save file and config
    import math, h5py, numpy as np, torch
    out_dir = "/cache/datasets/COCO-captions-vits16plus-embed"
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, f"{split}.h5")
    patch_size = 16

    # start process
    with h5py.File(h5_path, "w") as f, torch.no_grad():
        str_dt = h5py.string_dtype(encoding="utf-8")
        for batch in dl:
            id = batch["ids"][0]
            img = batch["images"][0]
            caps = batch["captions"][0]

            x = img.unsqueeze(0)
            embeds : dict = vision_encoder(x, is_training=True)

            C, H, W = img.shape
            CLS = embeds["x_norm_clstoken"].unsqueeze(1).numpy().astype(np.float32) # (B, 1, 384)
            REG  = embeds["x_storage_tokens"].numpy().astype(np.float32) # (B, 4, 384)
            PATCH = embeds["x_norm_patchtokens"].numpy().astype(np.float32) # (B, T, 384) T = HW / P^2

            grp = f.create_group(str(id))
            grp.create_dataset("id", data=id)
            grp.create_dataset("cls", data=CLS)
            grp.create_dataset("reg", data=REG)
            grp.create_dataset("patch", data=PATCH)
            grp.create_dataset("captions", data=np.array(caps, dtype=object), dtype=str_dt)

            grp.create_dataset("image_height", data=H)
            grp.create_dataset("image_width", data=W)
        f.flush()

@app.local_entrypoint()
def sweep():
    splits=["train", "test", "validation"]

    download_dataset_and_encode.map(splits)
