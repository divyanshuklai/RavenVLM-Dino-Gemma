"""
NOTE : MOVE THIS SCRIPT TO PROJECT ROOT TO USE

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
    .pip_install_from_pyproject("pyproject.toml")
)

project_root = os.path.dirname(__file__)

image = image.add_local_dir(
    local_path = f"{project_root}/models",
    remote_path = "/root/models"
)

image = image.add_local_dir(
    local_path= f"{project_root}/data",
    remote_path= "/root/data",
)

app = modal.App("VISION-DATASET-ENCODER")

@app.function(
    image = image,
    volumes={"/cache" : CACHE_VOL},
    timeout = 60 * 60 * 10,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G",
)
def download_dataset_and_encode(
    split : str
):
    # init model
    import torch
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

    from models.vision_encoder import build_vit_and_transform
    vision_encoder, transform = build_vit_and_transform(
        vit_type="DINOV3_ViT_S_16_PLUS",
        cache_file="/cache/models/models--facebookresearch--dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        do_resize=True,
        device=device
    )
    # vision_encoder.compile()
    # init dataloader
    from data.COCOCaptions import COCOCaptionsDatasetRAW, make_coco_raw_collate_fn

    dataset = COCOCaptionsDatasetRAW(
        split=split,
        transform=transform,
        all_captions=True,
        seed=42,
        cache_dir="/cache/datasets/COCO-Captions-raw",
    )
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=make_coco_raw_collate_fn(same_size=True),
        persistent_workers=True,
    ) 
    # init save file and config
    import h5py, numpy as np
    from tqdm.auto import tqdm
    out_dir = f"/cache/datasets/COCO-captions-vits16plus-embed-same-size/{split}"
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, f"{split}_crop_224.h5")
    patch_size = 16

    # total samples (assumption: dataset.__len__() is correct)
    N = len(dataset)

    with h5py.File(h5_path, "w") as f, torch.no_grad():
        str_dt = h5py.string_dtype(encoding="utf-8")

        # create datasets once (shapes based on your assumptions)
        # CLS: (N, 1, 384) ; REG: (N, 4, 384) ; PATCH: (N, 196, 384)
        d_id = f.create_dataset("id", shape=(N,), dtype=str_dt)              # store ids as strings (safe)
        d_cls = f.create_dataset("cls", shape=(N, 1, 384), dtype=np.float32)
        d_reg = f.create_dataset("reg", shape=(N, 4, 384), dtype=np.float32)
        d_patch = f.create_dataset("patch", shape=(N, 196, 384), dtype=np.float32)  # T=196 per your assumption
        d_captions = f.create_dataset("captions", shape=(N, 5), dtype=str_dt)  # each sample has 5 strings

        write_idx = 0
        for batch in tqdm(dl,
                          total=len(dl),
                          desc=f"[{split.upper()}]", 
                          dynamic_ncols=True,
                          mininterval=1.0,
                          leave=False):
            ids = batch["ids"]            # expected: list/iterable of ids
            imgs = batch["images"]        # tensor (B, C, H, W)
            caps = batch["captions"]      # expected: list of lists, shape (B, 5)

            B = len(ids)

            # move to device and run encoder in eval mode
            x = imgs.to(device, non_blocking=True)
            embeds: dict = vision_encoder(x, is_training=True)

            
            CLS = embeds["x_norm_clstoken"].unsqueeze(1).cpu().numpy().astype(np.float32)  # (B,1,384)
            REG = embeds["x_storage_tokens"].cpu().numpy().astype(np.float32)              # (B,4,384)
            PATCH = embeds["x_norm_patchtokens"].cpu().numpy().astype(np.float32)         # (B,T,384)

            # sanity: expected T == 196
            assert PATCH.shape[1] == 196, f"Unexpected patch token count {PATCH.shape[1]}, expected 196"

            # write into datasets at the right slice
            d_id[write_idx: write_idx + B] = np.array([str(i) for i in ids], dtype=object)
            d_cls[write_idx: write_idx + B, :, :] = CLS
            d_reg[write_idx: write_idx + B, :, :] = REG
            d_patch[write_idx: write_idx + B, :, :] = PATCH
            # captions: caps must be shape (B,5)
            for i, caplist in enumerate(caps):
                d_captions[write_idx + i, :] = [str(caplist[i]) for i in range(5)]

            write_idx += B

        f.flush()

@app.local_entrypoint()
def sweep():
    splits = ["train", "validation", "test"]   
    list(download_dataset_and_encode.map(splits))
