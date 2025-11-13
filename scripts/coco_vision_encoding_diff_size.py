"""
This script converts the COCO datasets to vision encodings for VARIABLE-SIZED images
using DINOV3-VITS+/16 on Modal.

It is adapted from coco_vision_encoding.py to handle variable image sizes by:
1. Not resizing images to a fixed crop.
2. Saving each sample's data into a separate HDF5 group to accommodate
   variable-length patch embeddings.
3. Sharding the training set into multiple files for manageability.
"""
import os
import modal
import h5py
import numpy as np
from tqdm.auto import tqdm
import torch

# --- Modal Setup ---

CACHE_VOL = modal.Volume.from_name("cache")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install_from_pyproject("pyproject.toml")
)

project_root = os.path.dirname(__file__)

# Add project directories to the image
image = image.add_local_dir(
    local_path=f"{project_root}/../models", remote_path="/root/models"
)
image = image.add_local_dir(
    local_path=f"{project_root}/../data", remote_path="/root/data"
)

app = modal.App("VISION-DATASET-ENCODER-DIFF-SIZE")

# --- Main Encoding Function ---

@app.function(
    image=image,
    volumes={"/cache": CACHE_VOL},
    timeout=60 * 60 * 12,  # 12 hours, as this is slower
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G",
)
def download_dataset_and_encode(split: str, shard_size: int = 10000):
    """
    Downloads a split of the COCO dataset, encodes images using DINOv3,
    and saves the embeddings to HDF5 files.

    Args:
        split (str): The dataset split to process ("train", "validation", or "test").
        shard_size (int): Number of samples per shard file for the training split.
    """
    # 1. Initialize Model and Transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.vision_encoder import build_vit_and_transform

    # Build transform WITHOUT resizing to a fixed crop
    vision_encoder, transform = build_vit_and_transform(
        vit_type="DINOV3_ViT_S_16_PLUS",
        cache_file="/cache/models/models--facebookresearch--dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        do_resize=False,  # Key change: process variable-sized images
        device=device,
    )

    # 2. Initialize DataLoader
    from data.COCOCaptions_raw import COCOCaptionsDatasetRAW, make_coco_raw_collate_fn

    dataset = COCOCaptionsDatasetRAW(
        split=split,
        transform=transform,
        all_captions=True,
        seed=42,
        cache_dir="/cache/datasets/COCO-Captions-raw",
    )
    
    # Use collate_fn for variable sizes
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,  # A small batch size is fine
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=make_coco_raw_collate_fn(same_size=False),
        persistent_workers=True,
    )

    # 3. Prepare Output Directory and HDF5 File Handling
    out_dir = "/cache/datasets/COCO-captions-vits16plus-embed"
    os.makedirs(out_dir, exist_ok=True)
    
    str_dt = h5py.string_dtype(encoding="utf-8")
    total_samples = len(dataset)
    samples_processed = 0
    
    pbar = tqdm(total=total_samples, desc=f"[{split.upper()}]", dynamic_ncols=True)

    # --- Sharding Logic ---
    file_handle = None
    shard_index = 0

    def get_h5_file(sample_index):
        nonlocal file_handle, shard_index
        
        if split != 'train':
            # Validation and test sets use a single file
            if file_handle is None:
                file_handle = h5py.File(os.path.join(out_dir, f"{split}.h5"), "w")
            return file_handle
        else:
            # Training set is sharded
            current_shard_index = sample_index // shard_size
            if file_handle is None or current_shard_index != shard_index:
                if file_handle:
                    file_handle.close()
                shard_index = current_shard_index
                shard_path = os.path.join(out_dir, f"train_shard-{shard_index}.h5")
                print(f"Creating new shard: {shard_path}")
                file_handle = h5py.File(shard_path, "w")
            return file_handle

    # 4. Processing Loop
    with torch.no_grad():
        for batch in dl:
            # The collate function gives us lists of tensors
            ids = batch["ids"]
            images = batch["images"]
            captions_batch = batch["captions"]

            # Process each item in the batch individually
            for i in range(len(ids)):
                sample_id = str(ids[i])
                img_tensor = images[i]
                captions = captions_batch[i]

                # Get the correct HDF5 file (shard or single file)
                f = get_h5_file(samples_processed)

                # Create a group for the sample
                group = f.create_group(sample_id)

                # Run encoder on the single image. is_training=True ensures a dict is returned.
                # Add batch dimension, move to device
                x = img_tensor.unsqueeze(0).to(device, non_blocking=True)
                embeds: dict = vision_encoder(x, is_training=True)

                # Extract embeddings and move to CPU
                # Handle cases where the model squeezes the batch dimension for B=1
                cls_token = embeds["x_norm_clstoken"]
                if cls_token.dim() == 1:
                    cls_token = cls_token.unsqueeze(0) # (D,) -> (1, D)
                cls_embed = cls_token.unsqueeze(1).cpu().numpy().astype(np.float32) # (1, D) -> (1, 1, D)
                patch_embeds = embeds["x_norm_patchtokens"].cpu().numpy().astype(np.float32)

                # Save datasets within the group
                group.create_dataset("cls", data=cls_embed, dtype=np.float32)
                group.create_dataset("patch", data=patch_embeds, dtype=np.float32)
                group.create_dataset("captions", data=np.array(captions, dtype=object), dtype=str_dt)
                
                samples_processed += 1
                pbar.update(1)

    # Clean up
    if file_handle:
        file_handle.close()
    pbar.close()
    print(f"Finished processing split '{split}'.")


@app.local_entrypoint()
def sweep():
    """Runs the encoding for all dataset splits."""
    splits = ["train", "validation", "test"]
    # Run sequentially to avoid overwhelming the system
    for split in splits:
        download_dataset_and_encode.remote(split)

