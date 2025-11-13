# data/COCOCaptions.py

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import json
import random
from pathlib import Path
import logging
import numpy as np
from typing import List, Dict, Tuple

# Setup logging
logger = logging.getLogger(__name__)

class CocoEmbedDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-computed ViT embeddings from COCO Captions.
    
    Handles two formats:
    1. 'same_size': Assumes a single HDF5 file per split (train.h5, validation.h5).
    2. 'diff_size': Assumes sharded HDF5 files for train split (train_shard-*.h5)
       and single files for validation/test.
    """
    
    def __init__(self, data_path: str, split: str, tokenizer, max_text_len: int, same_size: bool = False, deterministic_caption: bool = True, seed: int = 42):
        self.data_path = Path(data_path)
        self.split = split
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.same_size = same_size
        self.deterministic_caption = deterministic_caption
        self.seed = seed
        self.rng = random.Random(seed) # For deterministic caption sampling

        self.is_sharded_train = (split == 'train' and not same_size)
        self.h5_files: Dict[str, h5py.File] = {} # To store open file handles if needed
        
        if self.is_sharded_train:
            # Discover shards and build sample-to-shard mapping dynamically
            self.sample_id_to_shard_path = {}
            self.sample_ids = []
            
            shard_files = sorted(list(self.data_path.glob('train_shard-*.h5')))
            if not shard_files:
                raise FileNotFoundError(f"No training shard files found matching 'train_shard-*.h5' in {self.data_path}")

            for shard_path in shard_files:
                try:
                    with h5py.File(shard_path, 'r') as f:
                        shard_sample_ids = list(f.keys())
                        self.sample_ids.extend(shard_sample_ids)
                        for sample_id in shard_sample_ids:
                            self.sample_id_to_shard_path[sample_id] = shard_path
                except OSError as e:
                    logger.warning(f"Could not read shard file {shard_path}: {e}")
            
            self.sample_ids = sorted(self.sample_ids)
                    
        else:
            # "same_size" dataset (all splits) or "diff_size" (val/test)
            # Support multiple common file layouts:
            #  1) <root>/<split>.h5
            #  2) <root>/<split>_crop_224.h5
            #  3) <root>/<split>/<split>.h5
            #  4) <root>/<split>/<split>_crop_224.h5
            candidates = [
                self.data_path / f"{split}.h5",
                self.data_path / f"{split}_crop_224.h5",
                self.data_path / split / f"{split}.h5",
                self.data_path / split / f"{split}_crop_224.h5",
            ]
            chosen = None
            for p in candidates:
                if p.exists():
                    chosen = p
                    break
            if chosen is None:
                raise FileNotFoundError(
                    f"Could not find any of expected files for split='{split}' under {self.data_path}. "
                    f"Tried: {[str(p) for p in candidates]}"
                )
            self.h5_file_path = chosen

            # Load all sample IDs (group keys) into memory
            with h5py.File(self.h5_file_path, 'r') as f:
                if self.same_size:
                    # For same_size, IDs are in a dataset called 'id'
                    self.sample_ids = [sid.decode('utf-8') for sid in f['id'][:]]
                else:
                    # For diff_size, sample IDs are the group keys
                    self.sample_ids = sorted(list(f.keys()))
            logger.info(f"Using H5 file for split '{split}': {self.h5_file_path}")
            
        logger.info(f"Loaded {split} split ({'same_size' if same_size else 'diff_size'}) with {len(self.sample_ids)} samples.")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        sample_id = self.sample_ids[idx]
        
        if self.is_sharded_train:
            file_path = self.sample_id_to_shard_path.get(sample_id)
            if file_path is None or not file_path.exists():
                # This should ideally not happen if __init__ ran correctly
                raise RuntimeError(f"Could not find shard file for sample {sample_id}")

            with h5py.File(file_path, 'r') as f:
                group = f[sample_id]
                patch_embeds = torch.from_numpy(group['patch'][:]).squeeze(0) # Shape: (N_patches, D_vit)
                cls_embeds = torch.from_numpy(group['cls'][:]) # Shape: (1, 1, D_vit)
                captions_bytes = group['captions'][:]
        else:
            # Need to re-open file in __getitem__ for multiprocessing
            with h5py.File(self.h5_file_path, 'r') as f:
                if self.same_size:
                    # same_size: data is in top-level datasets, indexed by `idx`
                    patch_embeds = torch.from_numpy(f['patch'][idx])
                    cls_embeds = torch.from_numpy(f['cls'][idx])
                    captions_bytes = f['captions'][idx]
                else:
                    # diff_size (val/test): data is in groups per sample
                    group = f[sample_id]
                    patch_embeds = torch.from_numpy(group['patch'][:]).squeeze(0)
                    cls_embeds = torch.from_numpy(group['cls'][:])
                    captions_bytes = group['captions'][:]

        # Decode captions
        raw_captions = [c.decode('utf-8') for c in captions_bytes]
        
        # Select caption
        if self.deterministic_caption:
            # Use instance-specific RNG seeded with sample index
            # This ensures same caption per sample, but random across samples
            caption_rng = random.Random(self.seed + idx)
            chosen_caption = caption_rng.choice(raw_captions)
        else:
            # Use global RNG
            chosen_caption = self.rng.choice(raw_captions)

        # Tokenize text
        token_ids = self.tokenizer.encode(chosen_caption, add_special_tokens=True)

        # Truncate
        token_ids = token_ids[:self.max_text_len]

        # Create attention mask
        attention_mask = [1] * len(token_ids)

        # Pad
        padding_length = self.max_text_len - len(token_ids)
        token_ids = token_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)

        # Convert to tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Squeeze CLS token (1, 1, D) -> (1, D)
        cls_embeds = cls_embeds.squeeze(0).squeeze(0)

        return patch_embeds, cls_embeds, input_ids, attention_mask, raw_captions

def custom_collate_fn(batch):
    """
    Collate function to handle variable-length patch embeddings.
    Pads patch embeddings to the maximum length in the batch.
    """
    patch_embeds, cls_embeds, input_ids, attention_masks, raw_captions = zip(*batch)
    
    # Pad patch embeddings
    max_patch_len = max(p.shape[0] for p in patch_embeds)
    vit_dim = patch_embeds[0].shape[1]
    
    padded_patches = torch.zeros(len(batch), max_patch_len, vit_dim, dtype=torch.float32)
    patch_attention_mask = torch.zeros(len(batch), max_patch_len, dtype=torch.long)
    
    for i, p in enumerate(patch_embeds):
        seq_len = p.shape[0]
        padded_patches[i, :seq_len] = p
        patch_attention_mask[i, :seq_len] = 1

    # Stack other tensors
    stacked_cls = torch.stack(cls_embeds, dim=0)
    stacked_input_ids = torch.stack(input_ids, dim=0)
    stacked_attn_mask = torch.stack(attention_masks, dim=0)
    
    return padded_patches, stacked_cls, stacked_input_ids, stacked_attn_mask, raw_captions, patch_attention_mask

def get_dataloaders(
    config, 
    tokenizer, 
    stage: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get DataLoaders for a specific training stage.
    
    Stage 1: 4 epochs 'same' + 1 epoch 'diff'
    Stage 2: 4 epochs 'same' + 1 epoch 'diff'
    
    This function will be called inside the epoch loop in the main training script.
    Here, we just create loaders for a specific config.
    """
    
    # --- Training Loader ---
    # We create two datasets: one for same-size, one for diff-size
    train_dataset_same = CocoEmbedDataset(
        data_path=config.same_size_dataset_path,
        split='train',
        tokenizer=tokenizer,
        max_text_len=config.max_text_len,
        same_size=True,
        seed=config.seed
    )
    
    train_dataset_diff = CocoEmbedDataset(
        data_path=config.diff_size_dataset_path,
        split='train',
        tokenizer=tokenizer,
        max_text_len=config.max_text_len,
        same_size=False,
        seed=config.seed
    )

    train_loader_same = DataLoader(
        train_dataset_same,
        batch_size=config.batch_size_same,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    train_loader_diff = DataLoader(
        train_dataset_diff,
        batch_size=config.batch_size_diff,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    # --- Validation Loader ---
    # We use the "different size" validation set, as it's the most general
    val_dataset = CocoEmbedDataset(
        data_path=config.diff_size_dataset_path,
        split='validation', # Assuming 'validation.h5' from 'ls' output
        tokenizer=tokenizer,
        max_text_len=config.max_text_len,
        same_size=False,
        seed=config.seed
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_diff,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    return train_loader_same, train_loader_diff, val_loader