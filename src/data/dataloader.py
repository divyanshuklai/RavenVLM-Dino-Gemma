import random

import torch
import torch.nn as nn

import datasets as hfds
from transformers import AutoTokenizer, AutoImageProcessor

class CocoCaptions(torch.utils.data.Dataset):
    """
    Returns (PIL.Image, caption_str). 
    - split: 'train' | 'validation' | 'test'
    - caption_index: 0..4 for deterministic choice, or None for random choice from sentences_raw
    - seed: controls the per-sample random caption selection when caption_index is None
    """
    def __init__(self, gemma_id, vit_id, split="train", caption_index=None, seed=None, cache_dir=None):

        self.gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_id)
        self.vit_processor = AutoImageProcessor.from_pretrained(vit_id)

        ds = hfds.load_dataset(f"Multimodal-Fatima/COCO_captions_{split}", cache_dir=cache_dir)
        self.ds = ds[split]
        self.caption_index = caption_index
        if caption_index is not None:
            if caption_index < 0 or caption_index > 4:
                raise ValueError("caption_index must be in [0, 4] or None")
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]  # PIL Image (per dataset feature)
        caps = item["sentences_raw"]  # list[str], length 5
        if self.caption_index is None:
            caption = caps[self.rng.randrange(len(caps))]
        else:
            caption = caps[self.caption_index]

        image = self.vit_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

        return image, caption


def coco_collate(batch):
    images, captions = zip(*batch)
    return {
        "images":torch.stack(images), 
        "captions":list(captions)
    }


def make_coco_dataloader(
    gemma_id,
    vit_id,
    split="train",
    batch_size=8,
    shuffle=None,
    caption_index=None,
    seed=None,
    num_workers=0,
    pin_memory=False,
    cache_dir=None,
    persistent_workers=False,
    prefetch_factor=2,
    pin_memory_device="",
):
    """
    Construct a Dataloader for Coco captions datasets using hf:Multimodal-Fatima/COCO_captions_{split}.
    split, caption_index, seed and cache_dir sent to CocoCaptions Dataset class.
    rest used in DataLoader.
    """
    dataset = CocoCaptions(gemma_id, vit_id, split=split, caption_index=caption_index, seed=seed, cache_dir=cache_dir)
    if shuffle is None:
        shuffle = split == "train"

    loader_kwargs = {}
    if num_workers and num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    if pin_memory_device:
        loader_kwargs["pin_memory_device"] = str(pin_memory_device)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=coco_collate,
        **loader_kwargs,
    )