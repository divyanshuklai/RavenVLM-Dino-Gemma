import random

import torch
import torch.nn as nn

import datasets as hfds

class CocoCaptions(torch.utils.data.Dataset):
    """
    Returns (PIL.Image, caption_str). 
    - split: 'train' | 'validation' | 'test'
    - caption_index: 0..4 for deterministic choice, or None for random choice from sentences_raw
    - seed: controls the per-sample random caption selection when caption_index is None
    """
    def __init__(self, split="train", caption_index=None, seed=None, cache_dir=None):
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
        return image, caption


def coco_collate(batch):
    images = [img.convert('RGB') if hasattr(img, "mode") and img.mode != 'RGB' else img for img, _ in batch]
    captions = [cap for _, cap in batch]
    return images, captions


def make_coco_dataloader(
    split="train",
    batch_size=8,
    shuffle=None,
    caption_index=None,
    seed=None,
    num_workers=0,
    pin_memory=False,
    cache_dir=None,
):
    """
    Construct a Dataloader for Coco captions datasets using hf:Multimodal-Fatima/COCO_captions_{split}.
    split, caption_index, seed and cache_dir sent to CocoCaptions Dataset class.
    rest used in DataLoader.
    """
    dataset = CocoCaptions(split=split, caption_index=caption_index, seed=seed, cache_dir=cache_dir)
    if shuffle is None:
        shuffle = split == "train"
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=coco_collate,
    )