import random
import torch
from torchvision.transforms import v2
from datasets import load_dataset



class COCOCaptionsDatasetRAW(torch.utils.data.Dataset):
    def __init__(self, split : str, transform : v2.Transform, all_captions : bool = True, caption_index: None | int =None, 
                 seed=None, cache_dir=None):
        """
        Args
        - split : choose from "train", "test" and "validation"
        - transform : transform image, must change images to same sized tensor for now(collate functio)
        """
        super().__init__()
        self.ds = load_dataset(path=f"Multimodal-Fatima/COCO_captions_{split}", split=split, cache_dir=cache_dir)
        self.all_captions = all_captions 
        self.caption_index = caption_index
        if caption_index is not None:
            if not (0 <= caption_index < 5):
                raise ValueError("caption_index must be between 0..4 (inclusive)") 
        else:
            self.rng = random.Random(seed)

        self.transform = transform
    def __len__(self) -> int:
        return len(self.ds)
    
    def __getitem__(self, index) -> tuple[int, torch.Tensor, list[str]]:
        item = self.ds[index]
        id = item["id"]
        image = item["image"]
        image = self.transform(image)
        caps = item["sentences_raw"]
        if self.all_captions:
            caption = caps
        elif self.caption_index is None:
            caption = caps[self.rng.randrange(len(caps))]
        else:
            caption = caps[self.caption_index]
        
        return id, image, caption
    
def make_coco_raw_collate_fn(same_size=False):    
    def coco_collate_raw_var_size(batch):
        ids, images, captions = zip(*batch)
        return {
            "ids":list(ids),
            "images":list(images),
            "captions":list(captions)
        }
    def coco_collate_raw_same_size(batch):
        ids, images, captions = zip(*batch)
        return {
            "ids":list(ids),
            "images":torch.stack(images),
            "captions":list(captions)
        }
    collate_fn = coco_collate_raw_same_size if same_size else coco_collate_raw_var_size
    return collate_fn

def make_coco_raw_dataloader(
    split: str = "train",
    transform : v2.Transform = v2.Identity(),
    do_resize : bool = False,
    all_captions : bool = True,
    caption_index: int | None = None,
    seed: int = 42,
    cache_dir: str | None = None,
    batch_size: int = 32,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader for the RAW COCO captions dataset.
    """

    dataset = COCOCaptionsDatasetRAW(
        split=split,
        all_captions=all_captions,
        caption_index=caption_index,
        seed=seed,
        cache_dir=cache_dir,
        transform=transform
    )

    def _seed_worker(worker_id: int):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            ds = worker_info.dataset
            if hasattr(ds, "rng"):
                ds.rng = random.Random(worker_info.seed)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    dl = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=make_coco_raw_collate_fn(same_size=do_resize),
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
    )
    return dl
            
        



