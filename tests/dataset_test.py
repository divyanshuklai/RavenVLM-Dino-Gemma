import modal

app = modal.App("CZECH")

vol = modal.Volume.from_name("cache")

@app.function(volumes={"/cache": vol}, image=modal.Image.debian_slim().pip_install("requests","h5py","numpy"))
def main():
    import h5py, numpy as np
    #load a train dataset shard from datasets/COCO-captions-vits16plus-embed
    with h5py.File("/cache/datasets/COCO-captions-vits16plus-embed/train_shard-0.h5", "r") as f:
        sample = f["0"]
        cls = sample["cls"]
        patch = sample["patch"]
        captions = sample["captions"]
        print(f"Loaded {len(cls)} embeddings with shape {cls.shape}")
        print(f"Sample caption: {[caption for caption in captions]}")
        print(f"Patch embeddings shape {patch.shape}")