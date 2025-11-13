"""
Modal smoke tests to verify remote datasets, loaders, and volume wiring.

What this does (fast, read-only):
- Confirms Modal volume 'cache' is mounted and expected folders/files exist.
- Opens one COCO diff-size shard and inspects shapes/fields.
- Runs a tiny DataLoader iteration using data.COCOCaptions with a stub tokenizer.
- Optionally checks same-size validation file layout and reports any mismatch.

Run:
  modal run tests/modal_dataset_tests.py
"""
import os
import modal


app = modal.App(name="raven-modal-dataset-tests")
cache_vol = modal.Volume.from_name("cache")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
)

# Mount project packages needed for in-container imports (mirrors scripts/coco_vision_encoding.py)
_here = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_here, os.pardir))
image = image.add_local_dir(local_path=os.path.join(_project_root, "data"), remote_path="/root/data")
image = image.add_local_dir(local_path=os.path.join(_project_root, "models"), remote_path="/root/models")


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=60 * 30)
def verify_volume_layout():
    import os

    required_dirs = [
        "/cache/datasets/COCO-captions-vits16plus-embed",
        "/cache/datasets/COCO-captions-vits16plus-embed-same-size",
        "/cache/models/models--google--gemma-3-270m",
        "/cache/models/models--facebookresearch--dinov3",
    ]

    results = {}
    for d in required_dirs:
        results[d] = {
            "exists": os.path.exists(d),
            "is_dir": os.path.isdir(d),
        }

    # count shards
    shards_dir = "/cache/datasets/COCO-captions-vits16plus-embed"
    shard_files = []
    if os.path.isdir(shards_dir):
        try:
            shard_files = [f for f in os.listdir(shards_dir) if f.startswith("train_shard-") and f.endswith(".h5")]
        except Exception:
            shard_files = []

    # check gemma snapshot file presence
    gemma_snap_root = "/cache/models/models--google--gemma-3-270m/snapshots"
    found_gemma_weights = False
    if os.path.isdir(gemma_snap_root):
        for snap in os.listdir(gemma_snap_root):
            if os.path.isfile(os.path.join(gemma_snap_root, snap, "model.safetensors")):
                found_gemma_weights = True
                break

    # report
    print("Volume layout checks:")
    for d, info in results.items():
        print(f"- {d}: exists={info['exists']} is_dir={info['is_dir']}")
    print(f"- Shards in diff-size dir: {len(shard_files)} (e.g., {shard_files[:3]})")
    print(f"- Gemma model.safetensors present: {found_gemma_weights}")


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=60 * 30)
def smoke_open_one_shard():
    import h5py
    import os

    base = "/cache/datasets/COCO-captions-vits16plus-embed"
    # Pick the smallest-numbered shard that exists
    shard_path = None
    for i in range(0, 32):
        candidate = os.path.join(base, f"train_shard-{i}.h5")
        if os.path.exists(candidate):
            shard_path = candidate
            break
    if shard_path is None:
        print("No train_shard-*.h5 files found.")
        return

    with h5py.File(shard_path, "r") as f:
        # Try first group key
        all_keys = list(f.keys())
        if not all_keys:
            print(f"Shard has no groups: {shard_path}")
            return
        k0 = all_keys[0]
        g = f[k0]
        # Inspect datasets
        cls_ds = g["cls"]
        patch_ds = g["patch"]
        caps_ds = g["captions"]
        print(f"Opened {shard_path}")
        print(f"Sample id: {k0}")
        print(f"Raw shapes -> cls: {cls_ds.shape}; patch: {patch_ds.shape}; captions: {caps_ds.shape}")

        # Dataset design: stored with leading batch dim (1, T, 384). Loader code squeezes(0).
        # Diff-size shards may have variable T (e.g. 1200) unlike same-size set (expected 196 tokens).
        # Accept either (T,384) or (1,T,384). We'll squeeze for consistency.
        if len(patch_ds.shape) == 3 and patch_ds.shape[0] == 1:
            T = patch_ds.shape[1]
            squeezed_patch_shape = (T, patch_ds.shape[2])
        elif len(patch_ds.shape) == 2:
            T = patch_ds.shape[0]
            squeezed_patch_shape = patch_ds.shape
        else:
            raise AssertionError(f"Unexpected patch dataset shape {patch_ds.shape}")
        assert squeezed_patch_shape[1] == 384, "Patch embedding dim must be 384"
        assert T > 0, "Patch sequence length must be > 0"

        # CLS expected (1,1,384) or (1,384) after squeeze operations
        assert cls_ds.shape[-1] == 384, "CLS last dim should be 384"
        assert caps_ds.shape[0] >= 1, "At least one caption expected"
        print(f"Interpreted squeezed patch shape: {squeezed_patch_shape}; token count T={T}")
        if T == 196:
            print("Detected same-size style sample (196 tokens).")
        else:
            print("Detected diff-size variable token count (not 196); this is expected for uncropped images.")


class _StubTokenizer:
    """Minimal tokenizer stub for dataset tests.
    Implements __call__ returning objects with .input_ids and .attention_mask tensors,
    and get_vocab_size() used by training code elsewhere.
    """
    def __init__(self, pad_id: int = 0, vocab: int = 1024):
        self.pad_id = pad_id
        self._vocab = vocab

    def __call__(self, text, padding, truncation, max_length, return_tensors):
        import torch
        # ensure non-empty
        tok_len = max(2, min(max_length, len(text.split()) + 2))
        ids = [1] + [2] * (tok_len - 2) + [3]
        if len(ids) < max_length:
            ids = ids + [self.pad_id] * (max_length - len(ids))
        elif len(ids) > max_length:
            ids = ids[:max_length]
        attn = [1 if i != self.pad_id else 0 for i in ids]
        class _Obj:
            pass
        o = _Obj()
        o.input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        o.attention_mask = torch.tensor(attn, dtype=torch.long).unsqueeze(0)
        return o

    def get_vocab_size(self):
        return self._vocab


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=60 * 30)
def dataloader_smoke_diff_size():
    import sys, torch
    # Ensure /root is on sys.path to import mounted packages
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from data.COCOCaptions import CocoEmbedDataset, custom_collate_fn

    tokenizer = _StubTokenizer()
    dataset = CocoEmbedDataset(
        data_path="/cache/datasets/COCO-captions-vits16plus-embed",
        split="validation",  # smaller than train and present according to volume listing
        tokenizer=tokenizer,
        max_text_len=77,
        same_size=False,
        seed=42,
    )

    # small batch to keep memory low
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=False,
    )

    batch = next(iter(dl))
    patch, cls, input_ids, attn_mask, raw_caps, patch_mask = batch
    print("Batch shapes:")
    print({
        "patch": tuple(patch.shape),
        "cls": tuple(cls.shape),
        "input_ids": tuple(input_ids.shape),
        "attn_mask": tuple(attn_mask.shape),
        "patch_mask": tuple(patch_mask.shape),
        "captions_len": len(raw_caps),
    })
    # Assertions
    B = 2
    assert patch.shape[0] == B and patch.shape[-1] == 384, "Patch last dim should be 384"
    assert cls.shape == (B, 1, 384) or cls.shape == (B, 384), "CLS shape mismatch"
    assert input_ids.shape[0] == B and input_ids.shape[1] == 77, "Token length should be 77"
    assert attn_mask.shape == input_ids.shape, "Attention mask shape must match input_ids"
    assert patch_mask.shape[0] == B, "Patch mask B dim"
    assert isinstance(raw_caps, (list, tuple)) and len(raw_caps) == B, "Captions list length"


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=60 * 10)
def optional_same_size_layout_check():
    """Checks if same-size files are placed under root as expected by CocoEmbedDataset.
    Reports mismatch if files are nested under split directories.
    """
    base = "/cache/datasets/COCO-captions-vits16plus-embed-same-size"
    expected_files = [
        os.path.join(base, "train.h5"),
        os.path.join(base, "validation.h5"),
        os.path.join(base, "test.h5"),
    ]
    exists = {p: os.path.exists(p) for p in expected_files}

    # Also check common alternative naming produced by encoder script
    # (nested and root-level *_crop_224.h5)
    alt_validation = os.path.join(base, "validation", "validation_crop_224.h5")
    alt_train = os.path.join(base, "train", "train_crop_224.h5")
    alt_test = os.path.join(base, "test", "test_crop_224.h5")
    alt_validation_root = os.path.join(base, "validation_crop_224.h5")
    alt_train_root = os.path.join(base, "train_crop_224.h5")
    alt_test_root = os.path.join(base, "test_crop_224.h5")
    alts = {
        alt_validation: os.path.exists(alt_validation),
        alt_train: os.path.exists(alt_train),
        alt_test: os.path.exists(alt_test),
        alt_validation_root: os.path.exists(alt_validation_root),
        alt_train_root: os.path.exists(alt_train_root),
        alt_test_root: os.path.exists(alt_test_root),
    }

    print("Same-size expected files at root:")
    for p, ex in exists.items():
        print(f"- {p}: {ex}")
    print("Alternative nested files (from encoder script):")
    for p, ex in alts.items():
        print(f"- {p}: {ex}")
    if not all(exists.values()) and any(alts.values()):
        print("NOTE: Found alternative *_crop_224.h5 layout (root-level or nested). Loader should support this or files should be renamed.")


@app.local_entrypoint()
def main():
    print("Running Modal dataset smoke tests remotely…")
    verify_volume_layout.remote()
    smoke_open_one_shard.remote()
    dataloader_smoke_diff_size.remote()
    optional_same_size_layout_check.remote()
    print("Submitted all tests. Use modal logs to inspect outputs.")
