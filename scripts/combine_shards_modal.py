# combine_shards_modal.py
import os
import re
import shutil
import h5py
import modal
from tqdm.auto import tqdm

# === CONFIG ===
DATASET_REL_PATH = "datasets/COCO-captions-vits16plus-embed"
OUTPUT_FILENAME = "train_combined.h5"
CACHE_VOLUME_NAME = "cache"
SHARD_PATTERN = re.compile(r"train_shard-(\d+)\.h5$")  # numeric ordering

# Modal app + image (keep consistent with your other scripts)
app = modal.App()
image = modal.Image.debian_slim("3.13").pip_install("h5py", "tqdm")

# Use the same volume name you have in your other scripts
CACHE_VOLUME = modal.Volume.from_name(CACHE_VOLUME_NAME)


def _get_free_bytes(path="/cache"):
    """Return free bytes for filesystem containing path (inside container)."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free
    except Exception:
        return None


@app.function(
    image=image,
    volumes={"/cache": CACHE_VOLUME},
    timeout=60 * 60 * 4,  # 4 hours (adjust if you expect longer)
    retries=0,
)
def combine_shards():
    """
    Combine all train_shard-*.h5 (top-level groups per sample) into a single HDF5 file.
    Writes to: /cache/<DATASET_REL_PATH>/<OUTPUT_FILENAME>
    """
    dataset_dir = os.path.join("/cache", DATASET_REL_PATH)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found in-container: {dataset_dir}")

    # discover shards
    shards = []
    for fname in os.listdir(dataset_dir):
        m = SHARD_PATTERN.match(fname)
        if m:
            shards.append((int(m.group(1)), fname))
    shards.sort()

    if not shards:
        print("No shards found matching pattern 'train_shard-*.h5'. Nothing to do.")
        return

    shard_files = [os.path.join(dataset_dir, fname) for _, fname in shards]
    print(f"Found {len(shard_files)} shards. First few: {shard_files[:5]}")

    # Quick estimate: sum file sizes and check free disk space
    total_shard_bytes = sum(os.path.getsize(p) for p in shard_files)
    out_path = os.path.join(dataset_dir, OUTPUT_FILENAME)

    free_bytes = _get_free_bytes("/cache")
    if free_bytes is not None:
        print(f"Total shards size = {total_shard_bytes / (1024**3):.2f} GiB. Free space = {free_bytes / (1024**3):.2f} GiB.")
        # We need some breathing room; combining may need a bit of extra space during copying.
        if free_bytes < total_shard_bytes * 0.95:
            print("WARNING: free space appears to be less than total shards size. Combining may fail due to lack of disk space.")

    # If output exists, back it up (rename) to avoid accidental overwrite
    if os.path.exists(out_path):
        backup_path = out_path + ".bak"
        print(f"Output {out_path} already exists. Renaming to {backup_path}")
        os.replace(out_path, backup_path)

    # Begin combination
    total_groups_written = 0
    encountered_name_conflicts = 0

    # Using libver='latest' may help performance for large files
    try:
        with h5py.File(out_path, "w", libver="latest") as h_dst:
            for shard_path in tqdm(shard_files, desc="Shards", leave=True, dynamic_ncols=True):
                shard_basename = os.path.basename(shard_path)
                try:
                    with h5py.File(shard_path, "r") as h_src:
                        groups = list(h_src.keys())
                        # iterate groups one-by-one to avoid copying entire file at once
                        for gid in groups:
                            # If name already present in destination, avoid overwrite:
                            if gid in h_dst:
                                # append a suffix derived from shard filename to preserve the sample
                                safe_name = f"{gid}__{shard_basename.replace('.h5','')}"
                                encountered_name_conflicts += 1
                                print(f"[WARN] Name conflict for sample id '{gid}' -> writing as '{safe_name}'")
                            else:
                                safe_name = gid
                            try:
                                # Use h5py's efficient copy (preserves datasets/attrs)
                                h_src.copy(gid, h_dst, name=safe_name)
                                total_groups_written += 1
                            except Exception as e:
                                # log and continue - a single problematic group shouldn't stop whole job
                                print(f"[ERROR] Failed to copy group '{gid}' from {shard_basename}: {e}")
                except Exception as e:
                    print(f"[ERROR] Cannot open/read shard {shard_basename}: {e}. Skipping this shard.")
            h_dst.flush()

    except Exception as e:
        # If a fatal error occurs (likely disk full), try to remove partial output to avoid confusion
        print(f"[FATAL] Combining failed: {e}")
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
                print(f"Removed partial output file {out_path}")
            except Exception:
                print(f"Could not remove {out_path}; please inspect or delete manually.")
        raise

    print("\n=== Summary ===")
    print(f"Combined {total_groups_written} groups into: {out_path}")
    if encountered_name_conflicts:
        print(f"Name conflicts encountered: {encountered_name_conflicts} (conflicted names were suffixed with shard basename)")
    print("Done.")


@app.local_entrypoint()
def main():
    print("Starting combine_shards() on Modal (this will block until finished).")
    # .remote() returns a handle but calling .result() will block until finished and show errors.
    # The call below will run remotely in Modal and block until the function finishes.
    list(combine_shards.remote())
    # wait for it to finish and propagate exceptions if any
    print("Remote combine_shards() completed. Check the /cache datasets folder for train_combined.h5")
