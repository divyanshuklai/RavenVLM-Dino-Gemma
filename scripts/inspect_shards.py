import json
from pathlib import Path
from tqdm.auto import tqdm

import modal

app = modal.App()

# Change this image if you need additional system packages (apt_install) or a different Python version
image = modal.Image.debian_slim("3.13").pip_install("h5py", "numpy", "tqdm")

# Default volume name used in your example (you used `--volume cache` in modal shell)
# If your volume has a different name, pass --volume VOLUME_NAME to the script when running `modal run`.

@app.function(
        image=image,
        volumes={"/cache":modal.Volume.from_name("cache")},
        timeout=60*60*10,
        retries=0,
)
def inspect_shards(dataset_path: str = "/cache/datasets/COCO-captions-vits16plus-embed"):
    """Runs inside a Modal container. Scans every .h5 file under dataset_path and
    returns a JSON-serializable structure describing groups, datasets, shapes, dtypes and attributes.
    """
    import os
    import h5py

    out = {}
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found inside container: {dataset_path}")

    for fname in tqdm(sorted(os.listdir(dataset_path)), 
                      desc="INSPECT", 
                      total=len(os.listdir(dataset_path)), 
                      dynamic_ncols=True,
                      leave=False,
                      mininterval=1.0,
                    ):
        if not fname.endswith(".h5"):
            continue
        fpath = os.path.join(dataset_path, fname)
        try:
            with h5py.File(fpath, "r") as h:
                info = {}

                def visitor(name, obj):
                    # obj may be Group or Dataset
                    if isinstance(obj, h5py.Dataset):
                        # dataset details
                        info[name] = {
                            "type": "dataset",
                            "shape": obj.shape,
                            "dtype": str(obj.dtype),
                            "attrs": {k: (v.tolist() if hasattr(v, "tolist") else str(v)) for k, v in obj.attrs.items()},
                        }
                    else:
                        info[name] = {"type": "group"}

                h.visititems(visitor)
                out[fname] = info
        except Exception as e:
            out[fname] = {"error": str(e)}

    # Also write a JSON summary inside the container so you can fetch it later if needed
    summary_path = os.path.join(dataset_path, "shard_structure.json")
    with open(summary_path, "w") as fo:
        json.dump(out, fo, indent=2, default=str)

    print(f"Wrote structure to {summary_path}")
    return out


@app.local_entrypoint()
def main():
    # Note: we declare the volume in the function decorator above. If your volume name isn't "cache",
    # edit the decorator or recreate the app with the correct volume mapping.

    print("Spawning remote inspect_shards() on Modal...")
    # spawn() returns a FunctionCall object; use .get() to wait for the result and fetch it
    result = inspect_shards.remote()

    # print a readable summary to stdout
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
