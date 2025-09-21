import os
import itertools
import subprocess
import time
import modal
from pathlib import Path

HF_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)
EXP_VOL = modal.Volume.from_name("experiments-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install(
        # deps from pyproject
        "accelerate>=1.10.1",
        "datasets>=4.0.0",
        "evaluate>=0.4.5",
        "huggingface-hub[cli]>=0.34.4",
        "hydra-core>=1.3.2",
        "modal>=1.1.4",
        "pytest>=8.4.2",
        "pytorch-lightning>=2.5.5",
        "sentencepiece>=0.2.1",
        "tensorboard>=2.20.0",
        "torch>=2.8.0",
        "torchvision>=0.23.0",
        "transformers>=4.56.0",
        "trl>=0.22.2",
        "wandb>=0.21.4",
        # use uv to mirror local launcher behavior
        "uv>=0.4.0",
        "hf_transfer>=0.1.6",
    )
)

project_root = Path(__file__).resolve().parent.parent
# ...existing code...
# Removed duplicate assignment:
# project_root = Path(__file__).resolve().parent.parent

# Remove the broad add_local_dir of the whole repo
# image = image.add_local_dir(
#     project_root,
#     remote_path="/workspace/",
#     ignore=[".git",".venv","**/__pycache__", ".pytest_cache", "wandb", "outputs"]
# )

# Add only what’s needed for training
image = image.add_local_dir(
    project_root / "src",
    remote_path="/workspace/src",
    ignore=["**/__pycache__", ".pytest_cache"]
)
image = image.add_local_dir(
    project_root / "configs",
    remote_path="/workspace/configs",
)
image = image.add_local_dir(
    project_root / "scripts",
    remote_path="/workspace/scripts",
    ignore=["**/.ipynb_checkpoints"]
)

app = modal.App("captioner-hparam-sweep")

def _to_hydra_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)

@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache/hf": HF_VOL, "/workspace/experiments": EXP_VOL},
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")]
)
def train_one(overrides: dict):
    # HF + logging env
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("HF_DATASETS_CACHE", "/cache/hf")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Unbuffered output + clearer hydra errors + faster downloads
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # Disable W&B unless you pass a valid WANDB_API_KEY via secret
    os.environ.setdefault("WANDB_MODE", "enabled")
    os.environ.setdefault("WANDB_DIR", "/workspace/experiments")

    # Build Hydra CLI overrides
    hydra_args = [f"{k}={_to_hydra_value(v)}" for k, v in overrides.items()]
    # Only set a default experiment name if not provided
    if "env.experiment_name" not in overrides:
        hydra_args.append("env.experiment_name=sweep")

    # Use plain Python; uv --system isn't supported in this image
    cmd = ["python", "-u", "/workspace/src/engine/train.py", *hydra_args]

    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd="/workspace")

@app.local_entrypoint()
def sweep():
    grid = {
        "optimizer.lr": [5e-5, 1e-5, 1e-4],
        "data.batch_size": [2, 4],
        "model.include_patches": [False, True],
    }

    stamp = int(time.time())
    runs = []
    for values in itertools.product(*grid.values()):
        ovr = dict(zip(grid.keys(), values))
        ovr["env.experiment_name"] = f"sweeps/{stamp}"
        runs.append(ovr)

    # Run one locally for quick smoke test
    # Fan out on Modal:
    list(train_one.map(runs))