import os
import itertools
import subprocess
import modal

from pathlib import Path


HF_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)
EXP_VOL = modal.Volume.from_name("experiments", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install(
        # deps from pyproject + additional
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
        "uv>=0.4.0",
        "hf_transfer>=0.1.6",
    )
)

project_root = Path(__file__).resolve().parent.parent

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
    if v is None:
        return "null"
    elif isinstance(v, bool):
        return "true" if v else "false"
    return str(v)

@app.function(
    image=image,
    gpu="L40S",
    volumes={"/cache/hf": HF_VOL, "/workspace/experiments": EXP_VOL},
    timeout= 60 * 60 * 10,
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")]
)
def train_one(overrides):
    # HF + logging env
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("HF_DATASETS_CACHE", "/cache/hf")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Unbuffered output + clearer hydra errors + faster downloads
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    os.environ.setdefault("WANDB_MODE", "online")
    os.environ.setdefault("WANDB_DIR", "/workspace/experiments")

    hydra_args = [f"{k}={_to_hydra_value(v)}" for k, v in overrides.items()]

    cmd = ["python", "-u", "/workspace/src/engine/train.py", *hydra_args]

    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd="/workspace")

@app.local_entrypoint()
def sweep():

    common_ovr = {
        "trainer.max_epochs":5,
        "trainer.val_check_interval":1_000,
        "trainer.ckpt_every_n_train_steps":1_000,
        "logger.wandb.enabled":True,
        "trainer.log_every_n_steps":5,
        "trainer.amp":False,
        "data.num_workers":8,
        "data.val_batch_size":16,
    }

    grid = {
        "optimizer.lr": [2e-3, 1e-5],
        "data.batch_size": [16],
        "trainer.accumulate_grad_batches":[8],
        "model.include_patches": [True],
        "model.freeze_gemma":[False, True],
        "model.include_registers":[False],
        "trainer.gradient_clip_val":[1.0],
    }

    from datetime import datetime
    stamp = datetime.now().strftime("%d-%m-%Y_%H%M")
    runs = []
    for values in itertools.product(*grid.values()):
        ovr = dict(zip(grid.keys(), values))
        run_name = f"{stamp}_run_{len(runs)}"
        ovr["env.experiment_name"] = run_name
        ovr["env.output_dir"] = f"experiments/{stamp}/{run_name}"
        ovr["logger.wandb.group"] = f"{stamp}"
        ovr.update(common_ovr)
        runs.append(ovr)

    # Fan out on Modal:
    list(train_one.map(runs))

    print(f"THIS SWEEP WAS : {stamp}")
