import os

import torch
import torch.nn as nn

import pytorch_lightning as pl

from omegaconf import DictConfig


class LitCaptioner(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizer_cfg: dict):
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg

    def training_step(self, batch, batch_idx: int):
        images, captions = batch  #  coco_collate: (list[PIL.Image], list[str])
        out = self.model(images, captions) 
        loss = out.loss if hasattr(out, "loss") else out["loss"]

        bsz = images.shape[0] if hasattr(images, "shape") else out["loss"]
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bsz)
        return loss

    def validation_step(self, batch, batch_idx: int):
        images, captions = batch
        out = self.model(images, captions)
        val_loss = out.loss if hasattr(out, "loss") else out["loss"]

        bsz = images.shape[0] if hasattr(images, "shape") else len(images)
        self.log("val/loss", val_loss, prog_bar=True, on_epoch=True, batch_size=bsz)

    def configure_optimizers(self):
        lr = float(self.optimizer_cfg.get("lr", 1e-4))
        wd = float(self.optimizer_cfg.get("weight_decay", 0.01))
        betas = tuple(self.optimizer_cfg.get("betas", (0.9, 0.999)))
        eps = float(self.optimizer_cfg.get("eps", 1e-8))
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


def _setup_env(cfg: DictConfig):
    # HF cache path
    hf_home = cfg.env.get("hf_home")
    if hf_home:
        os.environ["HF_HOME"] = os.path.abspath(os.path.expanduser(hf_home))
        os.environ.setdefault("HF_DATASETS_CACHE", os.environ["HF_HOME"])
    # W&B
    if cfg.logger.wandb.enabled:
        os.environ.setdefault("WANDB_SILENT", "true")
        if cfg.env.get("output_dir"):
            os.environ.setdefault("WANDB_DIR", os.path.abspath(cfg.env.output_dir))


def _build_model(cfg: DictConfig) -> nn.Module:
    from src.models.caption_modelling import GemmaDinoImageCaptioner
    return GemmaDinoImageCaptioner(
        gemma_id=cfg.model.gemma_id,
        vit_id=cfg.model.vit_id,
        include_cls=cfg.model.include_cls,
        include_registers=cfg.model.include_registers,
        include_patches=cfg.model.include_patches,
        freeze_gemma=cfg.model.freeze_gemma,
    )


def _get_dataloaders(cfg: DictConfig):
    from src.data.dataloader import make_coco_dataloader
    train_loader = make_coco_dataloader(
        split=cfg.data.train_split,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        caption_index=cfg.data.caption_index,
        seed=cfg.env.seed,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        cache_dir=cfg.env.hf_home or None,
    )
    val_loader = None
    if cfg.data.use_val:
        val_loader = make_coco_dataloader(
            split=cfg.data.val_split,
            batch_size=cfg.data.val_batch_size,
            shuffle=False,
            caption_index=cfg.data.caption_index,
            seed=cfg.env.seed,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            cache_dir=cfg.env.hf_home or None,
        )
    return train_loader, val_loader