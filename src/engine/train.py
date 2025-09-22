import os
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import hydra
from omegaconf import DictConfig, OmegaConf

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.training import LitCaptioner, _setup_env, _build_model, _get_dataloaders

OmegaConf.register_new_resolver("div", lambda a, b: int(a) // int(b))

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    _setup_env(cfg)
    pl.seed_everything(int(cfg.env.seed), workers=True)

    model = _build_model(cfg)
    lit = LitCaptioner(model, optimizer_cfg=cfg.optimizer)

    train_loader, val_loader = _get_dataloaders(cfg)

    logger = None
    if cfg.logger.wandb.enabled:
        logger = WandbLogger(
            project=cfg.logger.wandb.project,
            name=cfg.logger.wandb.run_name,
            save_dir=cfg.env.output_dir,
            log_model=False,
        )

        hparams = OmegaConf.to_container(cfg, resolve=True)
        logger.log_hyperparams(hparams)

    ckpt_dir = os.path.join(cfg.env.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if cfg.data.use_val:
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{cfg.env.experiment_name}" + "-{epoch:02d}-{val_loss:.3f}",
            save_top_k=1,
            save_last=True,
            monitor="val/loss",
            mode="min",
            every_n_train_steps=cfg.trainer.ckpt_every_n_train_steps,
        )
    else:
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{cfg.env.experiment_name}" + "-{epoch:02d}",
            save_top_k=0,
            save_last=True,
            every_n_train_steps=cfg.trainer.ckpt_every_n_train_steps,
        )
    lr_cb = LearningRateMonitor(logging_interval="step")

    precision = "16-mixed" if bool(cfg.trainer.amp) else "32-true"

    trainer = pl.Trainer(
        accelerator="auto",
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
    )

    trainer.fit(
        lit,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.trainer.resume_from_checkpoint or None,
    )
    print(f"Done. Checkpoints -> {ckpt_dir}")


if __name__ == "__main__":
    main()