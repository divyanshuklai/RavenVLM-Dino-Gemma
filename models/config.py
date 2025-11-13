# models/config.py

import dataclasses
from dataclasses import dataclass
import os

# Base paths inside the Modal container
CACHE_PATH = "/cache"
MODEL_CACHE_PATH = os.path.join(CACHE_PATH, "models")
DATA_CACHE_PATH = os.path.join(CACHE_PATH, "datasets")

@dataclass
class TrainConfig:
    """
    Configuration for BLIP-2 style training on Modal.
    """
    
    # --- Paths ---
    data_path: str = DATA_CACHE_PATH
    same_size_dataset_path: str = os.path.join(DATA_CACHE_PATH, "COCO-captions-vits16plus-embed-same-size")
    diff_size_dataset_path: str = os.path.join(DATA_CACHE_PATH, "COCO-captions-vits16plus-embed")
    
    # Model paths from user-provided 'ls' output
    gemma_model_path: str = os.path.join(MODEL_CACHE_PATH, "models--google--gemma-3-270m")
    # Using the specific vits16plus model checkpoint from 'ls'
    dinov3_model_path: str = os.path.join(MODEL_CACHE_PATH, "models--facebookresearch--dinov3", "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")
    
    output_dir: str = os.path.join(CACHE_PATH, "experiments", "qformer")

    # --- WandB Logging ---
    wandb_project: str = "DinoGemmaCaptionerQFormer"
    wandb_entity: str = "divyanshukla"
    wandb_log_every_n_steps: int = 10
    validation_every_n_steps_ratio: float = 0.1 # Run validation every 1/10th of iterations

    # --- General Training Params ---
    seed: int = 42
    batch_size_same: int = 64
    batch_size_diff: int = 32  # Smaller for variable-sized embeddings due to padding
    num_workers: int = 4
    max_text_len: int = 77  # Max sequence length for captions

    # --- Stage 1: QFormer Pre-training (ITC, ITM, ITG) ---
    stage1_epochs_same: int = 4
    stage1_epochs_diff: int = 1
    stage1_lr: float = 1e-4
    # Loss weights
    itc_loss_weight: float = 1.0
    itm_loss_weight: float = 1.0
    itg_loss_weight: float = 1.0
    itm_negative_strategy: str = "in-batch" # Use in-batch negatives for ITM
    
    # --- Stage 2: QFormer -> LLM Generative Finetuning ---
    stage2_epochs_same: int = 4
    stage2_epochs_diff: int = 1
    stage2_lr: float = 1e-5
    
    # --- Model Architecture ---
    # From 'ls' output: shape=(..., 384), so ViT dim is 384
    vit_embed_dim: int = 384 
    
    # QFormer config
    qformer_num_query_tokens: int = 32
    qformer_hidden_dim: int = 768 # Internal dimension of QFormer
    qformer_num_heads: int = 8
    qformer_num_layers: int = 6
    
    # Gemma embedding dim (from Gemma3 config; current lightweight model emb_dim=640)
    # This must match language_model.GEMMA3_CONFIG_270M['emb_dim'] to avoid projection mismatch.
    gemma_hidden_dim: int = 640 
    
    # Adapter config
    adapter_hidden_layers: int = 1

    def to_dict(self):
        return dataclasses.asdict(self)