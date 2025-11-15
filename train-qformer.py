# train-qformer.py
import numpy as np
import random
import modal
import os
import torch
import torch.nn.functional as F
import wandb
import logging
from typing import List, Dict
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import from our project files
from models.config import TrainConfig
from data.COCOCaptions import get_dataloaders

# Import models 
from models.qformer import QFormer
from models.qformer_losses import QFormerForPretraining
from models.language_model import build_gemma_model_and_tokenizer
from models.vision_language_model import VLMQFormer, VisualPrefixAdapter

# For CIDEr validation
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


from models.config import CACHE_PATH

# --- Modal Setup ---

app = modal.App(name="raven-qformer-training")

# Define the Modal Volume for cache
cache_vol = modal.Volume.from_name("cache")

# Define the Modal Image
image = (
    modal.Image.debian_slim(python_version="3.13") # Using a common stable version
    .apt_install("default-jdk")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("evals", remote_path="/root/evals")
    .add_local_dir("models", remote_path="/root/models")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def set_seed(seed: int):
    """Set deterministic seeding"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model_state_dict: dict, optimizer_state_dict: dict, epoch: int, step: int, config: TrainConfig, stage: int, filename_prefix: str):
    """Saves a model checkpoint."""
    save_dir = Path(config.output_dir) / f"stage_{stage}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{filename_prefix}_epoch_{epoch}_step_{step}.pt"
    
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
    }, save_path)
    logger.info(f"Saved checkpoint to {save_path}")
    
    latest_path = save_dir / f"{filename_prefix}_latest.pt"
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
    }, latest_path)

def format_captions_for_cider(gts: Dict[str, List[str]], res: Dict[str, List[str]]):
    """Formats ground truth and generated captions for pycocoevalcap."""
    gts_fmt = {k: [{'caption': c} for c in v] for k, v in gts.items()}
    res_fmt = {k: [{'caption': v[0]}] if isinstance(v, list) else [{'caption': v}] for k, v in res.items()}

    tokenizer = PTBTokenizer()
    gts_tokenized = tokenizer.tokenize(gts_fmt)
    res_tokenized = tokenizer.tokenize(res_fmt)
    return gts_tokenized, res_tokenized

# --- Validation Functions ---

@torch.no_grad()
def run_validation_stage1(model: QFormerForPretraining, val_loader: DataLoader, device: torch.device, config: TrainConfig) -> Dict[str, float]:
    """Runs validation for Stage 1, returning average losses."""
    model.eval()
    total_loss, total_itc_loss, total_itm_loss, total_itg_loss = 0, 0, 0, 0
    
    for batch in tqdm(val_loader, desc="Validation Stage 1", leave=False):
        patch_embeds, _, text_ids, text_mask, _, patch_attention_mask = batch
        patch_embeds, text_ids, text_mask, patch_attention_mask = patch_embeds.to(device, dtype=torch.bfloat16), text_ids.to(device), text_mask.to(device), patch_attention_mask.to(device, dtype=torch.bool)        
        losses = model(vit_embeds=patch_embeds, text_input_ids=text_ids, text_attention_mask=text_mask, patch_attention_mask=patch_attention_mask)
        
        loss_itc = losses.get('itc_loss', 0.0)
        loss_itm = losses.get('itm_loss', 0.0)
        loss_itg = losses.get('itg_loss', 0.0)
        
        total_itc_loss += loss_itc.item()
        total_itm_loss += loss_itm.item()
        total_itg_loss += loss_itg.item()
        total_loss += (loss_itc * config.itc_loss_weight + loss_itm * config.itm_loss_weight + loss_itg * config.itg_loss_weight).item()

    num_batches = len(val_loader)
    return {
        "val/loss": total_loss / num_batches,
        "val/itc_loss": total_itc_loss / num_batches,
        "val/itm_loss": total_itm_loss / num_batches,
        "val/itg_loss": total_itg_loss / num_batches,
    }

@torch.no_grad()
def run_validation_stage2(vlm: VLMQFormer, tokenizer, val_loader: DataLoader, device: torch.device, config: TrainConfig) -> Dict[str, float]:
    """Runs validation for Stage 2, calculating CIDEr score."""
    vlm.eval()
    cider_scorer = Cider()
    gts_cider, res_cider = {}, {}
    sample_idx = 0
    total_loss = 0

    for batch in tqdm(val_loader, desc="Validation Stage 2", leave=False):
        patch_embeds, _, text_ids, _, raw_captions_batch, patch_attention_mask = batch
        patch_embeds, text_ids, patch_attention_mask = patch_embeds.to(device, dtype=torch.bfloat16), text_ids.to(device), patch_attention_mask.to(device, dtype=torch.bool)
        
        logits = vlm(vis_emb=patch_embeds, input_ids=text_ids, mode="stage2", patch_attention_mask=patch_attention_mask)
        
        logits = logits[:, config.qformer_num_query_tokens:-1, :].contiguous()
        labels = text_ids[:, 1:].contiguous()
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
        total_loss += loss.item()

        generated_texts = vlm.generate(vis_emb=patch_embeds, tokenizer=tokenizer, max_new_tokens=config.max_text_len, eos_token_id=tokenizer.eos_token_id, device=device, patch_attention_mask=patch_attention_mask)
        
        for i, gen_text in enumerate(generated_texts):
            img_id = str(sample_idx)
            res_cider[img_id] = [gen_text]
            gts_cider[img_id] = raw_captions_batch[i]
            sample_idx += 1

    gts_tokenized, res_tokenized = format_captions_for_cider(gts_cider, res_cider)
    cider_score, _ = cider_scorer.compute_score(gts_tokenized, res_tokenized)
    
    return {"val/loss": total_loss / len(val_loader), "val/cider_score": cider_score}

# --- Main Training Function ---

@app.function(image=image, gpu="A100", volumes={CACHE_PATH: cache_vol}, secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")], timeout=86400, retries=0)
def train(config: TrainConfig):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=config.to_dict())
    
    # Create a unique output directory for this run
    config.output_dir = os.path.join(config.output_dir, wandb.run.id)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training with config:\n{config.to_dict()}")
    
    logger.info("Loading models...")
    gemma_model, tokenizer = build_gemma_model_and_tokenizer(config.gemma_model_path)
    
    # --- STAGE 1: QFormer Pre-training ---
    logger.info("--- Starting Stage 1: QFormer Pre-training ---")
    
    qformer_core = QFormer(embed_dim=config.qformer_hidden_dim, num_blocks=config.qformer_num_layers, num_heads=config.qformer_num_heads, num_queries=config.qformer_num_query_tokens, vision_dim=config.vit_embed_dim, lm_vocab_size=tokenizer.get_vocab_size()).to(device, dtype=torch.bfloat16)
    qformer_for_pretraining = QFormerForPretraining(qformer=qformer_core, gemma_vocab_size=tokenizer.get_vocab_size()).to(device, dtype=torch.bfloat16)
    optimizer_s1 = torch.optim.AdamW(qformer_for_pretraining.parameters(), lr=config.stage1_lr)
    
    logger.info("Loading datasets...")
    train_loader_same, train_loader_diff, val_loader = get_dataloaders(config, tokenizer, stage=1)
    
    total_steps_s1 = (len(train_loader_same) * config.stage1_epochs_same) + (len(train_loader_diff) * config.stage1_epochs_diff)
    val_interval_s1 = max(1, int(total_steps_s1 * config.validation_every_n_steps_ratio))
    logger.info(f"Stage 1: {total_steps_s1} steps, validating every {val_interval_s1} steps.")

    global_step_s1 = 0
    qformer_for_pretraining.train()
    for loader, num_epochs in [(train_loader_same, config.stage1_epochs_same), (train_loader_diff, config.stage1_epochs_diff)]:
        for epoch in range(num_epochs):
            for batch in tqdm(loader, desc=f"Stage 1 Epoch {epoch+1}/{num_epochs}", leave=False):
                patch_embeds, _, text_ids, text_mask, _, patch_attention_mask = batch
                patch_embeds, text_ids, text_mask, patch_attention_mask = patch_embeds.to(device, dtype=torch.bfloat16), text_ids.to(device), text_mask.to(device), patch_attention_mask.to(device, dtype=torch.bool)
                
                optimizer_s1.zero_grad()
                losses = qformer_for_pretraining(vit_embeds=patch_embeds, text_input_ids=text_ids, text_attention_mask=text_mask, patch_attention_mask=patch_attention_mask)
                
                loss_itc = losses['itc_loss'] * config.itc_loss_weight
                loss_itm = losses['itm_loss'] * config.itm_loss_weight
                loss_itg = losses['itg_loss'] * config.itg_loss_weight
                total_loss = loss_itc + loss_itm + loss_itg
                
                total_loss.backward()
                optimizer_s1.step()
                
                if global_step_s1 % config.wandb_log_every_n_steps == 0:
                    wandb.log({"stage1/train_loss": total_loss.item(), "stage1/itc_loss": loss_itc.item(), "stage1/itm_loss": loss_itm.item(), "stage1/itg_loss": loss_itg.item(), "stage1/epoch": epoch, "global_step_s1": global_step_s1})
                
                if global_step_s1 > 0 and global_step_s1 % val_interval_s1 == 0:
                    logger.info(f"Running Stage 1 validation at step {global_step_s1}...")
                    val_metrics = run_validation_stage1(qformer_for_pretraining, val_loader, device, config)
                    wandb.log({"global_step_s1": global_step_s1, **val_metrics})
                    qformer_for_pretraining.train()

                global_step_s1 += 1
    
    save_checkpoint(qformer_for_pretraining.qformer.state_dict(), optimizer_s1.state_dict(), -1, global_step_s1, config, 1, "qformer_final")
    logger.info("--- Finished Stage 1 ---")

    # --- STAGE 2: Generative Finetuning ---
    logger.info("--- Starting Stage 2: Generative Finetuning ---")

    qformer_for_pretraining.qformer.eval()
    adapter = VisualPrefixAdapter(qformer_dim=config.qformer_hidden_dim, gemma_emb_dim=gemma_model.cfg["emb_dim"]).to(device, dtype=torch.bfloat16)
    vlm = VLMQFormer(qformer=qformer_for_pretraining.qformer, gemma=gemma_model.to(device, dtype=torch.bfloat16), adapter=adapter, freeze_gemma=True, stage="stage2").to(device)
    
    optimizer_s2 = torch.optim.AdamW(vlm.adapter.parameters(), lr=config.stage2_lr)
    
    total_steps_s2 = (len(train_loader_same) * config.stage2_epochs_same) + (len(train_loader_diff) * config.stage2_epochs_diff)
    val_interval_s2 = max(1, int(total_steps_s2 * config.validation_every_n_steps_ratio))
    logger.info(f"Stage 2: {total_steps_s2} steps, validating every {val_interval_s2} steps.")

    global_step_s2 = 0
    vlm.train()
    for loader, num_epochs in [(train_loader_same, config.stage2_epochs_same), (train_loader_diff, config.stage2_epochs_diff)]:
        for epoch in range(num_epochs):
            for batch in tqdm(loader, desc=f"Stage 2 Epoch {epoch+1}/{num_epochs}", leave=False):
                patch_embeds, _, text_ids, _, _, patch_attention_mask = batch
                patch_embeds, text_ids, patch_attention_mask = patch_embeds.to(device, dtype=torch.bfloat16), text_ids.to(device), patch_attention_mask.to(device, dtype=torch.bool)
                
                optimizer_s2.zero_grad()
                logits = vlm(vis_emb=patch_embeds, input_ids=text_ids, mode="stage2", patch_attention_mask=patch_attention_mask)
                
                logits = logits[:, config.qformer_num_query_tokens:-1, :].contiguous()
                labels = text_ids[:, 1:].contiguous()
                loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
                
                loss.backward()
                optimizer_s2.step()
                
                if global_step_s2 % config.wandb_log_every_n_steps == 0:
                    wandb.log({"stage2/train_loss": loss.item(), "stage2/epoch": epoch, "global_step_s2": global_step_s2})

                if global_step_s2 > 0 and global_step_s2 % val_interval_s2 == 0:
                    logger.info(f"Running Stage 2 validation at step {global_step_s2}...")
                    val_metrics = run_validation_stage2(vlm, tokenizer, val_loader, device, config)
                    wandb.log({"global_step_s2": global_step_s2, **val_metrics})
                    vlm.train()

                global_step_s2 += 1

    save_checkpoint(vlm.adapter.state_dict(), optimizer_s2.state_dict(), -1, global_step_s2, config, 2, "adapter_final")
    
    logger.info("--- Finished Training ---")
    wandb.finish()

@app.local_entrypoint()
def main():
    """Local entrypoint to run the training on Modal."""
    print("Starting Modal training job...")
    config = TrainConfig()
    config.qformer_hidden_dim = 640
    config.qformer_num_layers = 12
    train.remote(config)
