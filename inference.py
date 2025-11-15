# inference.py
import modal
import os
import torch
import torch.nn.functional as F
import wandb
import logging
import json
from typing import Tuple, List, Dict
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import h5py
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from models.config import TrainConfig
from data.COCOCaptions import CocoEmbedDataset, custom_collate_fn
from models.qformer import QFormer
from models.language_model import build_gemma_model_and_tokenizer
from models.vision_language_model import VLMQFormer, VisualPrefixAdapter
from models.config import CACHE_PATH

# --- Modal Setup ---
app = modal.App(name="raven-qformer-inference")

cache_vol = modal.Volume.from_name("cache")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("default-jdk")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("evals", remote_path="/root/evals")
    .add_local_dir("models", remote_path="/root/models")
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_captions_for_cider(gts: Dict[str, List[str]], res: Dict[str, List[str]]):
    """Formats ground truth and generated captions for pycocoevalcap."""
    gts_fmt = {k: [{'caption': c} for c in v] for k, v in gts.items()}
    res_fmt = {k: [{'caption': v[0]}] if isinstance(v, list) else [{'caption': v}] for k, v in res.items()}

    tokenizer = PTBTokenizer()
    gts_tokenized = tokenizer.tokenize(gts_fmt)
    res_tokenized = tokenizer.tokenize(res_fmt)
    return gts_tokenized, res_tokenized

@torch.no_grad()
def run_inference(vlm: VLMQFormer, tokenizer, test_loader: DataLoader, device: torch.device, config: TrainConfig) -> Tuple[Dict[str, float], List[Dict]]:
    """Runs inference, calculating CIDEr score and collecting results."""
    vlm.eval()
    cider_scorer = Cider()
    gts_cider, res_cider = {}, {}
    results_data = []
    sample_idx = 0

    for batch in tqdm(test_loader, desc="Inference", leave=False):
        patch_embeds, _, _, _, raw_captions_batch, patch_attention_mask = batch
        patch_embeds, patch_attention_mask = patch_embeds.to(device, dtype=torch.bfloat16), patch_attention_mask.to(device, dtype=torch.bool)
        
        generated_texts = vlm.generate(vis_emb=patch_embeds, tokenizer=tokenizer, max_new_tokens=config.max_text_len, eos_token_id=tokenizer.eos_token_id, device=device, patch_attention_mask=patch_attention_mask)
        
        for i, gen_text in enumerate(generated_texts):
            # The sample_idx corresponds to the index in the unshuffled dataset
            actual_sample_id = test_loader.dataset.sample_ids[sample_idx]
            
            # For CIDEr calculation
            res_cider[str(sample_idx)] = [gen_text]
            gts_cider[str(sample_idx)] = raw_captions_batch[i]
            
            # For JSON output
            results_data.append({
                "sample_id": actual_sample_id,
                "generated_caption": gen_text,
                "ground_truth_captions": raw_captions_batch[i]
            })
            
            sample_idx += 1

    gts_tokenized, res_tokenized = format_captions_for_cider(gts_cider, res_cider)
    cider_score, _ = cider_scorer.compute_score(gts_tokenized, res_tokenized)
    
    return {"test/cider_score": cider_score}, results_data

@app.function(image=image, gpu="A100", volumes={CACHE_PATH: cache_vol}, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=3600)
def inference(run_id: str, config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Starting inference for run {run_id} with config:\n{config.to_dict()}")
    
    logger.info("Loading models...")
    gemma_model, tokenizer = build_gemma_model_and_tokenizer(config.gemma_model_path)
    
    qformer_core = QFormer(
        embed_dim=config.qformer_hidden_dim, 
        num_blocks=config.qformer_num_layers, 
        num_heads=config.qformer_num_heads, 
        num_queries=config.qformer_num_query_tokens, 
        vision_dim=config.vit_embed_dim, 
        lm_vocab_size=tokenizer.get_vocab_size()
    ).to(device, dtype=torch.bfloat16)

    qformer_checkpoint_path = Path(config.output_dir) / run_id / "stage_1" / "qformer_final_latest.pt"
    logger.info(f"Loading QFormer checkpoint from {qformer_checkpoint_path}")
    qformer_checkpoint = torch.load(qformer_checkpoint_path, map_location=device)
    qformer_core.load_state_dict(qformer_checkpoint['model_state_dict'])

    adapter = VisualPrefixAdapter(qformer_dim=config.qformer_hidden_dim, gemma_emb_dim=gemma_model.cfg["emb_dim"]).to(device, dtype=torch.bfloat16)
    adapter_checkpoint_path = Path(config.output_dir) / run_id / "stage_2" / "adapter_final_latest.pt"
    logger.info(f"Loading Adapter checkpoint from {adapter_checkpoint_path}")
    adapter_checkpoint = torch.load(adapter_checkpoint_path, map_location=device)
    adapter.load_state_dict(adapter_checkpoint['model_state_dict'])

    vlm = VLMQFormer(qformer=qformer_core, gemma=gemma_model.to(device, dtype=torch.bfloat16), adapter=adapter, freeze_gemma=True, stage="stage2").to(device)

    logger.info("Loading test dataset...")
    # The test set is a single file, but we point to the directory and specify the split
    # The dataset class will construct the path: /cache/datasets/COCO-captions-vits16plus-embed/test.h5
    test_dataset = CocoEmbedDataset(
        data_path="/cache/datasets/COCO-captions-vits16plus-embed",
        split='test',
        tokenizer=tokenizer,
        max_text_len=config.max_text_len,
        same_size=False, # It's a single file, but not in the 'same_size' format
        deterministic_caption=True, # Important for consistent evaluation
        seed=config.seed
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size_diff, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    logger.info("Running inference...")
    metrics, results_data = run_inference(vlm, tokenizer, test_loader, device, config)
    
    logger.info(f"Inference complete. Metrics: {metrics}")
    print(f"CIDEr Score: {metrics['test/cider_score']}")

    # Save results to a JSON file
    inference_dir = Path(config.output_dir) / run_id / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    save_path = inference_dir / "results.json"
    
    with open(save_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    logger.info(f"Saved inference results to {save_path}")

@app.local_entrypoint()
def main():
    """Local entrypoint to run inference on Modal."""
    print("Starting Modal inference job...")
    config = TrainConfig()
    # Override config with the values from the training run
    config.qformer_hidden_dim = 640
    config.qformer_num_layers = 12
    
    run_id = "ewy64sj9"
    inference.remote(run_id, config)
