from huggingface_hub import hf_hub_download
import os

# Create cache directory
os.makedirs("cache/models", exist_ok=True)

try:
    # Download with explicit timeout and resume capability
    weights_file = hf_hub_download(
        repo_id="google/gemma-3-270m",
        filename="model.safetensors", 
        cache_dir="cache/models",
        resume_download=True,
        local_files_only=False
    )
    print(f"Downloaded to: {weights_file}")
    
    tokenizer_file = hf_hub_download(
        repo_id="google/gemma-3-270m",
        filename="tokenizer.json",
        cache_dir="cache/models", 
        resume_download=True,
        local_files_only=False
    )
    print(f"Tokenizer downloaded to: {tokenizer_file}")
    
except Exception as e:
    print(f"Download failed: {e}")