import sys
import os
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from models.language_model import build_gemma_model_and_tokenizer, generate_text_basic_stream
import torch


model, tokenizer = build_gemma_model_and_tokenizer("cache/models", device="cpu")
prompt = "The Biggest Star in the Milky Way Galaxy is "
input_ids = tokenizer.encode(prompt)
input_ids = torch.tensor(input_ids).unsqueeze(0)

print(f"Prompt : {prompt}")
for token in generate_text_basic_stream(model, input_ids, max_new_tokens=64):
    token_id = token.squeeze(0).tolist()
    print(tokenizer.decode(token_id), end="", flush=True)