import types
import torch
import torch.nn as nn


class FakeTokenizer:
    def __init__(self):
        self.boi_token_id = 1
        self.eoi_token_id = 2
        self.bos_token_id = 3
        self.eos_token_id = 4
        self.pad_token_id = 0

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, texts, return_tensors="pt", padding=True):
        # return a simple ramp of token ids
        max_len = max(len(t) for t in texts)
        input_ids = []
        attn = []
        for i, t in enumerate(texts):
            L = max_len
            ids = torch.arange(5, 5 + L)
            mask = torch.ones(L, dtype=torch.long)
            input_ids.append(ids)
            attn.append(mask)
        return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attn)}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["decoded"] * len(ids)


class FakeImageProcessor:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, images, return_tensors="pt"):
        b = len(images) if isinstance(images, list) else images.shape[0]
        return {"pixel_values": torch.randn(b, 3, 16, 16)}


class FakeHFModel(nn.Module):
    def __init__(self, hidden=32, vocab=50):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden)
        self.emb = nn.Embedding(vocab, hidden)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, input_ids=None):
        B = inputs_embeds.size(0) if inputs_embeds is not None else input_ids.size(0)
        # fake loss
        loss = torch.tensor(0.0, requires_grad=True)
        return types.SimpleNamespace(loss=loss, logits=torch.zeros(B, 1, self.emb.embedding_dim))

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3]])


class FakeViT(nn.Module):
    def __init__(self, hidden=32, tokens=6):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden)
        self.tokens = tokens

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def forward(self, pixel_values):
        B = pixel_values.size(0)
        return types.SimpleNamespace(last_hidden_state=torch.randn(B, self.tokens, self.config.hidden_size))


def test_model_forward_monkeypatch(monkeypatch):
    import src.models.caption_modelling as mod

    monkeypatch.setattr(mod, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(mod, "AutoImageProcessor", FakeImageProcessor)
    monkeypatch.setattr(mod, "AutoModelForCausalLM", FakeHFModel)
    monkeypatch.setattr(mod, "AutoModel", FakeViT)

    model = mod.GemmaDinoImageCaptioner(include_cls=True, include_registers=False, include_patches=False)

    images = [torch.zeros(3, 8, 8) for _ in range(2)]
    captions = ["a", "b"]
    out = model(images, captions)
    assert hasattr(out, "loss")

    gen = model.inference_generate(images, max_new_tokens=1)
    assert isinstance(gen, list) and len(gen) == 1 or len(gen) == 2
