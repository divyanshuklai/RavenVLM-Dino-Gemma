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
    # python
    def test_forward_labels_and_masks_basic(monkeypatch):
        import src.models.caption_modelling as mod
        torch.manual_seed(42)

        class CaptBatchEncoding:
            def __init__(self, input_ids):
                self.input_ids = input_ids

            def to(self, device):
                self.input_ids = self.input_ids.to(device)
                return self

        class CaptFakeTokenizer:
            def __init__(self):
                self.boi_token_id = 1
                self.eoi_token_id = 2
                self.bos_token_id = 3
                self.eos_token_id = 4
                self.pad_token_id = 0
                self.last_caption_ids = None
                self.last_attention_mask = None

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def __call__(self, texts, return_tensors="pt", padding="max_length", truncation=True, max_length=128):
                if isinstance(texts, str):
                    # Ensure prompt is a single token so expansion in model works
                    ids = torch.tensor([[7]], dtype=torch.long)
                    return CaptBatchEncoding(ids)
                # captions path
                B = len(texts)
                L = max_length
                input_ids = torch.arange(10, 10 + L, dtype=torch.long).unsqueeze(0).repeat(B, 1)
                attention_mask = torch.ones(B, L, dtype=torch.long)
                self.last_caption_ids = input_ids.clone()
                self.last_attention_mask = attention_mask.clone()
                return {"input_ids": input_ids, "attention_mask": attention_mask}

            def batch_decode(self, ids, skip_special_tokens=True):
                return ["decoded"] * ids.size(0)

        class CaptFakeLM(nn.Module):
            def __init__(self, hidden=16, vocab=200):
                super().__init__()
                self.emb = nn.Embedding(vocab, hidden)
                self.last_call = {}

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def get_input_embeddings(self):
                return self.emb

            def forward(self, inputs_embeds=None, attention_mask=None, labels=None, input_ids=None):
                B = inputs_embeds.size(0)
                T = inputs_embeds.size(1)
                self.last_call = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                loss = torch.tensor(0.0, requires_grad=True)
                logits = torch.zeros(B, T, self.emb.embedding_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                return types.SimpleNamespace(loss=loss, logits=logits)

        class CaptFakeViT6(nn.Module):
            def __init__(self, hidden=12, tokens=6):
                super().__init__()
                self.config = types.SimpleNamespace(hidden_size=hidden)
                self.tokens = tokens

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def forward(self, pixel_values):
                if isinstance(pixel_values, list):
                    B = len(pixel_values)
                    device = torch.device("cpu")
                else:
                    B = pixel_values.size(0)
                    device = pixel_values.device
                torch.manual_seed(42)
                return types.SimpleNamespace(
                    last_hidden_state=torch.randn(B, self.tokens, self.config.hidden_size, device=device)
                )

        monkeypatch.setattr(mod, "AutoTokenizer", CaptFakeTokenizer)
        monkeypatch.setattr(mod, "AutoModelForCausalLM", CaptFakeLM)
        monkeypatch.setattr(mod, "AutoModel", CaptFakeViT6)

        model = mod.GemmaDinoImageCaptioner(
            include_cls=True, include_registers=False, include_patches=False, max_caption_length=4
        )

        images = torch.randn(2, 3, 8, 8)
        captions = ["cap1", "cap2"]

        out = model(images, captions)
        assert hasattr(out, "loss")

        lm = model.gemma
        inputs_embeds = lm.last_call["inputs_embeds"]
        attn_mask = lm.last_call["attention_mask"]
        labels = lm.last_call["labels"]

        # With ViT tokens=6 and flags (True, False, False):
        # selected_count = 1, image_embed_len = selected + boi + eoi + bos + prompt = 1 + 4 = 5
        image_embed_len = 5
        cap_len = 4
        assert inputs_embeds.shape[1] == image_embed_len + cap_len

        # Labels: first image_embed_len positions are -100, rest are caption ids
        assert torch.all(labels[:, :image_embed_len] == -100)
        assert labels[:, image_embed_len:].shape[1] == cap_len
        assert torch.equal(labels[:, image_embed_len:], model.gemma_tokenizer.last_caption_ids.to(labels.device))

        # Attention mask: ones over image embeds then tokenizer mask
        assert torch.all(attn_mask[:, :image_embed_len] == 1)
        assert torch.equal(attn_mask[:, image_embed_len:], model.gemma_tokenizer.last_attention_mask.to(attn_mask.device))


    def test_forward_image_token_selection_variants(monkeypatch):
        import src.models.caption_modelling as mod
        torch.manual_seed(42)

        class CaptBatchEncoding:
            def __init__(self, input_ids):
                self.input_ids = input_ids

            def to(self, device):
                self.input_ids = self.input_ids.to(device)
                return self

        class CaptFakeTokenizer:
            def __init__(self):
                self.boi_token_id = 1
                self.eoi_token_id = 2
                self.bos_token_id = 3
                self.eos_token_id = 4
                self.pad_token_id = 0
                self.last_caption_ids = None
                self.last_attention_mask = None

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def __call__(self, texts, return_tensors="pt", padding="max_length", truncation=True, max_length=128):
                if isinstance(texts, str):
                    ids = torch.tensor([[7]], dtype=torch.long)
                    return CaptBatchEncoding(ids)
                B = len(texts)
                L = max_length
                input_ids = torch.arange(20, 20 + L, dtype=torch.long).unsqueeze(0).repeat(B, 1)
                attention_mask = torch.ones(B, L, dtype=torch.long)
                self.last_caption_ids = input_ids.clone()
                self.last_attention_mask = attention_mask.clone()
                return {"input_ids": input_ids, "attention_mask": attention_mask}

        class CaptFakeLM(nn.Module):
            def __init__(self, hidden=24, vocab=300):
                super().__init__()
                self.emb = nn.Embedding(vocab, hidden)
                self.last_call = {}

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def get_input_embeddings(self):
                return self.emb

            def forward(self, inputs_embeds=None, attention_mask=None, labels=None, input_ids=None):
                B = inputs_embeds.size(0)
                T = inputs_embeds.size(1)
                self.last_call = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, "labels": labels}
                loss = torch.tensor(0.0, requires_grad=True)
                logits = torch.zeros(B, T, self.emb.embedding_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                return types.SimpleNamespace(loss=loss, logits=logits)

        class CaptFakeViT10(nn.Module):
            def __init__(self, hidden=18, tokens=10):
                super().__init__()
                self.config = types.SimpleNamespace(hidden_size=hidden)
                self.tokens = tokens

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def forward(self, pixel_values):
                if isinstance(pixel_values, list):
                    B = len(pixel_values)
                    device = torch.device("cpu")
                else:
                    B = pixel_values.size(0)
                    device = pixel_values.device
                torch.manual_seed(42)
                return types.SimpleNamespace(
                    last_hidden_state=torch.randn(B, self.tokens, self.config.hidden_size, device=device)
                )

        monkeypatch.setattr(mod, "AutoTokenizer", CaptFakeTokenizer)
        monkeypatch.setattr(mod, "AutoModelForCausalLM", CaptFakeLM)
        monkeypatch.setattr(mod, "AutoModel", CaptFakeViT10)

        combos = [
            (False, False, False, 0),  # select 0 tokens
            (True, False, False, 1),   # select CLS
            (True, True, False, 5),    # CLS + 4 REG
            (True, True, True, 10),    # CLS + 4 REG + 5 patches
        ]
        max_caption_length = 3

        for include_cls, include_registers, include_patches, selected_count in combos:
            model = mod.GemmaDinoImageCaptioner(
                include_cls=include_cls,
                include_registers=include_registers,
                include_patches=include_patches,
                max_caption_length=max_caption_length,
            )

            images = torch.randn(1, 3, 8, 8)
            captions = ["x"]
            out = model(images, captions)
            assert hasattr(out, "loss")

            lm = model.gemma
            inputs_embeds = lm.last_call["inputs_embeds"]
            labels = lm.last_call["labels"]

            # image_embed_len = selected_count + boi + eoi + bos + prompt = selected_count + 4
            image_embed_len = selected_count + 4
            assert inputs_embeds.shape[1] == image_embed_len + max_caption_length

            assert torch.all(labels[:, :image_embed_len] == -100), f"Failed for flags {(include_cls, include_registers, include_patches)}"