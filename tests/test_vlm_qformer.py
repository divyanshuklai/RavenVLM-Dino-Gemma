import unittest
import torch

from models.qformer import QFormer
from models.language_model import Gemma3Model
from models.vision_language_model import VLMQFormer, VisualPrefixAdapter


def tiny_gemma_cfg(
    vocab_size=101,
    emb_dim=32,
    n_heads=4,
    n_layers=2,
    hidden_dim=64,
    head_dim=8,
    sliding_window=8,
    dtype=torch.float32,
):
    return {
        "vocab_size": vocab_size,
        "context_length": 512,
        "emb_dim": emb_dim,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "head_dim": head_dim,
        "qk_norm": True,
        "n_kv_groups": 1,
        "rope_local_base": 10_000.0,
        "rope_base": 1_000_000.0,
        "sliding_window": sliding_window,
        "layer_types": ["sliding_attention", "full_attention"] * (n_layers // 2) + ( ["sliding_attention"] if n_layers % 2 else []),
        "dtype": dtype,
        "query_pre_attn_scalar": head_dim,
    }


class VLMQFormerStage1Tests(unittest.TestCase):
    def setUp(self) -> None:
        # QFormer dims
        self.q_embed = 32
        self.num_blocks = 2
        self.num_heads = 4
        self.num_queries = 3
        self.vision_dim = 48

        # Build components
        self.qformer = QFormer(
            embed_dim=self.q_embed,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            num_queries=self.num_queries,
            vision_dim=self.vision_dim,
            lm_vocab_size=200,
        )

        # Tiny Gemma (only used for cfg in stage1)
        self.gemma = Gemma3Model(tiny_gemma_cfg(emb_dim=self.q_embed, dtype=torch.float32))

        # Identity adapter (no LN) to check scaling behavior
        adapter = VisualPrefixAdapter(qformer_dim=self.q_embed, gemma_emb_dim=self.q_embed, use_ln=False)
        with torch.no_grad():
            adapter.proj.weight.copy_(torch.eye(self.q_embed))

        self.model = VLMQFormer(
            qformer=self.qformer,
            gemma=self.gemma,
            adapter=adapter,
            freeze_gemma=True,
            prefix_requires_grad=True,
        )
        self.bs = 2
        self.nv = 5

    def test_stage1_prefix_embeddings_shape_and_scaling(self):
        vis = torch.randn(self.bs, self.nv, self.vision_dim)
        # get raw queries via stage1 (queries path)
        queries = self.model(vis, mode="stage1", prefix_return_embeddings=False)
        self.assertEqual(tuple(queries.shape), (self.bs, self.num_queries, self.q_embed))

        # get prefix embeddings (should be identity-projected and scaled by sqrt(emb_dim))
        prefix = self.model(vis, mode="stage1", prefix_return_embeddings=True)
        self.assertEqual(tuple(prefix.shape), (self.bs, self.num_queries, self.q_embed))

        scale = (self.q_embed ** 0.5)
        self.assertTrue(torch.allclose(prefix, queries * scale, atol=1e-5, rtol=1e-5))


class VLMQFormerStage2Tests(unittest.TestCase):
    def setUp(self) -> None:
        # Dims
        self.q_embed = 32
        self.num_blocks = 2
        self.num_heads = 4
        self.num_queries = 4
        self.vision_dim = 40

        # QFormer
        self.qformer = QFormer(
            embed_dim=self.q_embed,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            num_queries=self.num_queries,
            vision_dim=self.vision_dim,
            lm_vocab_size=300,
        )

        # Tiny Gemma for fast tests
        self.vocab = 97
        gemma_cfg = tiny_gemma_cfg(vocab_size=self.vocab, emb_dim=self.q_embed, n_layers=2, dtype=torch.float32)
        self.gemma = Gemma3Model(gemma_cfg)

        # Simple adapter (default LN OK)
        self.adapter = VisualPrefixAdapter(qformer_dim=self.q_embed, gemma_emb_dim=self.q_embed, use_ln=True)

        self.model = VLMQFormer(
            qformer=self.qformer,
            gemma=self.gemma,
            adapter=self.adapter,
            freeze_gemma=True,           # verify gemma is frozen
            prefix_requires_grad=True,   # adapter trainable
        )

        self.bs = 2
        self.nv = 6
        self.T = 7

    def test_stage2_logits_shape_and_grads(self):
        torch.manual_seed(0)
        vis = torch.randn(self.bs, self.nv, self.vision_dim)
        input_ids = torch.randint(0, self.vocab, (self.bs, self.T), dtype=torch.long)

        logits = self.model(vis, input_ids=input_ids, mode="stage2")
        self.assertEqual(tuple(logits.shape), (self.bs, self.num_queries + self.T, self.vocab))

        # quick backward check
        loss = logits.sum()
        loss.backward()

        # qformer params should have grads
        q_grads = [p.grad for p in self.qformer.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in q_grads))

        # adapter params should have grads
        a_grads = [p.grad for p in self.adapter.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in a_grads))

        # gemma is frozen, grads should be None
        g_grads = [p.grad for p in self.gemma.parameters()]
        self.assertTrue(all(g is None for g in g_grads))

    def test_reset_gemma_kv_cache_safe(self):
        # Should not raise
        self.model.reset_gemma_kv_cache()


if __name__ == "__main__":
    unittest.main()
