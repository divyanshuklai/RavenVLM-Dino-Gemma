import unittest
import torch

from models.qformer import QFormer


class QFormerMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embed_dim = 16
        self.num_blocks = 2
        self.num_heads = 4
        self.num_queries = 3
        self.vision_dim = 32
        self.lm_vocab_size = 100
        self.model = QFormer(
            embed_dim=self.embed_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            num_queries=self.num_queries,
            vision_dim=self.vision_dim,
            lm_vocab_size=self.lm_vocab_size,
        )

    def test_mask_bidirectional_multimodal(self):
        # L = Lq + Lt, here Lt=5
        L = self.num_queries + 5
        mask = self.model.get_mask("bidirectional-multimodal", num_embeds=L)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.shape, (L, L))
        # fully unmasked
        self.assertTrue(torch.equal(mask, torch.zeros_like(mask)))

    def test_mask_causal_multimodal(self):
        Lq = self.num_queries
        Lt = 6
        L = Lq + Lt
        mask = self.model.get_mask("causal-multimodal", num_embeds=L)
        self.assertEqual(mask.shape, (L, L))
        # queries -> text is blocked
        self.assertTrue(torch.all(mask[:Lq, Lq:]))
        # text -> queries is allowed (should be unmasked)
        self.assertTrue(torch.all(~mask[Lq:, :Lq]))
        # text causal masking: strictly upper triangle should be True
        expected_text_causal = torch.triu(torch.ones((Lt, Lt), dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.equal(mask[Lq:, Lq:], expected_text_causal))
        # queries self/self and text self diagonal are unmasked (diagonal False)
        self.assertTrue(torch.all(~torch.diag(mask)))

    def test_mask_bidirectional_unimodal(self):
        Lq = self.num_queries
        Lt = 4
        L = Lq + Lt
        mask = self.model.get_mask("bidirectional-unimodal", num_embeds=L)
        self.assertEqual(mask.shape, (L, L))
        # queries <-> text blocked both ways
        self.assertTrue(torch.all(mask[:Lq, Lq:]))
        self.assertTrue(torch.all(mask[Lq:, :Lq]))
        # within queries and within text unmasked (all False)
        self.assertTrue(torch.equal(mask[:Lq, :Lq], torch.zeros((Lq, Lq), dtype=torch.bool)))
        self.assertTrue(torch.equal(mask[Lq:, Lq:], torch.zeros((Lt, Lt), dtype=torch.bool)))


class QFormerForwardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embed_dim = 32
        self.num_blocks = 4
        self.num_heads = 4
        self.num_queries = 5
        self.vision_dim = 64
        self.lm_vocab_size = 200
        self.bs = 2
        self.nv = 8  # number of vision tokens
        self.seq = 7  # text length
        self.model = QFormer(
            embed_dim=self.embed_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            num_queries=self.num_queries,
            vision_dim=self.vision_dim,
            lm_vocab_size=self.lm_vocab_size,
        )

    def test_forward_without_text_returns_queries_only_tuple(self):
        vis_enc = torch.randn(self.bs, self.nv, self.vision_dim)
        # Even though not used, the current API requires keyword-only args
        text_tok = torch.zeros(self.bs, self.seq, dtype=torch.long)
        out = self.model(vis_enc, use_text=False, text_tok=text_tok, mask_type="bidirectional-multimodal")
        # Current implementation returns a single-element tuple
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 1)
        queries = out[0]
        self.assertEqual(tuple(queries.shape), (self.bs, self.num_queries, self.embed_dim))

    def test_forward_with_text_returns_queries_and_text(self):
        vis_enc = torch.randn(self.bs, self.nv, self.vision_dim)
        text_tok = torch.randint(0, self.lm_vocab_size, (self.bs, self.seq), dtype=torch.long)
        queries, text_enc = self.model(vis_enc, use_text=True, text_tok=text_tok, mask_type="causal-multimodal")
        self.assertEqual(tuple(queries.shape), (self.bs, self.num_queries, self.embed_dim))
        self.assertEqual(tuple(text_enc.shape), (self.bs, self.seq, self.embed_dim))
        # gradients should flow: quick backward on sum
        loss = queries.sum() + text_enc.sum()
        loss.backward()
        for p in self.model.parameters():
            self.assertIsNotNone(p.grad)


if __name__ == "__main__":
    unittest.main()
