import torch
import torch.nn as nn

class QFormerBlock(nn.Module):
    def __init__(self, embed_dim : int, num_heads : int, add_cross_attn : bool, 
                 vision_dim : int, num_queries : int):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.vision_dim = vision_dim
        self.has_cross_attn = add_cross_attn
        #joint self attention module
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        #cross attention module
        if add_cross_attn:
            self.ln2 = nn.LayerNorm(embed_dim)
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, kdim=vision_dim, vdim=vision_dim, batch_first=True)
        #queries FFN
        self.lnq = nn.LayerNorm(embed_dim)
        self.qffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)            
        )
        #text FFN
        self.lnt = nn.LayerNorm(embed_dim)
        self.tffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)            
        )

    def forward(self, vis_enc : torch.Tensor, learnt_queries : torch.Tensor, 
                use_text : bool = False, * , text_enc : torch.Tensor | None = None, 
                mask : torch.LongTensor | torch.BoolTensor | None = None,
                key_padding_mask : torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not use_text:
            #self attn
            x = self.ln1(self.self_attn(learnt_queries, learnt_queries, learnt_queries, need_weights=False)[0] + learnt_queries)
            #cross attn
            if self.has_cross_attn:
                x = self.ln2(self.cross_attn(x, vis_enc, vis_enc, key_padding_mask=key_padding_mask, need_weights=False)[0] + x)
            #feed forward
            x = self.lnq(self.qffn(x) + x)
            return x, None
        
        # validate inputs required for text path
        if text_enc is None or mask is None:
            raise ValueError("text_enc and mask must be provided when use_text=True")

        #combined self attn
        x = torch.cat([learnt_queries, text_enc], dim=1)
        x = self.ln1(self.self_attn(x, x, x, attn_mask=mask, need_weights=False)[0] + x)
        
        #decouple learnt queries and text embeds
        learnt_queries, text_enc = x[:, :self.num_queries, :], x[:, self.num_queries:, :]
        
        #cross attn learnt queries <-> vision encodings
        if self.has_cross_attn:
            learnt_queries = self.ln2(self.cross_attn(learnt_queries, vis_enc, vis_enc, key_padding_mask=key_padding_mask, need_weights=False)[0] + learnt_queries)

        #separate feed forward 
        learnt_queries = self.lnq(self.qffn(learnt_queries) + learnt_queries)
        text_enc = self.lnt(self.tffn(text_enc) + text_enc)
        return learnt_queries, text_enc

class QFormer(nn.Module):
    def __init__(self, embed_dim : int, num_blocks : int, num_heads : int,
                 num_queries : int, vision_dim : int, lm_vocab_size : int,
                 padding_idx : int = 0):
        super().__init__()

        assert embed_dim%num_heads == 0, "cannot divide embed_dim into num_heads for attention"

        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.vision_dim = vision_dim
        self.lm_vocab_size = lm_vocab_size

        #convert tokens to embeds
        self.text_embed = nn.Embedding(lm_vocab_size,embed_dim, padding_idx=padding_idx)
        #learnt queries
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        #QFormerBlocks
        self.blocks = nn.ModuleList([
            QFormerBlock(embed_dim, num_heads, True, vision_dim, num_queries)
            if i%2==0 else
            QFormerBlock(embed_dim, num_heads, False, vision_dim, num_queries)
            for i in range(num_blocks)
        ])

    def get_mask(self, mask_type : str, num_embeds :int) -> torch.Tensor:
        device = next(self.parameters()).device
        Lq = self.num_queries
        L = num_embeds
        assert L >= Lq, "num_embeds must be >= num_queries"
        Lt = L - Lq     # Lt : Len text seq 

        if mask_type == "bidirectional-multimodal":
            return torch.zeros((L, L), dtype=torch.bool, device=device)

        elif mask_type == "causal-multimodal":
            mask = torch.zeros((L, L), dtype=torch.bool, device=device)
            if Lt > 0:
                mask[:Lq, Lq:] = True  # block queries->text
                causal = torch.triu(torch.ones((Lt, Lt), dtype=torch.bool, device=device), diagonal=1)
                mask[Lq:, Lq:] = causal 
            return mask

        elif mask_type == "bidirectional-unimodal":
            mask = torch.zeros((L, L), dtype=torch.bool, device=device)
            if Lt > 0:
                mask[:Lq, Lq:] = True  # block queries->text
                mask[Lq:, :Lq] = True  # block text->queries
            return mask

        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")


    def forward(self, vis_enc : torch.Tensor, use_text : bool = False, 
                *, text_tok : torch.LongTensor | None = None, 
                mask_type : str | None = None,
                patch_attn_mask : torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        bs = vis_enc.shape[0]
        queries = self.queries.unsqueeze(0).expand(bs, -1, -1)

        if not use_text:
            for block in self.blocks:
                queries = block(vis_enc, queries, use_text=False, key_padding_mask=patch_attn_mask)[0]
            return queries, 
        
        if text_tok is None or mask_type is None:
            raise ValueError("text_tok and mask_type must be provided when use_text=True")

        text_enc = self.text_embed(text_tok)
        mask = self.get_mask(mask_type, self.num_queries + text_enc.shape[1])
        for block in self.blocks:
            queries, text_enc = block(vis_enc, queries, use_text=True, text_enc=text_enc, mask=mask, key_padding_mask=patch_attn_mask)

        return queries, text_enc

        
        


