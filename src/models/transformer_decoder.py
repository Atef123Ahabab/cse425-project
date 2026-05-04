"""Transformer decoder model (Task 3 minimal scaffold).
"""
import torch
import torch.nn as nn


class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size=129, embed_dim=256, nhead=8, num_layers=4, dim_feedforward=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size - 1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory=None):
        # autoregressive decoding uses causal mask; for scaffold we won't use memory
        emb = self.embedding(tgt) * (emb_scale := (self.embedding.embedding_dim ** 0.5))
        emb = emb.permute(1, 0, 2)  # seq_len, batch, dim
        seq_len = emb.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=emb.device), diagonal=1).bool()
        out = self.decoder(emb, emb, tgt_mask=mask)
        out = out.permute(1, 0, 2)
        logits = self.output_fc(out)
        return logits
