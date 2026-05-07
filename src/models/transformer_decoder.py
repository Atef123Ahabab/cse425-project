"""Transformer decoder model (Task 3 minimal scaffold).
"""
import torch
import torch.nn as nn
import json
import os


class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size=None, embed_dim=128, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        # Auto-detect vocab size from data/raw/vocab.json if not provided
        if vocab_size is None:
            vocab_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'vocab.json')
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r') as f:
                    vocab = json.load(f)
                vocab_size = len(vocab)
                print(f"Auto-detected vocab size: {vocab_size}")
            else:
                vocab_size = 129  # fallback
        
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
