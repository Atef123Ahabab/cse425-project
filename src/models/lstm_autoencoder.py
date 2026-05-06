"""LSTM Autoencoder model (Task 1).

Simple, minimal implementation for scaffold and smoke tests.
"""
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, vocab_size=129, embed_dim=32, hidden_dim=64, latent_dim=32, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size - 1)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.latent_fc = nn.Linear(hidden_dim, latent_dim)
        self.latent_fc_inv = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.output_fc = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.encoder(emb)
        h = h_n[-1]
        z = self.latent_fc(h)
        return z

    def decode(self, z, seq_len):
        # simple decoder: initialize hidden from latent and run zeros input tokens
        h0 = self.latent_fc_inv(z).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        batch = z.size(0)
        device = z.device
        inputs = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
        emb = self.embedding(inputs)
        out, _ = self.decoder(emb, (h0, c0))
        logits = self.output_fc(out)
        return logits

    def forward(self, x):
        z = self.encode(x)
        logits = self.decode(z, x.size(1))
        return logits
