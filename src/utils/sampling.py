"""Sampling utilities (top-k / top-p) and latent interpolation helpers."""
import torch
import torch.nn.functional as F


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., -1, None]] = -float('Inf')
    return out


def sample_from_logits(logits, temperature=1.0, top_k=None):
    logits = logits / temperature
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
