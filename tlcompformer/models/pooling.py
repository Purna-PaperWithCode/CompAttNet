import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.attn = nn.Linear(emb_dim, 1)

    def forward(self, x):
        """
        x: (B, N, D)
        """
        weights = torch.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)
