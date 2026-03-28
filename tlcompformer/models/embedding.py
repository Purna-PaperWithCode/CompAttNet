import torch
import torch.nn as nn
import math


class ElementEmbedding(nn.Module):
    def __init__(self, num_elements=118, emb_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_elements + 1, emb_dim)

    def forward(self, Z):
        """
        Z: LongTensor of shape (B, N)
        """
        return self.embedding(Z)


class StoichiometricEncoding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, frac):
        """
        frac: FloatTensor of shape (B, N)
        """
        device = frac.device
        div_term = torch.exp(
            torch.arange(0, self.emb_dim, 2, device=device)
            * -(math.log(10000.0) / self.emb_dim)
        )

        pe = torch.zeros(
            frac.size(0), frac.size(1), self.emb_dim, device=device
        )

        pe[..., 0::2] = torch.sin(frac.unsqueeze(-1) * div_term)
        pe[..., 1::2] = torch.cos(frac.unsqueeze(-1) * div_term)

        return pe
