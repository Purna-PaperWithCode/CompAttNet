import torch.nn as nn
from models.embedding import ElementEmbedding, StoichiometricEncoding


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim)
        )

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class ChemicalEncoder(nn.Module):
    def __init__(self, emb_dim=128, layers=4, heads=8):
        super().__init__()

        self.elem_emb = ElementEmbedding(emb_dim=emb_dim)
        self.stoich_emb = StoichiometricEncoding(emb_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, heads)
            for _ in range(layers)
        ])

    def forward(self, Z, frac):
        """
        Z: (B, N)
        frac: (B, N)
        """
        x = self.elem_emb(Z) + self.stoich_emb(frac)

        for block in self.blocks:
            x = block(x)

        return x
