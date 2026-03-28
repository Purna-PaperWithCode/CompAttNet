import torch.nn as nn
from models.encoder import ChemicalEncoder
from models.pooling import AttentionPooling
from models.heads import RegressionHead, ClassificationHead


class TLCompFormerLite(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()

        self.encoder = ChemicalEncoder(
            emb_dim=emb_dim,
            layers=4,
            heads=8
        )

        self.pool = AttentionPooling(emb_dim)

        # Bottleneck compression
        self.bottleneck = nn.Linear(emb_dim, emb_dim // 2)

        # Pretraining heads
        self.ef_head = RegressionHead(emb_dim // 2)
        self.ehull_head = RegressionHead(emb_dim // 2)
        self.stability_head = ClassificationHead(emb_dim // 2)

    def forward(self, Z, frac, task):
        """
        task: 'ef', 'ehull', or 'stability'
        """
        x = self.encoder(Z, frac)
        x = self.pool(x)
        x = self.bottleneck(x)

        if task == "ef":
            return self.ef_head(x)
        elif task == "ehull":
            return self.ehull_head(x)
        elif task == "stability":
            return self.stability_head(x)
        else:
            raise ValueError(f"Unknown task: {task}")
