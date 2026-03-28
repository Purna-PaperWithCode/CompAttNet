import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.to(device)

    def train_step(self, batch, task, optimizer):
        self.model.train()

        Z = batch["Z"].to(self.device)
        frac = batch["frac"].to(self.device)
        y = batch["target"].to(self.device)

        pred = self.model(Z, frac, task=task)
        loss = F.l1_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def val_step(self, batch, task):
        self.model.eval()

        Z = batch["Z"].to(self.device)
        frac = batch["frac"].to(self.device)
        y = batch["target"].to(self.device)

        pred = self.model(Z, frac, task=task)
        loss = F.l1_loss(pred, y)

        return loss.item()
