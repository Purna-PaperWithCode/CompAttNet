import torch
import numpy as np
from torch.utils.data import DataLoader
from models.model import TLCompFormerLite
from data.property_dataset import PropertyDataset


def evaluate_property(
    checkpoint_path,
    csv_path,
    task,
    batch_size=64,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PropertyDataset(csv_path, normalize=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = TLCompFormerLite()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            Z = batch["Z"].to(device)
            frac = batch["frac"].to(device)

            pred = model(Z, frac, task=task)

            pred = pred.cpu().numpy() * dataset.std + dataset.mean
            target = batch["target"].cpu().numpy() * dataset.std + dataset.mean

            preds.append(pred)
            targets.append(target)

    preds = np.vstack(preds).flatten()
    targets = np.vstack(targets).flatten()

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {"MAE": mae, "RMSE": rmse, "R2": r2}
