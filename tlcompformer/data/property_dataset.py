import torch
from torch.utils.data import Dataset
import pandas as pd
from pymatgen.core import Composition


class PropertyDataset(Dataset):
    """
    Generic dataset for composition -> single property
    Includes target normalization (VERY IMPORTANT)
    """

    def __init__(self, csv_path, max_elements=10, normalize=True):
        self.df = pd.read_csv(csv_path)
        self.max_elements = max_elements
        self.normalize = normalize

        if not {"formula", "target"}.issubset(self.df.columns):
            raise ValueError("CSV must contain columns: formula, target")

        # ---- Target normalization ----
        self.targets = self.df["target"].values.astype("float32")
        if self.normalize:
            self.mean = self.targets.mean()
            self.std = self.targets.std() + 1e-8
        else:
            self.mean = 0.0
            self.std = 1.0

    def __len__(self):
        return len(self.df)

    def _parse_formula(self, formula):
        comp = Composition(formula)
        Z = [el.Z for el in comp.elements]
        frac = [comp.get_atomic_fraction(el) for el in comp.elements]
        return Z, frac

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        Z, frac = self._parse_formula(row["formula"])

        # ---- Padding / truncation ----
        if len(Z) > self.max_elements:
            Z = Z[:self.max_elements]
            frac = frac[:self.max_elements]

        pad_len = self.max_elements - len(Z)
        Z += [0] * pad_len
        frac += [0.0] * pad_len

        # ---- Normalize target ----
        target = (row["target"] - self.mean) / self.std

        return {
            "Z": torch.tensor(Z, dtype=torch.long),
            "frac": torch.tensor(frac, dtype=torch.float32),
            "target": torch.tensor([target], dtype=torch.float32)
        }
