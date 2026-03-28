import torch
import csv
from torch.utils.data import DataLoader

from models.model import TLCompFormerLite
from data.property_dataset import PropertyDataset
from training.trainer import Trainer


def pretrain_model(
    data_root="data/processed",
    batch_size=16,
    epochs=3,
    lr=3e-4,
    device=None
):
    """
    Corrected thermodynamic pretraining:
    - Target normalization
    - Cosine LR scheduler
    - CSV logging (reviewer-friendly)
    - Windows-safe
    """

    # ---- DEVICE ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # ---- DATASETS (TRAIN ONLY) ----
    ef_train = PropertyDataset(
        f"{data_root}/formation_energy/train.csv",
        normalize=True
    )
    eh_train = PropertyDataset(
        f"{data_root}/energy_above_hull/train.csv",
        normalize=True
    )

    print("EF train size:", len(ef_train))
    print("EH train size:", len(eh_train))
    print("Starting training loop...")

    ef_loader = DataLoader(
        ef_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    eh_loader = DataLoader(
        eh_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # ---- MODEL ----
    model = TLCompFormerLite()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5
    )

    trainer = Trainer(model, device=device)

    # ---- LOGGING ----
    log_path = "checkpoints/pretrain_log.csv"
    log_file = open(log_path, "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["epoch", "avg_train_mae"])

    # ---- TRAIN LOOP ----
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0

        print(f"\nEpoch {epoch}/{epochs}")

        # ---- Formation Energy ----
        for i, batch in enumerate(ef_loader):
            loss = trainer.train_step(batch, "ef", optimizer)
            total_loss += loss
            steps += 1

            if i % 1000 == 0:
                print(f"  [EF] batch {i}, loss = {loss:.4f}")

        # ---- Energy Above Hull ----
        for i, batch in enumerate(eh_loader):
            loss = trainer.train_step(batch, "ehull", optimizer)
            total_loss += loss
            steps += 1

            if i % 200 == 0:
                print(f"  [EHull] batch {i}, loss = {loss:.4f}")

        avg_loss = total_loss / steps
        print(f"Epoch [{epoch}/{epochs}] | Avg Train MAE (normalized): {avg_loss:.4f}")

        logger.writerow([epoch, avg_loss])
        log_file.flush()

        scheduler.step()

    # ---- SAVE MODEL ----
    torch.save(model.state_dict(), "checkpoints/pretrained.pt")
    log_file.close()

    print("\nPretraining finished.")
    print("Model saved to checkpoints/pretrained.pt")
    print("Training log saved to checkpoints/pretrain_log.csv")
