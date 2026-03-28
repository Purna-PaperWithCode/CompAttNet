import torch
from torch.utils.data import DataLoader
from models.model import TLCompFormerLite
from data.property_dataset import PropertyDataset
from training.trainer import Trainer
import os


def finetune_formation_energy(
    data_root="data/processed",
    batch_size=32,
    epochs=5,
    lr=1e-4,
    device=None
):
    """
    Formation Energy fine-tuning:
    - Load pretrained model
    - Freeze ALL parameters
    - Unfreeze ONLY ef_head
    - Train on Ef data only
    """

    # -------- DEVICE --------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # -------- DATASET --------
    train_dataset = PropertyDataset(
        f"{data_root}/formation_energy/train.csv",
        normalize=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    print("EF fine-tuning dataset size:", len(train_dataset))
    print("Starting EF fine-tuning...")

    # -------- MODEL --------
    model = TLCompFormerLite()
    model.load_state_dict(
        torch.load("checkpoints/pretrained.pt", map_location=device)
    )
    model.to(device)

    # -------- FREEZE EVERYTHING --------
    for param in model.parameters():
        param.requires_grad = False

    # -------- UNFREEZE ONLY EF HEAD --------
    for param in model.ef_head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.ef_head.parameters(),
        lr=lr
    )

    trainer = Trainer(model, device=device)

    # -------- TRAIN LOOP --------
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0

        print(f"\nEpoch {epoch}/{epochs}")

        for i, batch in enumerate(train_loader):
            loss = trainer.train_step(batch, "ef", optimizer)
            total_loss += loss
            steps += 1

            # ---- progress print ----
            if i % 1000 == 0:
                print(f"  [EF fine-tune] batch {i}, loss = {loss:.4f}")

        avg_loss = total_loss / steps
        print(f"Epoch [{epoch}/{epochs}] | EF Fine-tune MAE (normalized): {avg_loss:.4f}")

    # -------- SAVE MODEL --------
    os.makedirs("checkpoints/finetuned", exist_ok=True)
    save_path = "checkpoints/finetuned/ef_finetuned.pt"

    torch.save(model.state_dict(), save_path)

    print("\nEF fine-tuning completed successfully.")
    print(f"Fine-tuned model saved at: {save_path}")
