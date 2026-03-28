from training.finetune_ef import finetune_formation_energy

if __name__ == "__main__":
    finetune_formation_energy(
        data_root="data/processed",
        batch_size=32,
        epochs=5
    )
