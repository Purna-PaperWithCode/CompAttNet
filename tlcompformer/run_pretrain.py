from training.pretrain import pretrain_model

if __name__ == "__main__":
    pretrain_model(
        data_root="data/processed",
        batch_size=16,
        epochs=2
    )
