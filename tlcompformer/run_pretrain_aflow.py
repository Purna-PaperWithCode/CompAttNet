from training.pretrain import pretrain_model

if __name__ == "__main__":
    pretrain_model(
        data_root="data/processed_aflow",  # 🔴 AFLOW ROOT
        batch_size=32,
        epochs=3,
        experiment_name="aflow"
    )