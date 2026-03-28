from evaluate import evaluate_property

metrics = evaluate_property(
    checkpoint_path="checkpoints/pretrained.pt",
    csv_path="data/processed/formation_energy/test.csv",
    task="ef"
)

print("Formation Energy TEST metrics:", metrics)
