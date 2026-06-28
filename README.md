# CompAttNet: Composition Attention Network for Materials Property Prediction

This repository contains the official implementation of **CompAttNet**, a lightweight composition-attention framework developed for predicting thermodynamic and electronic properties of inorganic materials directly from chemical compositions. The implementation accompanies the manuscript submitted to **Next Materials**.

CompAttNet predicts material properties without requiring explicit crystallographic information and supports multiple composition-based regression tasks, including:

- Formation Energy per Atom
- Electronic Band Gap
- Experimental Band Gap Correction
- Other Composition-Based Material Property Prediction Tasks

---

# 1. Repository Structure

```text
tlcompformer/
│
├── checkpoints/                  Saved pretrained and fine-tuned models
├── configs/                      Configuration files
├── data/                         Processed datasets
├── deployment/                   Deployment utilities
├── inference/                    Prediction scripts
├── models/                       Model architectures
├── results/                      Experimental results
├── training/                     Training utilities
├── utils/                        Helper functions
│
├── Final_Dataset_preparation.ipynb
├── ML_Models_For_MP_Prediction.ipynb
├── run_pretrain.py
├── run_pretrain_aflow.py
├── run_finetune.py
├── run_finetune_ef.py
├── evaluate.py
├── test_eval_ef.py
├── requirements.txt
└── README.md
```

---

# 2. Installation

Create a Python environment

```bash
conda create -n compattnet python=3.10 -y
conda activate compattnet
```

Install all dependencies

```bash
pip install -r requirements.txt
```

---

# 3. Dataset Preparation

The original datasets were obtained from the **Materials Project** and **AFLOW** repositories.

Dataset preprocessing is performed using

```text
Final_Dataset_preparation.ipynb
```

The preprocessing workflow consists of:

- Extract chemical compositions
- Extract target property values
- Remove incomplete entries
- Remove duplicate compositions
- Normalize stoichiometric formulas
- Generate processed CSV datasets

The processed datasets are stored under

```text
data/
```

---

# 4. Dataset Format

The input dataset should follow the CSV format shown below.

```csv
formula,target
SiO2,8.90
Al2O3,7.80
ZnO,3.30
TiO2,3.00
```

where

| Field | Description |
|------|-------------|
| formula | Chemical composition formula |
| target | Target material property |

Example directory structure

```text
data/
├── formation_energy/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
└── bandgap/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

---

# 5. Data Partition

The processed datasets are partitioned using the following split.

| Dataset | Percentage |
|---------|-----------:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The same data partition is employed throughout all experiments to ensure reproducibility and fair comparison.

---

# 6. Hyperparameter Configuration

## Pretraining

| Hyperparameter | Value |
|---------------|-------|
| Batch Size | 16 |
| Epochs | 3 |
| Learning Rate | 3 × 10⁻⁴ |
| Optimizer | AdamW |
| Learning Rate Scheduler | CosineAnnealingLR |
| Minimum Learning Rate | 1 × 10⁻⁵ |
| Data Normalization | Enabled |
| Data Shuffling | Enabled |

## Fine-Tuning

| Hyperparameter | Value |
|---------------|-------|
| Batch Size | 32 |
| Epochs | 5 |
| Learning Rate | 1 × 10⁻⁴ |
| Optimizer | AdamW |
| Data Normalization | Enabled |
| Data Shuffling | Enabled |
| Fine-Tuning Strategy | Freeze the backbone and train only the task-specific prediction head |

## Device Selection

| Parameter | Configuration |
|-----------|---------------|
| Device | CUDA (GPU) if available; otherwise CPU |

---

# 7. Model Training

Pretraining

```bash
python run_pretrain.py
```

AFLOW Pretraining

```bash
python run_pretrain_aflow.py
```

Fine-Tuning

```bash
python run_finetune.py
```

Formation Energy Fine-Tuning

```bash
python run_finetune_ef.py
```

---

# 8. Model Evaluation

```bash
python evaluate.py
```

Formation Energy Evaluation

```bash
python test_eval_ef.py
```

---

# 9. Experimental Workflow

```text
Materials Project / AFLOW
            │
            ▼
Dataset Preparation
            │
            ▼
Data Preprocessing
            │
            ▼
Model Pretraining
            │
            ▼
Task-Specific Fine-Tuning
            │
            ▼
Model Evaluation
            │
            ▼
Material Property Prediction
```

---

# 10. Repository Outputs

The repository generates:

- Pretrained model checkpoints
- Fine-tuned model checkpoints
- Training logs
- Prediction results
- Publication-quality figures
- Evaluation metrics

---

# 11. Software Requirements

- Python 3.10
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

Additional software dependencies are listed in

```text
requirements.txt
```

---

# 12. Reproducibility

This repository provides a complete workflow for reproducing the reported experiments, including:

- Dataset preparation notebook
- Processed training, validation, and testing datasets
- Training and fine-tuning scripts
- Evaluation scripts
- Hyperparameter configuration
- Software dependency specification
- Model checkpoints
- Complete documentation

---

# Citation

If you use this repository in your research, please cite the associated publication.

**CompAttNet: Composition Attention Network for Materials Property Prediction**

(The complete citation will be updated after publication.)

---

# License

This repository is released for academic and research purposes.
