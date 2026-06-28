CompAttNet: Composition Attention Network for Materials Property Prediction
===========================================================================

This repository contains the official implementation of CompAttNet, developed
for the manuscript submitted to Next Materials.

CompAttNet is a lightweight composition-attention framework that predicts
material properties directly from chemical compositions without requiring
explicit crystal structural information.

The framework supports multiple regression tasks including:

• Formation Energy per Atom
• Band Gap Prediction
• Experimental Band Gap Correction
• Other Composition-Based Material Property Prediction

---------------------------------------------------------------------
1. PROJECT DIRECTORY STRUCTURE
---------------------------------------------------------------------

<img width="378" height="342" alt="image" src="https://github.com/user-attachments/assets/a0c6c6eb-0716-45af-b3b8-65ebd921c4ea" />

---------------------------------------------------------------------
2. INSTALLATION
---------------------------------------------------------------------

Create a Python environment

    conda create -n compattnet python=3.10 -y

Activate the environment

    conda activate compattnet

Install dependencies

    pip install -r requirements.txt

---------------------------------------------------------------------
3. DATASET PREPARATION
---------------------------------------------------------------------

The original datasets were collected from

• Materials Project
• AFLOW Database

Dataset preprocessing was performed using

    Final_Dataset_preparation.ipynb

The preprocessing workflow consists of

• Extract chemical compositions
• Extract target property values
• Remove incomplete entries
• Remove duplicate compositions
• Normalize stoichiometric formulas
• Generate processed CSV files

The processed datasets are stored under

    data/

---------------------------------------------------------------------
4. DATASET FORMAT
---------------------------------------------------------------------

The input dataset should follow CSV format.

Example

formula,target
SiO2,8.90
Al2O3,7.80
ZnO,3.30
TiO2,3.00

where

formula : Chemical composition

target  : Material property value

Example directory

data/
    formation_energy/
        train.csv
        validation.csv
        test.csv

    bandgap/
        train.csv
        validation.csv
        test.csv

---------------------------------------------------------------------
5. DATA PARTITION
---------------------------------------------------------------------

The processed datasets are divided into

• Training set (70%)
• Validation set (15%)
• Testing set (15%)

The same data partition is used throughout all experiments to ensure
reproducibility and fair comparison.

---------------------------------------------------------------------
6. HYPERPARAMETER CONFIGURATION
---------------------------------------------------------------------

Default pretraining configuration

• Batch size                : 16
• Epochs                    : 3
• Learning rate             : 3 × 10⁻⁴
• Optimizer                 : AdamW
• LR Scheduler              : CosineAnnealingLR
• Minimum Learning Rate     : 1 × 10⁻⁵
• Data normalization        : Enabled
• Data shuffling            : Enabled

Default fine-tuning configuration

• Batch size                : 32
• Epochs                    : 5
• Learning rate             : 1 × 10⁻⁴
• Optimizer                 : AdamW
• Data normalization        : Enabled
• Data shuffling            : Enabled
• Fine-tuning strategy      : Freeze backbone and train only the task-specific prediction head

Device selection

• CUDA (GPU) if available
• CPU otherwise

---------------------------------------------------------------------
7. TRAINING
---------------------------------------------------------------------

Pretraining

    python run_pretrain.py

AFLOW pretraining

    python run_pretrain_aflow.py

Fine-tuning

    python run_finetune.py

Formation-energy fine-tuning

    python run_finetune_ef.py

---------------------------------------------------------------------
8. MODEL EVALUATION
---------------------------------------------------------------------

Evaluate the trained model

    python evaluate.py

Formation-energy evaluation

    python test_eval_ef.py

---------------------------------------------------------------------
9. REPOSITORY WORKFLOW
---------------------------------------------------------------------

Materials Project / AFLOW

            │

            ▼

Final_Dataset_preparation.ipynb

            │

            ▼

Processed CSV Dataset

            │

            ▼

Model Pretraining

            │

            ▼

Task-specific Fine-tuning

            │

            ▼

Model Evaluation

            │

            ▼

Prediction Results

---------------------------------------------------------------------
10. OUTPUT FILES
---------------------------------------------------------------------

Generated outputs include

• Pretrained models
• Fine-tuned models
• Training logs
• Prediction results
• Publication-quality figures

---------------------------------------------------------------------
11. SOFTWARE REQUIREMENTS
---------------------------------------------------------------------

Python 3.10

PyTorch

NumPy

Pandas

Scikit-learn

Matplotlib

Additional dependencies are listed in

requirements.txt

---------------------------------------------------------------------
12. REPRODUCIBILITY
---------------------------------------------------------------------

The repository provides

• Dataset preparation scripts
• Processed training, validation, and testing datasets
• Complete training and evaluation scripts
• Hyperparameter configuration
• Environment specification
• Model checkpoints
• Documentation for reproducing the reported experiments

---------------------------------------------------------------------
Project
---------------------------------------------------------------------

CompAttNet
Composition Attention Network for Materials Property Prediction

Developed for the manuscript submitted to Next Materials.
