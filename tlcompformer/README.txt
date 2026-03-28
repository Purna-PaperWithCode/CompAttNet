CompAttNet: Composition Attention Network for Materials Property Prediction
=====================================================================

This repository contains the official implementation of CompAttNet, developed for manuscript submission to Next Materials.

The framework predicts material properties directly from chemical composition using an attention-based neural architecture. The implementation supports multiple scalar regression tasks, including:

- Formation energy per atom
- Band gap prediction
- Experimental band gap correction
- Other composition-driven material properties

---------------------------------------------------------------------
1. PROJECT DIRECTORY STRUCTURE
---------------------------------------------------------------------

tlcompformer/
│
├── __pycache__/                Cached Python bytecode
├── checkpoints/               Saved training checkpoints
├── configs/                   YAML configuration files
├── data/                      Input datasets and processed splits
├── deployment/                Deployment-ready scripts and artifacts
├── Figures/                   Publication-quality generated figures
├── inference/                 Inference and prediction scripts
├── models/                    Model architecture definitions
├── results/                   Main experimental outputs
├── Results_Personal/          Custom experiment outputs
├── training/                  Training engine and utilities
├── utils/                     Helper functions and shared tools
│
├── run_pretrain.py            Base pretraining pipeline
├── run_pretrain_aflow.py      Attention-flow pretraining
├── run_finetune.py            General fine-tuning pipeline
├── run_finetune_ef.py         Formation-energy fine-tuning
├── evaluate.py               Model evaluation script
├── test_eval_ef.py           Formation-energy evaluation test
│
├── requirements.txt
└── README.txt

---------------------------------------------------------------------
2. INSTALLATION
---------------------------------------------------------------------

Step 1: Create Python environment

    conda create -n compattnet python=3.10 -y
    conda activate compattnet

Step 2: Install all dependencies

    pip install -r requirements.txt

---------------------------------------------------------------------
3. DATASET FORMAT
---------------------------------------------------------------------

Prepare the dataset as a CSV file using the following structure:

    formula,target
    SiO2,8.90
    Al2O3,7.80
    ZnO,3.30
    TiO2,3.00

Field descriptions:

- formula : Chemical composition formula
- target  : Target material property value

Recommended storage location:

data/raw/train.csv
data/raw/test.csv
data/raw/validation.csv

---------------------------------------------------------------------
Project: CompAttNet / TLCompFormer