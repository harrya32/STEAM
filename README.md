# Improving the Generation and Evaluation of Synthetic Data for Downstream Medical Causal Inference

This repository contains the implementation for the NeurIPS 2025 paper: **"Improving the Generation and Evaluation of Synthetic Data for Downstream Medical Causal Inference"**.

## Overview

This repository provides:
- **Proposed metrics**: Novel evaluation metrics for assessing synthetic data quality for causal inference tasks (see `src/metrics.py`)
- **Generation methods**: STEAM, our proposed synthetic data generation approach (see `src/generation.py`)
- **Experimental reproducibility**: Code to reproduce all experimental results from the paper (see `exps/` folder)

## Repository structure

```
steam/
├── src/                          # Core implementation
│   ├── metrics.py                # Proposed evaluation metrics
│   ├── generation.py             # Generic and STEAM synthetic data generation methods
│   └── catenets_dp/              # Differentially private PO estimators
├── exps/                         # Experimental scripts
│   ├── simulated_generic_comparisons.py
│   ├── real_generic_comparisons.py
│   ├── existing_metric_failures.py
│   ├── hyperparameter_stability.py
│   ├── ablation.py
│   └── cgms/                     # CGMs experiments
├── data/                         # Data loading and preprocessing
├──results/                       # Results from experiments
├── requirements.txt              # Generic/STEAM generation dependencies
├── requirements_dcm.txt          # Causal generation dependencies
└── setup.py                      # Package installation
```

## Installation

### Setup for generic comparison experiments

1. Create and activate a conda environment:
```bash
conda create -n steam python=3.10
conda activate steam
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install .
```

### Setup for CGM generation

For the CGM-specific experiments, use the dedicated requirements file:

```bash
conda create -n steam_cgm python=3.10
conda activate steam_cgm
pip install -r requirements_cgm.txt
pip install .
```

## Data preparation

Before running experiments, prepare the data:

```bash
cd data
python load_data.py
cd ..
```

## Running experiments

### Standard experiments

Navigate to the experiments folder and run the desired experiment script:

```bash
cd exps
python metric_exp.py       # Existing metric analysis
python real_generic_exp.py       # Real data generic v STEAM
python simulated_generic_exp.py  # Simulated data generic v STEAM
python hyperparameter_exp.py       # Stability analysis
python ablation_exp.py                       # Joint generation of X,W ablation study
python classifier_ablation_exp.py     # Q_W|X classifier ablation
```
By default, the experiments will run for all model/data/iterations combinations from the paper. Specific settings can be made with the CLI.

### CGM experiments

For CGM-specific experiments, generate the data with:

```bash
cd exps/cgms
python causal_generation.py
cd ..
```
By default, the experiments will run for all model/data/iterations combinations from the paper. Specific settings can be made with the CLI.

After generating the CGM data, compare with real data via our metrics:

```bash
conda activate steam
python get_cgm_results.py
```

## Citation
```bibtex
@inproceedings{
  amad2025improving,
  title={Improving the generation and evaluation of synthetic data for downstream medical causal inference},
  author={Harry Amad and Zhaozhi Qian and Dennis Frauen and Julianna Piskorz and Stefan Feuerriegel and Mihaela van der Schaar},
  booktitle={The 39th Conference on Neural Information Processing Systems},
  year={2025}
}
```
