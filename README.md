# Federated Matrix Factorization with Differential Privacy

Implementation of a federated recommender system based on Matrix Factorization (MF) with Differential Privacy (DP) guarantees, developed as part of an undergraduate thesis at Universidad Autónoma de Madrid.

## Models

- **MF**: Centralized Matrix Factorization baseline
- **FL**: Federated Matrix Factorization
- **FL-DP**: Federated Matrix Factorization with Differential Privacy (Gaussian mechanism + RDP accounting)

## Datasets

Experiments are conducted on [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) and [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/).

## Pipeline

1. Create data splits
2. Hyperparameter search
3. Train models
4. Evaluate RMSE and privacy budget ε

All experiments are reproducible via seeds.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Create splits
./scripts/splits/run_splits.sh

# Hyperparameter search
./scripts/search/run_hparam_search.sh

# Train
./scripts/model/run_training.sh

# Evaluate
./scripts/model/run_evaluations.sh
```

## Project Structure

```
src/
├── models/       # MF, FL and FL-DP model implementations
├── training/     # Training scripts
├── evaluation/   # RMSE evaluation
├── plots/        # Result visualization
├── data/         # Data loading and splitting
└── utils/        # Shared utilities
scripts/
├── splits/       # Data split scripts
├── search/       # Hyperparameter search scripts
├── model/        # Training and evaluation scripts
└── plots/        # Plot generation scripts
```
