import argparse
import os
import json
import torch
import pandas as pd
import numpy as np

from src.utils.experiments import build_base_path
from src.utils.evaluation import load_mia_dataset, save_classifier_results


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def standardize(train_tensor, test_tensor):
    mean = train_tensor.mean(dim=0, keepdim=True)
    std = train_tensor.std(dim=0, keepdim=True) + 1e-8
    return (train_tensor - mean) / std, (test_tensor - mean) / std


def compute_auc(scores, labels):
    """
    Rank-based AUC (Mann–Whitney formulation).
    """
    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    order = np.argsort(scores)
    ranks = np.argsort(order)

    pos = labels == 1
    n_pos = pos.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    auc = (ranks[pos].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return float(auc)


def compute_accuracy(probs, labels):
    preds = (probs >= 0.5).float()
    return float((preds == labels).float().mean())


# -------------------------------------------------
# Logistic Regression Model
# -------------------------------------------------

class LogisticAttack(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main(args):
    device = torch.device("cpu")

    base_path = build_base_path(args, args.model)
    
    train_df, test_df = load_mia_dataset(base_path)

    features = ["score", "error", "squared_error", "norm_p", "norm_q", "reg_loss", "centered_score"]

    X_train = torch.tensor(train_df[features].values, dtype=torch.float32, device=device)
    y_train = torch.tensor(train_df["label"].values, dtype=torch.float32, device=device)

    X_test = torch.tensor(test_df[features].values, dtype=torch.float32, device=device)
    y_test = torch.tensor(test_df["label"].values, dtype=torch.float32, device=device)

    # -----------------------------
    # Standardization
    # -----------------------------
    X_train, X_test = standardize(X_train, X_test)

    # -----------------------------
    # Model
    # -----------------------------
    model = LogisticAttack(input_dim=X_train.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.attack_lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(args.attack_epochs):
        model.train()

        logits = model(X_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # -----------------------------
    # Evaluation
    # -----------------------------
    model.eval()

    with torch.no_grad():
        train_logits = model(X_train)
        test_logits = model(X_test)

        train_probs = torch.sigmoid(train_logits)
        test_probs = torch.sigmoid(test_logits)

        train_acc = compute_accuracy(train_probs, y_train)
        test_acc = compute_accuracy(test_probs, y_test)

        train_auc = compute_auc(train_probs, y_train)
        test_auc = compute_auc(test_probs, y_test)

    results = {
        "attack_type": "logistic_regression_torch",
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_auc": train_auc,
        "test_auc": test_auc,
        "attack_epochs": args.attack_epochs,
        "attack_lr": args.attack_lr,
        "features_used": features,
    }

    save_classifier_results(base_path, results)

    print("\nTorch classifier attack completed.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)

    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--local_epochs", type=int)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--reg", type=float, required=True)
    ap.add_argument("--rounds", type=int, required=True)

    ap.add_argument("--noise_multiplier", type=float)

    ap.add_argument("--attack_lr", type=float, default=1e-2)
    ap.add_argument("--attack_epochs", type=int, default=200)

    args = ap.parse_args()

    main(args)
