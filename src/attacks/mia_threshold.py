import argparse
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils.experiments import build_base_path
from src.utils.evaluation import load_mia_dataset, save_attack_results


# -------------------------------------------------
# Threshold search
# -------------------------------------------------

def find_best_threshold(scores, labels):
    """
    Finds threshold that maximizes accuracy on training set.
    Lower score = more member-like (for error feature).
    """

    thresholds = np.unique(scores)
    best_acc = 0.0
    best_t = thresholds[0]

    for t in thresholds:
        preds = (scores <= t).astype(int)
        acc = accuracy_score(labels, preds)

        if acc > best_acc:
            best_acc = acc
            best_t = t

    return best_t, best_acc


# -------------------------------------------------
# Main
# -------------------------------------------------

def main(args):
    base_path = build_base_path(args, args.model)
    train_df, test_df = load_mia_dataset(base_path)

    feature = args.feature

    train_scores = train_df[feature].values
    train_labels = train_df["label"].values

    test_scores = test_df[feature].values
    test_labels = test_df["label"].values

    # -------------------------------------------------
    # Learn threshold on training data
    # -------------------------------------------------
    threshold, train_acc = find_best_threshold(train_scores, train_labels)

    # -------------------------------------------------
    # Evaluate on test
    # -------------------------------------------------
    test_preds = (test_scores <= threshold).astype(int)
    test_acc = accuracy_score(test_labels, test_preds)

    # For AUC: convert to member-like score
    # If feature is "error", smaller = more member-like
    if feature == "error":
        member_scores = -test_scores
    else:
        member_scores = test_scores

    test_auc = roc_auc_score(test_labels, member_scores)

    results = {
        "attack_type": "threshold",
        "feature_used": feature,
        "threshold": float(threshold),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
    }

    save_attack_results(base_path, results)

    print("\nThreshold attack completed.")


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

    ap.add_argument(
        "--feature",
        type=str,
        default="error",
        choices=["error", "score", "norm_p", "norm_q"]
    )

    args = ap.parse_args()

    main(args)
