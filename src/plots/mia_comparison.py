import argparse
import matplotlib.pyplot as plt
import numpy as np

from src.utils.experiments import build_base_path
from src.utils.evaluation import load_classifier_results
from src.utils.plots import build_results_path, save_figure


# -------------------------------------------------
# Main
# -------------------------------------------------

def main(args):
    fl_eval_path = build_base_path(args, "FL")
    fl_results = load_classifier_results(fl_eval_path)

    all_results = {"FL": fl_results}

    for noise in args.noise_multipliers:
        args.noise_multiplier = noise
        dp_eval_path = build_base_path(args, "FL_DP")
        dp_results = load_classifier_results(dp_eval_path)
        label = f"FL_DP\n(σ={noise})"
        all_results[label] = dp_results

    models = list(all_results.keys())
    x = np.arange(len(models))
    width = 0.25

    train_accs = [all_results[m].get("train_accuracy", None) for m in models]
    test_accs  = [all_results[m].get("test_accuracy",  None) for m in models]
    test_aucs  = [all_results[m].get("test_auc",       None) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("MIA Classifier Results Comparison (FL vs FL_DP)", fontsize=14, fontweight="bold")

    def dynamic_ylim(values, margin=0.002):
        """Zoom Y axis to actual value range with a small margin around 0.5."""
        all_vals = [v for v in values if v is not None]
        lo = min(min(all_vals), 0.5) - margin * 10
        hi = max(max(all_vals), 0.5) + margin * 10
        return lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.3

    # --- Subplot 1: Accuracy (train vs test) ---
    ax = axes[0]
    bars1 = ax.bar(x - width / 2, train_accs, width, label="Train Accuracy", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, test_accs,  width, label="Test Accuracy",  color="tomato",    alpha=0.85)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Random baseline (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Attack Accuracy")
    ax.set_ylim(*dynamic_ylim(train_accs + test_accs))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # --- Subplot 2: Test AUC ---
    ax = axes[1]
    bars3 = ax.bar(x, test_aucs, width * 1.8, color="steelblue", alpha=0.85)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Random baseline (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title("Attack Test AUC")
    ax.set_ylim(*dynamic_ylim(test_aucs))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars3:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    output_path = build_results_path(args)
    save_figure(plt, output_path, "mia_comparison")
    plt.close()

    print(f"\nSaved MIA comparison plot to {output_path}")


# -------------------------------------------------
# CLI
# -------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset",      type=str,   required=True)
    ap.add_argument("--k",            type=int,   required=True)
    ap.add_argument("--lr",           type=float, required=True)
    ap.add_argument("--local_epochs", type=int)
    ap.add_argument("--batch_size",   type=int,   required=True)
    ap.add_argument("--reg",          type=float, required=True)
    ap.add_argument("--rounds",       type=int,   required=True)

    ap.add_argument(
        "--noise_multipliers",
        type=float,
        nargs="+",
        required=True,
        help="List of noise multipliers for FL_DP",
    )

    args = ap.parse_args()
    main(args)