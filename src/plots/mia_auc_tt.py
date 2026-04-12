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
    all_results = {}

    fl_path = build_base_path(args, "FL")
    all_results["FL"] = load_classifier_results(fl_path)

    for noise in args.noise_multipliers:
        args.noise_multiplier = noise
        dp_path = build_base_path(args, "FL_DP")
        all_results[f"FL_DP\n(σ={noise})"] = load_classifier_results(dp_path)

    models     = list(all_results.keys())
    x          = np.arange(len(models))
    width      = 0.25

    train_aucs = [all_results[m].get("train_auc") for m in models]
    test_aucs  = [all_results[m].get("test_auc")  for m in models]

    # Separate models with/without train_auc (threshold attack doesn't produce it)
    has_train_auc = [v is not None for v in train_aucs]

    def dynamic_ylim(values, margin=0.002):
        all_vals = [v for v in values if v is not None]
        lo = min(min(all_vals), 0.5) - margin * 10
        hi = max(max(all_vals), 0.5) + margin * 10
        return lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.15

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "MIA Classifier: Train AUC vs Test AUC",
        fontsize=14, fontweight="bold",
    )

    # Side-by-side bars for models that have train_auc
    x_with    = [xi for xi, has in zip(x, has_train_auc) if has]
    ta_with   = [v  for v,  has in zip(train_aucs, has_train_auc) if has]
    tea_with  = [v  for v,  has in zip(test_aucs,  has_train_auc) if has]
    x_without = [xi for xi, has in zip(x, has_train_auc) if not has]
    tea_wo    = [v  for v,  has in zip(test_aucs,  has_train_auc) if not has]

    if x_with:
        ax.bar([xi - width / 2 for xi in x_with], ta_with,  width,
               label="Train AUC", color="steelblue", alpha=0.85, zorder=3)
        ax.bar([xi + width / 2 for xi in x_with], tea_with, width,
               label="Test AUC",  color="tomato",    alpha=0.85, zorder=3)
    if x_without:
        ax.bar(x_without, tea_wo, width * 1.6,
               label="Test AUC" if not x_with else "_nolegend_",
               color="tomato", alpha=0.85, zorder=3)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2,
               label="Random baseline (0.5)", zorder=2)

    # Annotate values and delta
    for xi, test_val, train_val, has in zip(x, test_aucs, train_aucs, has_train_auc):
        if has:
            # Train AUC label: inside bar near top to avoid clash with delta
            ax.text(xi - width / 2, train_val - 0.0003, f"{train_val:.3f}",
                    ha="center", va="top", fontsize=8, color="white", fontweight="bold")
            # Test AUC label: above its bar
            ax.text(xi + width / 2, test_val + 0.0001, f"{test_val:.3f}",
                    ha="center", va="bottom", fontsize=8, color="tomato")
            # Delta badge: above the taller bar with extra clearance
            gap  = train_val - test_val
            sign = "+" if gap >= 0 else ""
            ax.text(xi, max(train_val, test_val) + 0.0006,
                    f"Δ={sign}{gap:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color="dimgray",
                    bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                              ec="gray", alpha=0.8))
        else:
            ax.text(xi, test_val + 0.0001, f"{test_val:.3f}",
                    ha="center", va="bottom", fontsize=8, color="tomato")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("AUC")
    ax.set_ylim(*dynamic_ylim([v for v in train_aucs + test_aucs if v is not None]))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Note for models without train_auc
    if not all(has_train_auc):
        missing = [m.replace("\n", " ") for m, has in zip(models, has_train_auc) if not has]
        ax.annotate(
            f"⚠ Train AUC not available for: {', '.join(missing)}\n"
            "(threshold attack does not produce train AUC)",
            xy=(0.01, 0.02), xycoords="axes fraction",
            fontsize=8, color="gray",
            bbox=dict(boxstyle="round", fc="white", ec="lightgray", alpha=0.8),
        )

    plt.tight_layout()
    output_path = build_results_path(args)
    save_figure(plt, output_path, "mia_auc_train_vs_test")
    plt.close()

    print(f"\nSaved Train AUC vs Test AUC plot to {output_path}")


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