import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.utils.experiments import build_base_path
from src.utils.evaluation import load_classifier_results
from src.utils.plots import build_results_path, save_figure


ATTACK_COLORS = {
    "threshold":                 ("darkorange", "orange"),
    "logistic_regression_torch": ("steelblue",  "deepskyblue"),
}

ATTACK_LABELS = {
    "threshold":                 "Threshold",
    "logistic_regression_torch": "Logistic Regression",
}


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def load_all_results(args):
    """Return an ordered dict  {model_label: results_dict}."""
    entries = {}

    fl_path = build_base_path(args, "FL")
    entries["FL"] = load_classifier_results(fl_path)

    for noise in args.noise_multipliers:
        args.noise_multiplier = noise
        dp_path = build_base_path(args, "FL_DP")
        entries[f"FL_DP\n(σ={noise})"] = load_classifier_results(dp_path)

    return entries


# -------------------------------------------------
# Main
# -------------------------------------------------

def main(args):
    all_results = load_all_results(args)

    models     = list(all_results.keys())
    n          = len(models)
    x          = np.arange(n)
    width      = 0.28

    # ── Figure layout: 3 subplots ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("MIA Attack Detail by Model (FL vs FL_DP)", fontsize=14, fontweight="bold")

    metric_defs = [
        ("test_accuracy",  "Test Accuracy"),
        ("test_auc",       "Test AUC"),
        ("train_accuracy", "Train Accuracy"),
    ]

    def dynamic_ylim(values, margin=0.002):
        """Zoom Y axis to actual value range keeping 0.5 always visible."""
        all_vals = [v for v in values if v is not None]
        lo = min(min(all_vals), 0.5) - margin * 10
        hi = max(max(all_vals), 0.5) + margin * 10
        return lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.3

    for ax, (metric_key, metric_title) in zip(axes, metric_defs):
        values      = [r.get(metric_key) for r in all_results.values()]
        attack_types = [r.get("attack_type", "unknown") for r in all_results.values()]

        # Use a single consistent color per attack type across all bars in this subplot.
        # If all models share the same attack type, one color; otherwise per-bar color.
        unique_attacks = list(dict.fromkeys(attack_types))  # preserves order, deduplicates
        if len(unique_attacks) == 1:
            bar_colors = ATTACK_COLORS.get(unique_attacks[0], ("steelblue", "steelblue"))[0]
        else:
            bar_colors = [ATTACK_COLORS.get(at, ("gray", "lightgray"))[0] for at in attack_types]

        bars = ax.bar(x, values, width * 2.2, color=bar_colors, alpha=0.85, edgecolor="white")

        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Random baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(metric_title)
        ax.set_title(metric_title)
        ax.set_ylim(*dynamic_ylim(values))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for bar, val in zip(bars, values):
            if val is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0001,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )

    # Shared legend for attack types
    legend_patches = [
        mpatches.Patch(color=colors[0], alpha=0.85, label=ATTACK_LABELS.get(at, at))
        for at, colors in ATTACK_COLORS.items()
    ]
    legend_patches.append(
        mpatches.Patch(color="none", label="")  # spacer
    )
    fig.legend(
        handles=legend_patches + [
            plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.2,
                       label="Random baseline (0.5)")
        ],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    output_path = build_results_path(args)
    save_figure(plt, output_path, "mia_detail_by_attack")
    plt.close()

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n{'Model':<20} {'Attack type':<30} {'Test Acc':>10} {'Test AUC':>10}")
    print("-" * 75)
    for label, r in all_results.items():
        model_clean = label.replace("\n", " ")
        print(
            f"{model_clean:<20} {r.get('attack_type','?'):<30} "
            f"{r.get('test_accuracy', float('nan')):>10.4f} "
            f"{r.get('test_auc', float('nan')):>10.4f}"
        )
    print(f"\nSaved MIA detail plot to {output_path}")


# -------------------------------------------------
# CLI
# -------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset",      type=str,   required=True)
    ap.add_argument("--k",            type=int,   required=True)
    ap.add_argument("--lr",           type=float, required=True)
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