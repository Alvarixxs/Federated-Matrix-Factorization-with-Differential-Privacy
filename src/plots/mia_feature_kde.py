import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from src.utils.experiments import build_base_path
from src.utils.plots import build_results_path, save_figure


FEATURES = ["score", "error", "norm_p", "norm_q"]

MEMBER_COLOR     = "steelblue"
NONMEMBER_COLOR  = "tomato"


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def plot_kde(ax, values, color, label):
    values = values.dropna()
    if len(values) < 2:
        return
    kde = gaussian_kde(values)
    xs  = np.linspace(values.min(), values.max(), 300)
    ax.plot(xs, kde(xs), color=color, linewidth=2, label=label)
    ax.fill_between(xs, kde(xs), alpha=0.15, color=color)


def load_features_for_model(base_path):
    attack_path = os.path.join(base_path, "mia_attack")
    members    = pd.read_csv(os.path.join(attack_path, "mia_in_features.csv"))
    nonmembers = pd.read_csv(os.path.join(attack_path, "mia_out_features.csv"))
    return members, nonmembers


# -------------------------------------------------
# Main
# -------------------------------------------------

def main(args):
    # Build model list
    models = {}

    fl_path = build_base_path(args, "FL")
    models["FL"] = fl_path

    for noise in args.noise_multipliers:
        args.noise_multiplier = noise
        dp_path = build_base_path(args, "FL_DP")
        models[f"FL_DP (σ={noise})"] = dp_path

    n_models   = len(models)
    n_features = len(FEATURES)

    # Grid: rows = models, cols = features
    fig, axes = plt.subplots(
        n_models, n_features,
        figsize=(4 * n_features, 3 * n_models),
        squeeze=False,
    )

    fig.suptitle(
        "MIA Feature Distributions: Members vs Non-Members",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for row, (model_label, base_path) in enumerate(models.items()):
        members, nonmembers = load_features_for_model(base_path)

        for col, feature in enumerate(FEATURES):
            ax = axes[row][col]

            plot_kde(ax, members[feature],    MEMBER_COLOR,    "Member")
            plot_kde(ax, nonmembers[feature], NONMEMBER_COLOR, "Non-member")

            if row == 0:
                ax.set_title(feature, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(model_label, fontsize=10)

            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_yticks([])

            # Legend only in the top-right subplot
            if row == 0 and col == n_features - 1:
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color=MEMBER_COLOR,    linewidth=2, label="Member"),
                    Line2D([0], [0], color=NONMEMBER_COLOR, linewidth=2, label="Non-member"),
                ]
                ax.legend(handles=legend_elements, fontsize=8, loc="upper right",
                          frameon=True, framealpha=0.9)

    plt.tight_layout()
    output_path = build_results_path(args)
    save_figure(plt, output_path, "mia_feature_kde")
    plt.close()

    print(f"\nSaved MIA feature KDE plot to {output_path}")


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