import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.utils.experiments import build_base_path
from src.utils.evaluation import load_evaluation, load_classifier_results
from src.utils.plots import build_results_path, save_figure


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def fmt(value, decimals=4):
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def collect_row(model_label, base_path, model_type):
    rmse_train, rmse_test, eps = load_evaluation(base_path)
    clf = load_classifier_results(base_path)

    return {
        "Model":        model_label,
        "Type":         model_type,
        "ε (epsilon)":  fmt(eps, 2) if eps is not None else "∞",
        "RMSE Train":   fmt(rmse_train),
        "RMSE Test":    fmt(rmse_test),
        "Attack":       clf.get("attack_type", "?").replace("_torch", "").replace("_", " "),
        "Train Acc":    fmt(clf.get("train_accuracy")),
        "Test Acc":     fmt(clf.get("test_accuracy")),
        "Train AUC":    fmt(clf.get("train_auc")),
        "Test AUC":     fmt(clf.get("test_auc")),
    }


# Column groups for coloring headers
COL_GROUPS = {
    "Model":       "model",
    "Type":        "model",
    "ε (epsilon)": "privacy",
    "RMSE Train":  "utility",
    "RMSE Test":   "utility",
    "Attack":      "attack",
    "Train Acc":   "attack",
    "Test Acc":    "attack",
    "Train AUC":   "attack",
    "Test AUC":    "attack",
}

GROUP_COLORS = {
    "model":   "#dce8f5",
    "privacy": "#d5f0dd",
    "utility": "#fff3cd",
    "attack":  "#fde8e8",
}


# -------------------------------------------------
# Main
# -------------------------------------------------

def main(args):
    rows = []

    fl_path = build_base_path(args, "FL")
    rows.append(collect_row("FL", fl_path, "Federated"))

    for noise in args.noise_multipliers:
        args.noise_multiplier = noise
        dp_path = build_base_path(args, "FL_DP")
        rows.append(collect_row(f"FL_DP (σ={noise})", dp_path, "Federated + DP"))

    columns = list(COL_GROUPS.keys())
    cell_data = [[row[c] for c in columns] for row in rows]

    n_rows = len(rows)
    n_cols = len(columns)

    fig_w = max(14, n_cols * 1.5)
    fig_h = n_rows * 0.55 + 0.7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    fig.suptitle(
        "Summary: Utility & MIA Privacy Results",
        fontsize=14, fontweight="bold", y=0.98,
    )

    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for col_idx, col_name in enumerate(columns):
        cell = table[0, col_idx]
        cell.set_facecolor(GROUP_COLORS[COL_GROUPS[col_name]])
        cell.set_text_props(fontweight="bold", fontsize=9)

    # Style data rows
    for row_idx in range(n_rows):
        row_data = rows[row_idx]
        is_dp = row_data["Type"] == "Federated + DP"
        row_bg = "#f9f9f9" if row_idx % 2 == 0 else "#ffffff"

        for col_idx, col_name in enumerate(columns):
            cell = table[row_idx + 1, col_idx]
            cell.set_facecolor(row_bg)

            # AUC closer to 0.5 = better privacy = more green
            if col_name in ("Test AUC", "Train AUC") and row_data[col_name] != "—":
                try:
                    val = float(row_data[col_name])
                    distance = abs(val - 0.5)
                    max_dist  = 0.015          # tighter range: 0=green, >=0.015=white
                    t = max(0.0, 1.0 - distance / max_dist)   # 1 = perfect privacy
                    # interpolate white -> green
                    r = 1.0 - t * 0.45
                    g = 1.0 - t * 0.05
                    b = 1.0 - t * 0.45
                    cell.set_facecolor(mcolors.to_hex((r, g, b)))
                except ValueError:
                    pass

    # Group label annotations above header
    group_spans = {}
    for col_idx, col_name in enumerate(columns):
        g = COL_GROUPS[col_name]
        if g not in group_spans:
            group_spans[g] = [col_idx, col_idx]
        else:
            group_spans[g][1] = col_idx

    plt.tight_layout()
    output_path = build_results_path(args)
    save_figure(plt, output_path, "mia_summary_table")
    plt.close()

    print(f"\nSaved summary table to {output_path}")


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