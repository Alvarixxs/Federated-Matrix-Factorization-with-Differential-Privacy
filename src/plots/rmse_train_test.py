import argparse
import matplotlib.pyplot as plt

from src.utils.experiments import build_base_path
from src.utils.evaluation import load_evaluation
from src.utils.plots import build_results_path, save_figure


# -------------------------------------------------
# Main
# -------------------------------------------------

def main(args):
    mf_eval_path = build_base_path(args, "MF")
    mf_train, mf_test, _ = load_evaluation(mf_eval_path)

    fl_eval_path = build_base_path(args, "FL")
    fl_train, fl_test, _ = load_evaluation(fl_eval_path)

    models = ["MF", "FL"]
    train_values = [mf_train, fl_train]
    test_values = [mf_test, fl_test]

    for noise in args.noise_multipliers:
        args.noise_multiplier = noise
        dp_eval_path = build_base_path(args, "FL_DP")
        dp_train, dp_test, eps = load_evaluation(dp_eval_path)

        label = f"FL_DP (σ={noise})"
        if eps is not None:
            label += f"\nε={round(eps,2)}"

        models.append(label)
        train_values.append(dp_train)
        test_values.append(dp_test)


    x = range(len(models))
    width = 0.35

    plt.figure(figsize=(12, 6))

    # Train
    plt.plot(x, train_values, 'o-', linewidth=2, markersize=8, label="Train RMSE")

    # Test
    plt.plot(x, test_values, 's-', linewidth=2, markersize=8, label="Test RMSE")

    plt.xticks(x, models, rotation=20)
    plt.ylabel("Final RMSE")
    plt.title("Final RMSE Comparison (MF vs FL vs FL_DP)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path = build_results_path(args)
    save_figure(plt, output_path, "rmse_comparison")
    plt.close()

    print(f"\nSaved final RMSE comparison to {output_path}")


# -------------------------------------------------
# CLI
# -------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, required=True)

    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--local_epochs", type=int)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--reg", type=float, required=True)
    ap.add_argument("--rounds", type=int, required=True)

    ap.add_argument(
        "--noise_multipliers",
        type=float,
        nargs="+",
        required=True,
        help="List of noise multipliers for FL_DP"
    )

    args = ap.parse_args()

    main(args)
