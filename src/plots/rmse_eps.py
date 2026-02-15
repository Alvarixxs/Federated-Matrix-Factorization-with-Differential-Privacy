import argparse
import matplotlib.pyplot as plt

from src.utils.experiments import build_base_path
from src.utils.evaluation import load_evaluation
from src.utils.plots import build_results_path, save_figure


# -------------------------------------------------
# Main
# -------------------------------------------------

def main(args):

    epsilons = []
    rmse_tests = []
    rmse_trains = []

    for noise in args.noise_multipliers:
        args.noise_multiplier = noise

        dp_eval_path = build_base_path(args, "FL_DP")
        dp_train, dp_test, eps = load_evaluation(dp_eval_path)

        if eps is None:
            print(f"Warning: epsilon not found for noise={noise}")
            continue

        epsilons.append(eps)
        rmse_tests.append(dp_test)
        rmse_trains.append(dp_train)

    # Sort by epsilon (important for clean plot)
    sorted_data = sorted(zip(epsilons, rmse_tests, rmse_trains))
    epsilons, rmse_tests, rmse_trains = zip(*sorted_data)

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    plt.figure(figsize=(8, 6))

    plt.plot(epsilons, rmse_tests, 'o-', linewidth=2, markersize=8, label="Test RMSE")
    plt.plot(epsilons, rmse_trains, 's--', linewidth=2, markersize=6, label="Train RMSE")

    plt.xlabel("Privacy Budget (ε)")
    plt.ylabel("RMSE")
    plt.title("Utility–Privacy Trade-off (FL_DP)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    # Optional: invert x-axis if you want smaller epsilon on right
    # plt.gca().invert_xaxis()

    plt.tight_layout()

    output_path = build_results_path(args)
    save_figure(plt, output_path, "epsilon_vs_rmse")
    plt.close()

    print(f"\nSaved epsilon vs RMSE plot to {output_path}")


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
