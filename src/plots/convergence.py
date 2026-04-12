import argparse
import json
import os
import matplotlib.pyplot as plt

from src.utils.experiments import build_base_path
from src.utils.plots import build_results_path, save_figure


def load_history(base_path):
    history_path = os.path.join(base_path, "history.json")
    with open(history_path, "r") as f:
        data = json.load(f)
    return data["rmse_train"], data["rmse_val"]


def plot_convergence(history_train, history_val, ax, xlabel=True, ylabel=True):
    epochs = range(1, len(history_train) + 1)

    ax.plot(epochs, history_train, color='black', linewidth=1.5, label='Entrenamiento')
    ax.plot(epochs, history_val, color='black', linewidth=1.5, linestyle='--', label='Validación')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Quitar grid
    ax.grid(False)

    # Marcas cada 10 épocas
    ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.04))

    if xlabel:
        ax.set_xlabel('Época')
    else:
        ax.set_xticklabels([])

    if ylabel:
        ax.set_ylabel('RMSE')
    else:
        ax.set_yticklabels([])

    ax.legend(frameon=False)


def main(args):
    if args.local_epochs is not None:
        # FL
        model_path = build_base_path(args, "FL")
        model_name = "FL"
    else:
        # MF
        model_path = build_base_path(args, "MF")
        model_name = "MF"

    history_train, history_val = load_history(model_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_convergence(history_train, history_val, ax)
    plt.tight_layout()

    results_path = build_results_path(args)
    save_figure(fig, results_path, f"convergence_{model_name}_{args.dataset}_{args.local_epochs}")
    plt.close()

    print(f"Grafico de convergencia guardado en {results_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--local_epochs", type=int)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--reg", type=float, required=True)
    ap.add_argument("--rounds", type=int, required=True)

    args = ap.parse_args()

    main(args)