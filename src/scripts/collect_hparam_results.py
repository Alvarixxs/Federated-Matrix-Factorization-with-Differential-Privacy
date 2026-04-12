import json
import os
import argparse
import numpy as np
from src.utils.experiments import build_base_path


def load_eval(base_path):
    eval_path = os.path.join(base_path, "eval.json")
    if not os.path.exists(eval_path):
        return None
    with open(eval_path) as f:
        return json.load(f)


def collect_over_seeds(args, model, cfg, seeds, noise_multiplier=None):
    rmse_trains = []
    rmse_vals = []
    rmse_tests = []

    for seed in seeds:
        ns = argparse.Namespace(
            dataset=args.dataset,
            seed=seed,
            local_epochs=cfg.get("local_epochs", None),
            noise_multiplier=noise_multiplier,
            **{k: v for k, v in cfg.items() if k != "local_epochs"}
        )

        base_path = build_base_path(ns, model)
        data = load_eval(base_path)

        if data is None:
            continue

        rmse_trains.append(data["rmse_train"])
        rmse_vals.append(data["rmse_val"])
        rmse_tests.append(data["rmse_test"])

    if not rmse_trains:
        return None

    return {
        "rmse_train": (np.mean(rmse_trains), np.std(rmse_trains)),
        "rmse_val":   (np.mean(rmse_vals),   np.std(rmse_vals)),
        "rmse_test":  (np.mean(rmse_tests),  np.std(rmse_tests)),
    }


def print_row(label, results):
    if results is None:
        print(f"  [falta] {label}")
        return
    print(
        f"{label:<30} "
        f"{results['rmse_train'][0]:.4f} ± {results['rmse_train'][1]:.4f}   "
        f"{results['rmse_val'][0]:.4f} ± {results['rmse_val'][1]:.4f}   "
        f"{results['rmse_test'][0]:.4f} ± {results['rmse_test'][1]:.4f}"
    )


def main(args):
    configs = [json.loads(c) for c in args.configs]
    seeds = list(range(args.n_seeds))

    print(f"\n{'Configuracion':<30} {'RMSE train':>20} {'RMSE val':>20} {'RMSE test':>20}")
    print("-" * 95)

    for cfg in configs:
        label = f"k={cfg['k']} lr={cfg['lr']} reg={cfg['reg']} bs={cfg['batch_size']} rounds={cfg['rounds']}"
        if "local_epochs" in cfg:
            label += f" le={cfg['local_epochs']}"

        results = collect_over_seeds(args, args.model, cfg, seeds)
        print_row(label, results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--model", type=str, default="MF")
    ap.add_argument("--configs", type=str, nargs="+", required=True)
    ap.add_argument("--n_seeds", type=int, default=5)
    args = ap.parse_args()
    main(args)