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


def collect_over_seeds(args, model, seeds, noise_multiplier=None):
    rmse_trains = []
    rmse_vals = []
    rmse_tests = []
    epsilons = []

    for seed in seeds:
        ns = argparse.Namespace(
            dataset=args.dataset,
            seed=seed,
            k=args.k,
            lr=args.lr,
            reg=args.reg,
            batch_size=args.batch_size,
            rounds=args.rounds,
            local_epochs=args.local_epochs if model in ("FL", "FL_DP") else None,
            noise_multiplier=noise_multiplier
        )

        base_path = build_base_path(ns, model)
        data = load_eval(base_path)

        if data is None:
            continue

        rmse_trains.append(data["rmse_train"])
        rmse_vals.append(data["rmse_val"])
        rmse_tests.append(data["rmse_test"])
        if "epsilon" in data:
            epsilons.append(data["epsilon"])

    if not rmse_trains:
        return None

    return {
        "rmse_train": np.mean(rmse_trains),
        "rmse_val":   np.mean(rmse_vals),
        "rmse_test":  np.mean(rmse_tests),
        "epsilon":    np.mean(epsilons) if epsilons else None,
    }


def main(args):
    seeds = list(range(args.n_seeds))
    noise_multipliers = args.noise_multipliers

    print(f"\n{'Modelo':<25} {'RMSE train':>12} {'RMSE val':>10} {'RMSE test':>10} {'ε':>10}")
    print("-" * 72)

    for noise in noise_multipliers:
        results = collect_over_seeds(args, "FL_DP", seeds, noise_multiplier=noise)
        if results is None:
            print(f"  FL_DP (σ={noise}) [falta]")
            continue
        eps = f"{results['epsilon']:.4f}" if results['epsilon'] is not None else "-"
        print(f"FL_DP (σ={noise:<6}) {results['rmse_train']:>12.4f} {results['rmse_val']:>10.4f} {results['rmse_test']:>10.4f} {eps:>10}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--local_epochs", type=int, required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--reg", type=float, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--noise_multipliers", type=float, nargs="+", required=True)
    args = ap.parse_args()
    main(args)