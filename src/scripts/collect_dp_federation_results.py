import argparse
import json
import os
import numpy as np
from src.utils.experiments import build_base_path


def load_eval(base_path):
    eval_path = os.path.join(base_path, "eval.json")
    if not os.path.exists(eval_path):
        return None
    with open(eval_path) as f:
        return json.load(f)


def collect_over_seeds(args, seeds, n_clients, noise_multiplier):
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
            local_epochs=args.local_epochs,
            n_clients=n_clients,
            noise_multiplier=noise_multiplier,
        )
        base_path = build_base_path(ns, "FL_DP")
        data = load_eval(base_path)
        if data is None:
            continue
        rmse_tests.append(data["rmse_test"])
        if "epsilon" in data and data["epsilon"] is not None:
            epsilons.append(data["epsilon"])

    if not rmse_tests:
        return None

    return {
        "rmse_test": np.mean(rmse_tests),
        "epsilon":   np.mean(epsilons) if epsilons else None,
    }


def main(args):
    seeds = list(range(args.n_seeds))

    print(f"\n{'|C|':<8} {'sigma_abs':<12} {'sigma_mult':<12} {'RMSE test':>10} {'epsilon':>10}")
    print("-" * 58)

    q = args.sample_rate
    C = args.clip_norm

    for n_clients in args.n_clients_list:
        for sigma_abs in args.sigma_abs_list:
            noise_mult = round(sigma_abs * n_clients * q / (2 * C), 6)
            results = collect_over_seeds(args, seeds, n_clients, noise_mult)
            if results is None:
                print(f"{n_clients:<8} {sigma_abs:<12} {noise_mult:<12} {'[falta]':>10}")
                continue
            eps = f"{results['epsilon']:.4f}" if results['epsilon'] is not None else "-"
            print(f"{n_clients:<8} {sigma_abs:<12} {noise_mult:<12} {results['rmse_test']:>10.4f} {eps:>10}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--local_epochs", type=int, required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--reg", type=float, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--sample_rate", type=float, default=0.1)
    ap.add_argument("--clip_norm", type=float, default=1.0)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_clients_list", type=int, nargs="+", required=True)
    ap.add_argument("--sigma_abs_list", type=float, nargs="+", required=True)
    args = ap.parse_args()
    main(args)