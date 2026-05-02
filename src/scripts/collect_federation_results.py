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


def collect_over_seeds(args, seeds, n_clients):
    rmse_trains = []
    rmse_vals = []
    rmse_tests = []

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
            noise_multiplier=None,
        )

        base_path = build_base_path(ns, "FL")
        data = load_eval(base_path)

        if data is None:
            continue

        rmse_trains.append(data["rmse_train"])
        rmse_vals.append(data["rmse_val"])
        rmse_tests.append(data["rmse_test"])

    if not rmse_trains:
        return None

    return {
        "rmse_train": np.mean(rmse_trains),
        "rmse_val": np.mean(rmse_vals),
        "rmse_test": np.mean(rmse_tests),
    }


def main(args):
    seeds = list(range(args.n_seeds))

    print(f"\n{'n_clients':<12} {'RMSE train':>12} {'RMSE val':>10} {'RMSE test':>10}")
    print("-" * 48)

    for n_clients in args.n_clients_list:
        results = collect_over_seeds(args, seeds, n_clients)
        if results is None:
            print(f"  n_clients={n_clients} [falta]")
            continue
        print(f"{n_clients:<12} {results['rmse_train']:>12.4f} {results['rmse_val']:>10.4f} {results['rmse_test']:>10.4f}")


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
    ap.add_argument("--n_clients_list", type=int, nargs="+", required=True)
    args = ap.parse_args()
    main(args)