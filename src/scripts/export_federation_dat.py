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

        rmse_tests.append(data["rmse_test"])

    if not rmse_tests:
        return None

    return np.mean(rmse_tests)


def export_dat(rows, filepath, header):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(" ".join(str(v) for v in row) + "\n")


def main(args):
    seeds = list(range(args.n_seeds))
    out_dir = os.path.join("latex", "data")

    rows = []
    for n_clients in args.n_clients_list:
        rmse = collect_over_seeds(args, seeds, n_clients)
        if rmse is None:
            continue
        rows.append([n_clients, rmse])

    export_dat(rows, os.path.join(out_dir, f"federation_{args.dataset}.dat"), "n_clients rmse_test")

    print(f"Datos exportados a {out_dir}")


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