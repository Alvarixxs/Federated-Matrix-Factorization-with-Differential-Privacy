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


def collect_over_seeds(args, model, seeds, noise_multiplier=None):
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

        rmse_tests.append(data["rmse_test"])
        if "epsilon" in data and data["epsilon"] is not None:
            epsilons.append(data["epsilon"])

    if not rmse_tests:
        return None

    return {
        "rmse_test": np.mean(rmse_tests),
        "epsilon":   np.mean(epsilons) if epsilons else None,
    }


def export_dat(rows, filepath, header):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(" ".join(str(v) for v in row) + "\n")


def main(args):
    seeds = list(range(args.n_seeds))
    out_dir = os.path.join("latex", "data")

    # FL como referencia
    fl_results = collect_over_seeds(args, "FL", seeds)
    fl_rmse = fl_results["rmse_test"] if fl_results else None

    # FL_DP por sigma
    rows_sigma = []
    rows_eps = []

    for noise in args.noise_multipliers:
        results = collect_over_seeds(args, "FL_DP", seeds, noise_multiplier=noise)
        if results is None:
            continue
        rows_sigma.append([noise, results["rmse_test"]])
        if results["epsilon"] is not None:
            rows_eps.append([results["epsilon"], results["rmse_test"]])

    # Exportar RMSE vs sigma
    export_dat(rows_sigma, os.path.join(out_dir, f"dp_rmse_sigma_{args.dataset}.dat"), "sigma rmse_test")

    # Exportar RMSE vs epsilon
    export_dat(rows_eps, os.path.join(out_dir, f"dp_rmse_eps_{args.dataset}.dat"), "epsilon rmse_test")

    # Exportar FL como referencia (linea horizontal)
    if fl_rmse is not None:
        fl_rows = [[noise, fl_rmse] for noise in args.noise_multipliers]
        export_dat(fl_rows, os.path.join(out_dir, f"fl_rmse_sigma_{args.dataset}.dat"), "sigma rmse_test")

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
    ap.add_argument("--noise_multipliers", type=float, nargs="+", required=True)
    args = ap.parse_args()
    main(args)