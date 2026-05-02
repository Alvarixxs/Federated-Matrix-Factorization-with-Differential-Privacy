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


def export_dat(rows, filepath, header):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(" ".join(str(v) for v in row) + "\n")


def main(args):
    seeds = list(range(args.n_seeds))
    out_dir = os.path.join("latex", "data")
    q = args.sample_rate
    C = args.clip_norm

    for n_clients in args.n_clients_list:
        rows_sigma = []
        rows_eps = []

        for sigma_abs in args.sigma_abs_list:
            noise_mult = round(sigma_abs * n_clients * q / (2 * C), 6)
            results = collect_over_seeds(args, seeds, n_clients, noise_mult)
            if results is None:
                continue
            rows_sigma.append([round(sigma_abs * 100, 4), results["rmse_test"]])
            EPS_THRESHOLD = 50.0

            # En rows_eps:
            if results["epsilon"] is not None and results["epsilon"] <= EPS_THRESHOLD:
                rows_eps.append([results["epsilon"], results["rmse_test"]])

        export_dat(
            rows_sigma,
            os.path.join(out_dir, f"dp_sigma_{args.dataset}_nc{n_clients}.dat"),
            "sigma_abs rmse_test"
        )
        export_dat(
            rows_eps,
            os.path.join(out_dir, f"dp_eps_{args.dataset}_nc{n_clients}.dat"),
            "epsilon rmse_test"
        )

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
    ap.add_argument("--sample_rate", type=float, default=0.1)
    ap.add_argument("--clip_norm", type=float, default=1.0)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_clients_list", type=int, nargs="+", required=True)
    ap.add_argument("--sigma_abs_list", type=float, nargs="+", required=True)
    args = ap.parse_args()
    main(args)