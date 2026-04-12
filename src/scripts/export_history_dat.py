import argparse
import json
import os

from src.utils.experiments import build_base_path


def export_dat(history, filepath, step=5):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("epoch rmse\n")
        for epoch, rmse in enumerate(history, start=1):
            if epoch % step == 0 or epoch == 1:
                f.write(f"{epoch} {rmse:.6f}\n")


def main(args):
    if args.local_epochs is not None:
        model = "FL"
    else:
        model = "MF"

    base_path = build_base_path(args, model)
    history_path = os.path.join(base_path, "history.json")

    with open(history_path, "r") as f:
        data = json.load(f)

    history_train = data["rmse_train"]
    history_val = data["rmse_val"]

    out_dir = os.path.join("latex", "data")

    export_dat(history_train, os.path.join(out_dir, f"convergence_{model}_{args.dataset}_train.dat"))
    export_dat(history_val, os.path.join(out_dir, f"convergence_{model}_{args.dataset}_val.dat"))

    print(f"Datos exportados a {out_dir}")


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