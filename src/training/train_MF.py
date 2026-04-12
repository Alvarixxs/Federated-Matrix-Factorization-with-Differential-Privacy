import argparse
import json
import os
import torch

from src.models.MF import MF, MFConfig
from src.utils.seeding import set_seed
from src.utils.data import load_training_data, load_val_data, load_metadata
from src.utils.training import save_model
from src.utils.experiments import build_base_path


def main(args):
    set_seed(args.seed)

    training_data = load_training_data(args.dataset)
    val_data = load_val_data(args.dataset)
    metadata = load_metadata(args.dataset)
    n_users = metadata['n_users']
    n_items = metadata['n_items']

    cfg = MFConfig(
        k=args.k,
        lr=args.lr,
        batch_size=args.batch_size,
        reg=args.reg,
        rounds=args.rounds
    )

    model = MF(
        n_users=n_users,
        n_items=n_items,
        cfg=cfg,
        device=torch.device("cpu"),
    )

    print(f"Entrenando MF en {args.dataset} (k={args.k}, lr={args.lr}, reg={args.reg}, rounds={args.rounds})")
    history_train, history_val = model.train(training_data, val_data)

    base_path = build_base_path(args, "MF")
    save_model(base_path, model.P, model.Q, model.bu, model.bi, model.mu, args)
    
    with open(os.path.join(base_path, "history.json"), "w") as f:
        json.dump({"rmse_train": history_train, "rmse_val": history_val}, f, indent=4)

    print(f"Entrenamiento completado. Modelo guardado en {base_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", type=str, required=True)

    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--rounds", type=int, default=30)

    args = ap.parse_args()

    main(args)