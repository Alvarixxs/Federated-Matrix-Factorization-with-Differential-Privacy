import argparse
import os
import torch

from src.models.FL_client import ClientConfig, FLClient
from src.models.FL_server import FLServer, ServerConfig
from src.utils.seeding import set_seed
from src.utils.data import load_training_data, load_metadata
from src.utils.training import reconstruct_user_factors, save_model
from src.utils.experiments import build_base_path


def main(args):
    set_seed(args.seed)

    training_data = load_training_data(args.dataset)
    metadata = load_metadata(args.dataset)
    n_users = metadata['n_users']
    n_items = metadata['n_items']

    clients = [
        FLClient(
            user_id=user_id,
            cfg=ClientConfig(
                k=args.k,
                local_epochs=args.local_epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                reg=args.reg,
            ),
            device=torch.device("cpu"),
    ) for user_id in range(n_users)]

    server = FLServer(
        cfg=ServerConfig(
            k=args.k,
            sample_rate=args.sample_rate,
            rounds=args.rounds,
        ),
        device=torch.device("cpu"),
        n_items=n_items
    )

    training_data_per_user = {}

    for u, i, r in training_data:
        training_data_per_user.setdefault(u, []).append((i, r))

    server.train(clients, training_data_per_user, training_data)
    P, bu = reconstruct_user_factors(clients, args.k, torch.device("cpu"))
    base_path = build_base_path(args, "FL")
    save_model(base_path, P, server.Q, bu, server.bi, server.mu, args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", type=str, required=True)

    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--local_epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--rounds", type=int, default=30)

    ap.add_argument("--sample_rate", type=float, default=0.1)

    args = ap.parse_args()

    main(args)
