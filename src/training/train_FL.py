import argparse
import json
import os
import random
import torch

from src.models.FL_client import ClientConfig, FLClient
from src.models.FL_server import FLServer, ServerConfig
from src.utils.seeding import set_seed
from src.utils.data import load_training_data, load_val_data, load_metadata
from src.utils.training import reconstruct_user_factors, save_model
from src.utils.experiments import build_base_path


def assign_users_to_clients(n_users: int, n_clients: int, seed: int):
    """Asigna n_users usuarios a n_clients clientes de forma aleatoria."""
    random.seed(seed)
    user_ids = list(range(n_users))
    random.shuffle(user_ids)

    # Distribuir lo mas uniformemente posible
    clients_users = [[] for _ in range(n_clients)]
    for i, u in enumerate(user_ids):
        clients_users[i % n_clients].append(u)

    return clients_users


def main(args):
    set_seed(args.seed)

    training_data = load_training_data(args.dataset)
    val_data = load_val_data(args.dataset)
    metadata = load_metadata(args.dataset)
    n_users = metadata['n_users']
    n_items = metadata['n_items']

    n_clients = args.n_clients if args.n_clients is not None else n_users

    # Asignar usuarios a clientes
    clients_users = assign_users_to_clients(n_users, n_clients, args.seed)

    clients = [
        FLClient(
            client_id=cid,
            user_ids=user_list,
            cfg=ClientConfig(
                k=args.k,
                local_epochs=args.local_epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                reg=args.reg,
            ),
            device=torch.device("cpu"),
        ) for cid, user_list in enumerate(clients_users)
    ]

    server = FLServer(
        cfg=ServerConfig(
            k=args.k,
            sample_rate=args.sample_rate,
            rounds=args.rounds,
        ),
        device=torch.device("cpu"),
        n_items=n_items
    )

    # Agrupar training data por user_id primero
    training_data_per_user = {}
    for u, i, r in training_data:
        training_data_per_user.setdefault(u, []).append((i, r))

    # Agrupar training data por cliente (dict: user_id -> [(item, rating)])
    training_data_per_client = {}
    for cid, user_list in enumerate(clients_users):
        training_data_per_client[cid] = {u: training_data_per_user.get(u, []) for u in user_list}

    print(f"Entrenando FL en {args.dataset} (k={args.k}, lr={args.lr}, reg={args.reg}, rounds={args.rounds}, local_epochs={args.local_epochs}, n_clients={n_clients})")
    history_train, history_val = server.train(clients, training_data_per_client, val_data=val_data)

    P, bu = reconstruct_user_factors(clients, args.k, torch.device("cpu"), n_users)
    base_path = build_base_path(args, "FL")
    save_model(base_path, P, server.Q, bu, server.bi, server.mu, args)

    with open(os.path.join(base_path, "history.json"), "w") as f:
        json.dump({"rmse_train": history_train, "rmse_val": history_val}, f, indent=4)

    print(f"Entrenamiento completado. Modelo guardado en {base_path}")


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
    ap.add_argument("--n_clients", type=int, default=None, help="Número de clientes. Si no se especifica, cada usuario es un cliente.")

    args = ap.parse_args()

    main(args)