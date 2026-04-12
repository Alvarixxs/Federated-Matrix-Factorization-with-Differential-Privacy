import torch
import argparse
import random
import types
from src.utils.seeding import set_seed
from src.utils.data import load_training_data, load_metadata
from src.models.FL_client import FLClient, ClientConfig
from src.models.FL_server import FLServer, ServerConfig
from src.models.FL_DP_server import FLDPServer, ServerConfig as DPServerConfig, DPConfig

def train_model(training_data_per_user, n_users, n_items, args, exclude_user):
    set_seed(args.seed)
    clients = [
        FLClient(user_id=u, cfg=ClientConfig(k=args.k, lr=args.lr, local_epochs=args.local_epochs, batch_size=args.batch_size, reg=args.reg), device=torch.device("cpu"))
        for u in range(n_users)
        if u != exclude_user
    ]
    server = FLServer(cfg=ServerConfig(k=args.k, sample_rate=args.sample_rate, rounds=args.rounds), device=torch.device("cpu"), n_items=n_items)
    data = {u: v for u, v in training_data_per_user.items() if u != exclude_user}
    all_data = [(u, i, r) for u, interactions in data.items() for (i, r) in interactions]
    server.train(clients, training_data_per_user)
    return server.Q, server.bi


def train_model_dp(training_data_per_user, n_users, n_items, args, noise_multiplier, exclude_user):
    set_seed(args.seed)
    clients = [
        FLClient(user_id=u, cfg=ClientConfig(k=args.k, lr=args.lr, local_epochs=args.local_epochs, batch_size=args.batch_size, reg=args.reg), device=torch.device("cpu"))
        for u in range(n_users)
        if u != exclude_user
    ]
    server = FLDPServer(cfg=DPServerConfig(k=args.k, sample_rate=args.sample_rate, rounds=args.rounds), dp_cfg=DPConfig(clip_norm=args.clip_norm, noise_multiplier=noise_multiplier), device=torch.device("cpu"), n_items=n_items)
    data = {u: v for u, v in training_data_per_user.items() if u != exclude_user}
    server.train(clients, training_data_per_user)
    return server.Q, server.bi


def main(args):
    training_data = load_training_data(args.dataset)
    metadata = load_metadata(args.dataset)
    n_users = metadata['n_users']
    n_items = metadata['n_items']


    training_data_per_user = {}
    for u, i, r in training_data:
        training_data_per_user.setdefault(u, []).append((i, r))

    # Buscar usuarios con más y menos interacciones
    user_counts = {u: len(v) for u, v in training_data_per_user.items()}
    user_a = max(user_counts, key=user_counts.get)  # más interacciones
    user_b = min(user_counts, key=user_counts.get)  # menos interacciones
    print(f"user_a={user_a} ({user_counts[user_a]} interacciones), user_b={user_b} ({user_counts[user_b]} interacciones)")
    import numpy as np

    n_experiments = 20
    results_fl = []
    results_dp = {sigma: [] for sigma in [0.5, 1.0, 2.0, 4.0]}

    for exp_seed in range(n_experiments):
        args.seed = exp_seed

        Q_a, bi_a = train_model(training_data_per_user, n_users, n_items, args, exclude_user=user_b)
        Q_b, bi_b = train_model(training_data_per_user, n_users, n_items, args, exclude_user=user_a)
        results_fl.append(torch.norm(Q_a - Q_b).item())

        for sigma in [0.5, 1.0, 2.0, 4.0]:
            Q_a, bi_a = train_model_dp(training_data_per_user, n_users, n_items, args, sigma, exclude_user=user_b)
            Q_b, bi_b = train_model_dp(training_data_per_user, n_users, n_items, args, sigma, exclude_user=user_a)
            results_dp[sigma].append(torch.norm(Q_a - Q_b).item())

    print(f"FL  — mean: {np.mean(results_fl):.4f}  std: {np.std(results_fl):.4f}")
    for sigma in [0.5, 1.0, 2.0, 4.0]:
        m = np.mean(results_dp[sigma])
        s = np.std(results_dp[sigma])
        print(f"FL_DP (σ={sigma}) — mean: {m:.4f}  std: {s:.4f}")

    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--local_epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--sample_rate", type=float, default=0.1)
    ap.add_argument("--clip_norm", type=float, default=1.0)
    args = ap.parse_args()
    main(args)