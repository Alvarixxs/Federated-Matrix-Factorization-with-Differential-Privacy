import os
import json
import torch
from typing import Tuple, List


def save_model(save_path, P, Q, bu, bi, mu, args, accountant=None):
    os.makedirs(save_path, exist_ok=True)

    torch.save({
        "P": P.cpu(),
        "Q": Q.cpu(),
        "bu": bu.cpu(),
        "bi": bi.cpu(),
        "mu": mu,
    }, os.path.join(save_path, "weights.pt"))

    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump({
            "k": args.k,
            "lr": args.lr,
            "local_epochs": args.local_epochs if hasattr(args, "local_epochs") else None,
            "batch_size": args.batch_size,
            "reg": args.reg,
            "rounds": args.rounds
        }, f, indent=4)

    if accountant is not None:
        with open(os.path.join(save_path, "privacy.json"), "w") as f:
            json.dump({"clip_norm": args.clip_norm, "noise_multiplier": args.noise_multiplier}, f, indent=4)

    if accountant is not None:
        torch.save(accountant, os.path.join(save_path, "accountant.pt"))


def load_model(load_path):
    weights = torch.load(os.path.join(load_path, "weights.pt"))

    P = weights["P"]
    Q = weights["Q"]
    bu = weights["bu"]
    bi = weights["bi"]
    mu = weights["mu"]

    accountant = None
    accountant_path = os.path.join(load_path, "accountant.pt")
    if os.path.exists(accountant_path):
        accountant = torch.load(accountant_path, weights_only=False)

    return P, Q, bu, bi, mu, accountant


def reconstruct_user_factors(
    clients: List,
    k: int,
    device: torch.device,
    n_users: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    P = torch.zeros(n_users, k, device=device)
    bu = torch.zeros(n_users, device=device)

    for client in clients:
        for u in client.user_ids:
            P[u] = client.p_u[u].detach()
            bu[u] = client.b_u[u].detach()

    return P, bu