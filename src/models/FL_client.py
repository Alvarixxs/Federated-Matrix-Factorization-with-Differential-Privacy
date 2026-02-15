from __future__ import annotations
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from dataclasses import dataclass


@dataclass
class ClientConfig:
    k: int
    lr: float
    local_epochs: int
    batch_size: int
    reg: float  # L2 regularization strength
    

class FLClient:
    def __init__(
        self,
        user_id: int,
        cfg: ClientConfig,
        device: torch.device,
    ) :
        self.user_id = user_id
        self.cfg = cfg
        self.device = device

        # Local user parameters (persist)
        self.p_u = (0.01 * torch.randn(self.cfg.k, device=self.device))
        self.b_u = torch.tensor(0.0, device=self.device)

    def local_train(
        self,
        mu: float,
        Q_items: torch.Tensor,
        bi_items: torch.Tensor,
        training_data: List[Tuple[int, float]]
    ):
        if len(training_data) == 0:
            return {}

        # Items this client touched
        items = sorted(set(i for (i, _) in training_data))
        item_pos = {item_id: t for t, item_id in enumerate(items)}

        # Local copies of touched item params
        Q_u = Q_items[items].clone().detach().requires_grad_(True)     # [m, k]
        bi_u = bi_items[items].clone().detach().requires_grad_(True)   # [m]

        # Local user params (learnable this round, then persisted)
        p_u = self.p_u.clone().detach().requires_grad_(True)
        b_u = self.b_u.clone().detach().requires_grad_(True)

        # Build local dataset
        ii = torch.tensor([item_pos[i] for (i, _) in training_data],
                          device=self.device, dtype=torch.long)
        rr = torch.tensor([r for (_, r) in training_data],
                          device=self.device, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(ii, rr),
            batch_size=min(self.cfg.batch_size, len(rr)),
            shuffle=True
        )

        opt = torch.optim.SGD([p_u, b_u, Q_u, bi_u], lr=self.cfg.lr)

        mu_t = torch.tensor(mu, device=self.device, dtype=torch.float32)

        for _ in range(self.cfg.local_epochs):
            for (i_batch, r_batch) in loader:
                q_batch = Q_u[i_batch]        # [B, k]
                bi_batch = bi_u[i_batch]      # [B]

                pred = mu_t + b_u + bi_batch + (q_batch @ p_u)  # [B]
                err = pred - r_batch
                mse = (err ** 2).mean()

                # L2 regularization
                l2 = (
                    p_u.pow(2).sum()
                    + Q_u.pow(2).sum()
                    + b_u.pow(2)
                    + bi_u.pow(2).sum()
                )

                loss = mse + self.cfg.reg * l2

                opt.zero_grad()
                loss.backward()
                opt.step()

        # Persist updated user params locally
        self.p_u = p_u.detach()
        self.b_u = b_u.detach()

        with torch.no_grad():
            base_Q = Q_items[items]
            base_bi = bi_items[items]
            delta_Q = (Q_u.detach() - base_Q)          # [m, k]
            delta_bi = (bi_u.detach() - base_bi)       # [m]

        # Upload updated item params for touched items
        uploads: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        with torch.no_grad():
            for t, item_id in enumerate(items):
                uploads[item_id] = (delta_Q[t].detach().clone(), delta_bi[t].detach().clone())

        return uploads