from __future__ import annotations
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ClientConfig:
    k: int
    lr: float
    local_epochs: int
    batch_size: int
    reg: float


class FLClient:
    def __init__(
        self,
        client_id: int,
        user_ids: List[int],
        cfg: ClientConfig,
        device: torch.device,
    ):
        self.client_id = client_id
        self.user_ids = user_ids
        self.cfg = cfg
        self.device = device

        # Parámetros locales por usuario
        self.p_u = {u: (0.01 * torch.randn(self.cfg.k, device=self.device)) for u in user_ids}
        self.b_u = {u: torch.tensor(0.0, device=self.device) for u in user_ids}

    def local_train(
        self,
        mu: float,
        Q_items: torch.Tensor,
        bi_items: torch.Tensor,
        training_data: Dict[int, List[Tuple[int, float]]]
    ):
        # training_data es un dict: user_id -> [(item, rating), ...]
        all_data = [(u, i, r) for u, interactions in training_data.items() for (i, r) in interactions]

        if len(all_data) == 0:
            return {}

        items = sorted(set(i for (_, i, _) in all_data))
        item_pos = {item_id: t for t, item_id in enumerate(items)}

        Q_c = Q_items[items].clone().detach().requires_grad_(True)
        bi_c = bi_items[items].clone().detach().requires_grad_(True)

        # Parámetros de usuario del cliente (stackeados)
        user_list = sorted(self.user_ids)
        user_pos = {u: idx for idx, u in enumerate(user_list)}
        P_c = torch.stack([self.p_u[u] for u in user_list]).clone().detach().requires_grad_(True)
        bu_c = torch.stack([self.b_u[u] for u in user_list]).clone().detach().requires_grad_(True)

        # Dataset
        uu = torch.tensor([user_pos[u] for (u, _, _) in all_data], device=self.device, dtype=torch.long)
        ii = torch.tensor([item_pos[i] for (_, i, _) in all_data], device=self.device, dtype=torch.long)
        rr = torch.tensor([r for (_, _, r) in all_data], device=self.device, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(uu, ii, rr),
            batch_size=min(self.cfg.batch_size, len(rr)),
            shuffle=True
        )

        opt = torch.optim.SGD([P_c, bu_c, Q_c, bi_c], lr=self.cfg.lr)
        mu_t = torch.tensor(mu, device=self.device, dtype=torch.float32)

        for _ in range(self.cfg.local_epochs):
            for (u_batch, i_batch, r_batch) in loader:
                p_batch = P_c[u_batch]
                bu_batch = bu_c[u_batch]
                q_batch = Q_c[i_batch]
                bi_batch = bi_c[i_batch]

                pred = mu_t + bu_batch + bi_batch + (q_batch * p_batch).sum(dim=1)
                err = pred - r_batch
                mse = (err ** 2).mean()

                l2 = (
                    P_c.pow(2).sum()
                    + Q_c.pow(2).sum()
                    + bu_c.pow(2).sum()
                    + bi_c.pow(2).sum()
                )

                loss = mse + self.cfg.reg * l2

                opt.zero_grad()
                loss.backward()
                opt.step()

        # Persistir parámetros de usuario
        for u in user_list:
            self.p_u[u] = P_c[user_pos[u]].detach()
            self.b_u[u] = bu_c[user_pos[u]].detach()

        with torch.no_grad():
            base_Q = Q_items[items]
            base_bi = bi_items[items]
            delta_Q = (Q_c.detach() - base_Q)
            delta_bi = (bi_c.detach() - base_bi)

        uploads: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        with torch.no_grad():
            for t, item_id in enumerate(items):
                uploads[item_id] = (delta_Q[t].detach().clone(), delta_bi[t].detach().clone())

        return uploads