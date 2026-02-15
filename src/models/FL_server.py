from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Dict, List, Tuple

import torch


@dataclass
class ServerConfig:
    k: int
    sample_rate: float
    rounds: int


class FLServer:
    def __init__(self, cfg: ServerConfig, device: torch.device, n_items: int):
        self.cfg = cfg
        self.device = device

        self.Q = (0.01 * torch.randn(n_items, self.cfg.k, device=self.device))
        self.bi = torch.zeros(n_items, device=self.device)

    def _sample_clients(self, clients):
        selected = [c for c in clients if random.random() < self.cfg.sample_rate]
        return selected

    def aggregate_item_updates(
        self,
        updates_list: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    ):
        m = len(updates_list)
        if m == 0:
            return

        w = torch.cat([
            self.Q.reshape(-1),
            self.bi.reshape(-1)
        ])

        w_sum = torch.zeros(w.shape, device=self.device)

        for upd in updates_list:
            dQ_full = torch.zeros_like(self.Q)
            db_full = torch.zeros_like(self.bi)

            for item_id, (dQ_i, db_i) in upd.items():
                dQ_full[item_id] = dQ_i
                db_full[item_id] = db_i

            w_client = torch.cat([
                dQ_full.reshape(-1),
                db_full.reshape(-1)
            ])

            w_sum += w_client

        w_update = w_sum / m

        with torch.no_grad():
            w = w + w_update

            new_Q_flat = w[:self.Q.numel()]
            new_b_flat = w[self.Q.numel():]

            self.Q = new_Q_flat.reshape_as(self.Q)
            self.bi = new_b_flat.reshape_as(self.bi)
        
    def train(self, clients, training_data_per_user, training_data):
        self.mu = sum(r for (_, r) in sum(training_data_per_user.values(), [])) / max(sum(len(v) for v in training_data_per_user.values()), 1)

        for _ in range(1,  self.cfg.rounds + 1):
            selected = self._sample_clients(clients)
            updates_list = []

            for client in selected:
                uploads = client.local_train(
                    mu=self.mu,
                    Q_items=self.Q,
                    bi_items=self.bi,
                    training_data=training_data_per_user[client.user_id]
                )
                updates_list.append(uploads)

            self.aggregate_item_updates(updates_list)