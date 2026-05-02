from __future__ import annotations
from dataclasses import dataclass
import json
import os
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

    def _compute_rmse(self, clients, mu, data):
        with torch.no_grad():
            squared_errors = []
            # Mapear user_id -> (p_u, b_u) desde los clientes
            user_map = {}
            for c in clients:
                for u in c.user_ids:
                    user_map[u] = (c.p_u[u], c.b_u[u])

            for user_id, item_id, rating in data:
                if user_id not in user_map:
                    continue
                p_u, b_u = user_map[user_id]
                pred = (
                    mu
                    + b_u
                    + self.bi[item_id]
                    + p_u.dot(self.Q[item_id])
                ).item()
                squared_errors.append((pred - rating) ** 2)

        return (sum(squared_errors) / len(squared_errors)) ** 0.5 if squared_errors else float('inf')

    def train(self, clients, training_data_per_client, val_data=None):
        # Calcular mu sobre todas las interacciones de todos los clientes
        all_interactions = []
        for cid, user_data in training_data_per_client.items():
            for u, interactions in user_data.items():
                all_interactions.extend(interactions)

        self.mu = sum(r for (_, r) in all_interactions) / max(len(all_interactions), 1)

        # Construir lista plana de datos de entrenamiento para RMSE
        train_data = []
        for cid, user_data in training_data_per_client.items():
            for u, interactions in user_data.items():
                for (i, r) in interactions:
                    train_data.append((u, i, r))

        history_train = []
        history_val = []

        for t in range(1, self.cfg.rounds + 1):
            selected = self._sample_clients(clients)
            updates_list = []

            for client in selected:
                uploads = client.local_train(
                    mu=self.mu,
                    Q_items=self.Q,
                    bi_items=self.bi,
                    training_data=training_data_per_client[client.client_id]
                )
                updates_list.append(uploads)

            self.aggregate_item_updates(updates_list)

            if val_data is not None:
                rmse_train = self._compute_rmse(clients, self.mu, train_data)
                rmse_val = self._compute_rmse(clients, self.mu, val_data)
                history_train.append(rmse_train)
                history_val.append(rmse_val)

        return history_train, history_val