from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Dict, List, Tuple
import torch
from opacus.accountants import RDPAccountant


@dataclass
class ServerConfig:
    k: int
    sample_rate: float
    rounds: int

@dataclass
class DPConfig:
    clip_norm: float # L2 clipping bound (user-level = one client)
    noise_multiplier: float # Gaussian noise multiplier (std = noise_multiplier * clip_norm)


class FLDPServer:
    def __init__(self, cfg: ServerConfig, device: torch.device, n_items: int, dp_cfg: DPConfig) -> None:
        self.cfg = cfg
        self.device = device

        self.Q = (0.01 * torch.randn(n_items, self.cfg.k, device=self.device))
        self.bi = torch.zeros(n_items, device=self.device)

        self.dp_cfg = dp_cfg
        self.accountant = RDPAccountant()

    def _sample_clients(self, clients):
        selected = [c for c in clients if random.random() < self.cfg.sample_rate]
        return selected

    def aggregate_item_updates(
        self,
        updates_list: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        round_seed
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

            l2 = torch.linalg.vector_norm(w_client, ord=2)
            scale = min(1.0, self.dp_cfg.clip_norm / (l2 + 1e-12))
            w_client = w_client * scale

            w_sum += w_client

        w_avg = w_sum / m


        sensitivity = self.dp_cfg.clip_norm / m
        noise_std = self.dp_cfg.noise_multiplier * sensitivity

        w_avg_clean = w_avg.clone()

        generator = torch.Generator()
        generator.manual_seed(round_seed)
        noise = torch.randn(w_avg.shape, generator=generator) * noise_std
        w_update = w_avg + noise

        print(f"||noise||: {torch.norm(noise).item():.4f}")
        print(f"||w_avg||: {torch.norm(w_avg_clean).item():.4f}")
        print(f"ratio noise/signal: {torch.norm(noise).item() / torch.norm(w_avg_clean).item():.4f}")

        with torch.no_grad():
            w = w + w_update

            new_Q_flat = w[:self.Q.numel()]
            new_b_flat = w[self.Q.numel():]

            self.Q = new_Q_flat.reshape_as(self.Q)
            self.bi = new_b_flat.reshape_as(self.bi)

        self.accountant.step(
            noise_multiplier=self.dp_cfg.noise_multiplier,
            sample_rate=self.cfg.sample_rate,
        )


    def train(self, clients, training_data_per_user):
        self.mu = sum(r for (_, r) in sum(training_data_per_user.values(), [])) / max(sum(len(v) for v in training_data_per_user.values()), 1)

        for t in range(1,  self.cfg.rounds + 1):
            random.seed(42 + t)
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

            self.aggregate_item_updates(updates_list, round_seed=42+t)