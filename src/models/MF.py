import torch
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass

from src.utils.data import load_test_data

@dataclass
class MFConfig:
    k: int
    lr: float
    reg: float
    batch_size: int
    rounds: int


class MF:
    def __init__(
        self,
        cfg: MFConfig,
        device: torch.device,
        n_users: int,
        n_items: int,
    ):
        self.cfg = cfg
        self.device = device

        # Parameters
        self.P = 0.01 * torch.randn(n_users, cfg.k, device=device)
        self.Q = 0.01 * torch.randn(n_items, cfg.k, device=device)
        self.bu = torch.zeros(n_users, device=device)
        self.bi = torch.zeros(n_items, device=device)

        # Enable gradients
        self.P.requires_grad_(True)
        self.Q.requires_grad_(True)
        self.bu.requires_grad_(True)
        self.bi.requires_grad_(True)

    def train(self, training_data):
        self.mu = sum(r for (_, _, r) in training_data) / len(training_data)
        users = torch.tensor([u for (u, _, _) in training_data], dtype=torch.long)
        items = torch.tensor([i for (_, i, _) in training_data], dtype=torch.long)
        ratings = torch.tensor([r for (_, _, r) in training_data], dtype=torch.float32)
        dataset = TensorDataset(users, items, ratings)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        opt = torch.optim.SGD([self.P, self.Q, self.bu, self.bi], lr=self.cfg.lr)
        mu_t = torch.tensor(self.mu, device=self.device)

        for _ in range(1, self.cfg.rounds + 1):
            for u, i, r in loader:
                u = u.to(self.device)
                i = i.to(self.device)
                r = r.to(self.device)

                pred = (
                    mu_t
                    + self.bu[u]
                    + self.bi[i]
                    + (self.P[u] * self.Q[i]).sum(dim=1)
                )

                err = pred - r
                mse = (err ** 2).mean()

                l2 = (
                    self.P[u].pow(2).sum()
                    + self.Q[i].pow(2).sum()
                    + self.bu[u].pow(2).sum()
                    + self.bi[i].pow(2).sum()
                )

                loss = mse + self.cfg.reg * l2

                opt.zero_grad()
                loss.backward()
                opt.step()