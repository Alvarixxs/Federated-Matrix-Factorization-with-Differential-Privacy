import argparse
import torch

from src.utils.data import load_mia_in, load_mia_out
from src.utils.training import load_model
from src.utils.experiments import build_base_path
from src.utils.evaluation import save_mia_features


def compute_features(P, Q, bu, bi, mu, data, label, reg):
    rows = []

    for user_id, item_id, rating in data:
        p = P[user_id]
        q = Q[item_id]

        pred = (mu + bu[user_id] + bi[item_id] + p.dot(q)).item()

        error = abs(pred - rating)
        squared_error = (pred - rating) ** 2
        norm_p = torch.norm(p).item()
        norm_q = torch.norm(q).item()
        reg_loss = squared_error + reg * (norm_p**2 + norm_q**2)
        centered_score = (pred - mu - bu[user_id].item())

        rows.append([pred, error, squared_error, norm_p, norm_q, reg_loss, centered_score, label])

    return rows


def main(args):
    base_path = build_base_path(args, args.model)

    # Load trained model
    P, Q, bu, bi, mu, _ = load_model(base_path)

    mia_in_data = load_mia_in(args.dataset)
    features_in = compute_features(
        P, Q, bu, bi, mu,
        mia_in_data,
        label=1,
        reg=args.reg
    )
    save_mia_features(base_path, features_in, split="mia_in")

    mia_out_data = load_mia_out(args.dataset)
    features_out = compute_features(
        P, Q, bu, bi, mu,
        mia_out_data,
        label=0,
        reg=args.reg
    )
    save_mia_features(base_path, features_out, split="mia_out")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)

    # base architecture
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--local_epochs", type=int)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--reg", type=float, required=True)
    ap.add_argument("--rounds", type=int, required=True)

    ap.add_argument("--noise_multiplier", type=float)

    args = ap.parse_args()

    main(args)
