import argparse
import torch

from src.utils.data import load_mia_in, load_mia_out
from src.utils.training import load_model
from src.utils.experiments import build_base_path
from src.utils.evaluation import save_mia_features


# -------------------------------------------------
# Feature computation
# -------------------------------------------------

def compute_features(P, Q, bu, bi, mu, data, label):
    rows = []

    for user_id, item_id, rating in data:
        pred = (
            mu
            + bu[user_id]
            + bi[item_id]
            + P[user_id].dot(Q[item_id])
        ).item()

        error = abs(pred - rating)
        norm_p = torch.norm(P[user_id]).item()
        norm_q = torch.norm(Q[item_id]).item()

        rows.append([pred, error, norm_p, norm_q, label])

    return rows


# -------------------------------------------------
# Main
# -------------------------------------------------

def main(args):
    base_path = build_base_path(args, args.model)

    # Load trained model
    P, Q, bu, bi, mu, _ = load_model(base_path)

    # ---------------------------
    # MIA IN (label=1)
    # ---------------------------
    mia_in_data = load_mia_in(args.dataset)
    features_in = compute_features(
        P, Q, bu, bi, mu,
        mia_in_data,
        label=1
    )
    save_mia_features(base_path, features_in, split="mia_in")

    # ---------------------------
    # MIA OUT (label=0)
    # ---------------------------
    mia_out_data = load_mia_out(args.dataset)
    features_out = compute_features(
        P, Q, bu, bi, mu,
        mia_out_data,
        label=0
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

    # DP (optional, needed for path resolution)
    ap.add_argument("--noise_multiplier", type=float)

    args = ap.parse_args()

    main(args)
