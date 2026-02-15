import argparse

from src.utils.data import load_test_data, load_training_data
from src.utils.training import load_model
from src.utils.evaluation import save_evaluation
from src.utils.experiments import build_base_path


def compute_rmse(P, Q, bu, bi, mu, test_data):
    squared_error_sum = 0.0
    count = 0

    for user_id, item_id, rating in test_data:
        pred = (mu + bu[user_id] + bi[item_id] + P[user_id].dot(Q[item_id])).item()
        squared_error_sum += (pred - rating) ** 2
        count += 1

    rmse = (squared_error_sum / count) ** 0.5 if count > 0 else float('inf')
    return rmse


def main(args):
    base_path = build_base_path(args, args.model)
    P, Q, bu, bi, mu, accountant = load_model(base_path)

    training_data = load_training_data(args.dataset)
    test_data = load_test_data(args.dataset)

    rmse_train = compute_rmse(P, Q, bu, bi, mu, training_data)
    rmse_test = compute_rmse(P, Q, bu, bi, mu, test_data)

    eps = None
    if accountant is not None:
        eps, _ = accountant.get_privacy_spent(delta=args.delta)

    save_evaluation(base_path, rmse_train, rmse_test, eps)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)

    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--local_epochs", type=int)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--reg", type=float, required=True)
    ap.add_argument("--rounds", type=int, required=True)

    ap.add_argument("--noise_multiplier", type=float)

    ap.add_argument("--delta", type=float)

    args = ap.parse_args()

    main(args)
