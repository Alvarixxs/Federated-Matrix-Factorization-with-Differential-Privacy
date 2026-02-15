import argparse
import numpy as np
import pandas as pd

from src.utils.experiments import build_base_path
from src.utils.evaluation import (
    load_mia_features,
    save_mia_dataset
)


def stratified_manual_split(df, test_frac, seed):
    """
    Manual stratified split without sklearn.
    Keeps label balance.
    """

    rng = np.random.default_rng(seed)

    train_parts = []
    test_parts = []

    for label in df["label"].unique():
        df_label = df[df["label"] == label]

        indices = np.arange(len(df_label))
        rng.shuffle(indices)

        split_idx = int(len(indices) * (1 - test_frac))

        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        train_parts.append(df_label.iloc[train_idx])
        test_parts.append(df_label.iloc[test_idx])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, test_df


def main(args):
    base_path = build_base_path(args, args.model)

    # Load features
    df_in = load_mia_features(base_path, "mia_in")
    df_out = load_mia_features(base_path, "mia_out")

    # Concatenate safely (no .append)
    df = pd.concat([df_in, df_out], ignore_index=True)

    # Manual stratified split
    train_df, test_df = stratified_manual_split(
        df,
        test_frac=args.test_frac,
        seed=args.seed
    )

    # Save everything
    save_mia_dataset(base_path, df, train_df, test_df)


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

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.3)

    args = ap.parse_args()

    main(args)
