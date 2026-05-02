import os
import json
import argparse
import pandas as pd

from src.utils.seeding import set_seed


def load_dataset(dataset_name: str):

    if dataset_name == "ml-100k":
        path = os.path.join("data", "ml-100k", "raw", "u.data")
        df = pd.read_csv(
            path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"]
        )

    elif dataset_name == "ml-latest-small":
        path = os.path.join("data", "ml-latest-small", "raw", "ratings.csv")
        df = pd.read_csv(path)
        df = df[["userId", "movieId", "rating"]].rename(columns={"userId": "user_id", "movieId": "item_id"})


    elif dataset_name == "ml-1m":
        path = os.path.join("data", "ml-1m", "raw", "ratings.dat")
        df = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"]
        )

    else:
        raise ValueError("Dataset no soportado. Usa 'ml-100k' o 'ml-1m'.")

    user_ids = df["user_id"].unique()
    item_ids = df["item_id"].unique()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {m: i for i, m in enumerate(item_ids)}

    df["user_id"] = df["user_id"].map(user2idx)
    df["item_id"] = df["item_id"].map(item2idx)

    return df


def create_mia_sets(train_df, attack_df, seed):

    set_seed(seed)

    mia_size = len(attack_df)

    mia_in = train_df.sample(
        n=mia_size,
        random_state=seed
    )

    mia_out = attack_df

    return mia_in.reset_index(drop=True), mia_out.reset_index(drop=True)


def split_by_user(df, train_ratio, val_ratio, test_ratio, seed):
    train_list = []
    val_list = []
    test_list = []

    for user_id, user_data in df.groupby("user_id"):
        user_data = user_data.sample(frac=1, random_state=seed)

        n = len(user_data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_user = user_data.iloc[:n_train]
        val_user = user_data.iloc[n_train:n_train + n_val]
        test_user = user_data.iloc[n_train + n_val:]

        train_list.append(train_user)
        val_list.append(val_user)
        test_list.append(test_user)

    return (
        pd.concat(train_list).reset_index(drop=True),
        pd.concat(val_list).reset_index(drop=True),
        pd.concat(test_list).reset_index(drop=True),
    )


def main(args):
    set_seed(args.seed)

    print(f"Cargando dataset {args.dataset}...")
    df = load_dataset(args.dataset)

    print("Creando splits por usuario...")
    train_df, val_df, test_df = split_by_user(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    print("Creando conjuntos MIA...")
    mia_in, mia_out = create_mia_sets(train_df, test_df, seed=args.seed)

    output_dir = os.path.join("data", args.dataset, "splits")
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, "train_model.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_model.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_model.csv"), index=False)
    mia_in.to_csv(os.path.join(output_dir, "mia_in.csv"), index=False)
    mia_out.to_csv(os.path.join(output_dir, "mia_out.csv"), index=False)

    metadata = {
        "dataset": args.dataset,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "total_interactions": len(df),
        "n_users": df['user_id'].nunique(),
        "n_items": df['item_id'].nunique(),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "avg_interactions_per_user": round(len(df) / df['user_id'].nunique(), 2),
        "avg_train_interactions_per_user": round(len(train_df) / df['user_id'].nunique(), 2),
    }

    os.makedirs(os.path.join("data", args.dataset, "metadata"), exist_ok=True)
    with open(os.path.join("data", args.dataset, "metadata", "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("Splits creados correctamente.")
    print(json.dumps(metadata, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, required=True)
    parser.add_argument("--val_ratio", type=float, required=True)
    parser.add_argument("--test_ratio", type=float, required=True)
    args = parser.parse_args()
    main(args)