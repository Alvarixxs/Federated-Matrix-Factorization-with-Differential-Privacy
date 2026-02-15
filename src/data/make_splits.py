import os
import json
import argparse
import pandas as pd

from src.utils.seeding import set_seed


def split_by_user(df, train_ratio, test_ratio, seed):
    """
    Split interno por usuario.
    """

    train_list = []
    test_list = []
    attack_list = []

    for user_id, user_data in df.groupby("user_id"):

        user_data = user_data.sample(frac=1, random_state=seed)

        n = len(user_data)
        n_train = int(n * train_ratio)
        n_test = int(n * test_ratio)

        train_user = user_data.iloc[:n_train]
        test_user = user_data.iloc[n_train:n_train + n_test]
        attack_user = user_data.iloc[n_train + n_test:]

        train_list.append(train_user)
        test_list.append(test_user)
        attack_list.append(attack_user)

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    attack_df = pd.concat(attack_list).reset_index(drop=True)

    return train_df, test_df, attack_df


def load_dataset(dataset_name: str):
    """
    Carga ML-100K o ML-1M y devuelve DataFrame estándar:
    user_id, item_id, rating
    """

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


def main(args):

    set_seed(args.seed)

    print(f"\n📥 Cargando dataset {args.dataset}...")
    df = load_dataset(args.dataset)

    print("🔀 Creando splits por usuario...")
    train_df, test_df, attack_df = split_by_user(
        df,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    print("🎯 Creando conjuntos MIA...")
    mia_in, mia_out = create_mia_sets(
        train_df,
        attack_df,
        seed=args.seed
    )

    # Crear carpeta de salida específica por dataset
    output_dir = os.path.join("data", args.dataset, "splits")
    os.makedirs(output_dir, exist_ok=True)

    # Guardar CSVs
    train_df.to_csv(os.path.join(output_dir, "train_model.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_model.csv"), index=False)
    attack_df.to_csv(os.path.join(output_dir, "attack_holdout.csv"), index=False)
    mia_in.to_csv(os.path.join(output_dir, "mia_in.csv"), index=False)
    mia_out.to_csv(os.path.join(output_dir, "mia_out.csv"), index=False)

    # Metadata
    metadata = {
        "dataset": args.dataset,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "test_ratio": args.test_ratio,
        "attack_ratio": 0.0 if 1.0 - args.train_ratio - args.test_ratio < 1e-4 else round(1.0 - args.train_ratio - args.test_ratio, 2),
        "total_interactions": len(df),
        "n_users": df['user_id'].nunique(),
        "n_items": df['item_id'].nunique(),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "attack_size": len(attack_df)
    }

    os.makedirs(os.path.join("data", args.dataset, "metadata"), exist_ok=True)
    with open(os.path.join("data", args.dataset, "metadata", "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n✅ Splits creados correctamente")
    print(json.dumps(metadata, indent=4))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument("--dataset", type=str, required=True, help="ml-latest-small, ml-100k o ml-1m")
    parser.add_argument("--train_ratio", type=float, required=True, help="Proporción de interacciones para entrenamiento")
    parser.add_argument("--test_ratio", type=float, required=True, help="Proporción de interacciones para prueba")

    args = parser.parse_args()
    main(args)
