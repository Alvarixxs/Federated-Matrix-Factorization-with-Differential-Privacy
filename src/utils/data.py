import json
import pandas as pd
import os


def load_training_data(dataset):
    csv_path = os.path.join("data", dataset, "splits", "train_model.csv")
    df = pd.read_csv(csv_path)
    return list(df[['user_id','item_id','rating']].itertuples(index=False, name=None))


def load_test_data(dataset):
    csv_path = os.path.join("data", dataset, "splits", "test_model.csv")
    df = pd.read_csv(csv_path)
    return list(df[['user_id','item_id','rating']].itertuples(index=False, name=None))


def load_metadata(dataset):
    metadata_path = os.path.join("data", dataset, "metadata", "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


def load_mia_in(dataset):
    csv_path = os.path.join("data", dataset, "splits", "mia_in.csv")
    df = pd.read_csv(csv_path)
    return list(df[['user_id','item_id','rating']].itertuples(index=False, name=None))


def load_mia_out(dataset):
    csv_path = os.path.join("data", dataset, "splits", "mia_out.csv")
    df = pd.read_csv(csv_path)
    return list(df[['user_id','item_id','rating']].itertuples(index=False, name=None))