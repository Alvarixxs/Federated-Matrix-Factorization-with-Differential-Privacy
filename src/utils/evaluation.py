import json
import os
import pandas as pd


def save_evaluation(save_path, rmse_train, rmse_val, rmse_test, eps=None):
    results = {
        "rmse_train": rmse_train,
        "rmse_val": rmse_val,
        "rmse_test": rmse_test,
    }
    if eps is not None:
        results["epsilon"] = eps

    save_path = os.path.join(save_path, "eval.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


def load_evaluation(base_path):
    eval_path = os.path.join(base_path, "eval.json")
    with open(eval_path, "r") as f:
        data = json.load(f)
    return data["rmse_train"], data["rmse_val"], data["rmse_test"], data.get("epsilon", None)


def save_mia_features(base_path, rows, split):
    attack_path = os.path.join(base_path, "mia_attack")
    os.makedirs(attack_path, exist_ok=True)

    df = pd.DataFrame(
        rows,
        columns=["score", "error", "squared_error", "norm_p", "norm_q", "reg_loss", "centered_score", "label"]
    )

    df.to_csv(os.path.join(attack_path, f"{split}_features.csv"),
              index=False)


def load_mia_features(base_path, split):
    path = os.path.join(base_path, "mia_attack",
                        f"{split}_features.csv")
    return pd.read_csv(path)


def save_mia_dataset(base_path, full_df, train_df, test_df):
    attack_path = os.path.join(base_path, "mia_attack")
    os.makedirs(attack_path, exist_ok=True)

    full_df.to_csv(os.path.join(attack_path, "mia_dataset.csv"),
                   index=False)
    train_df.to_csv(os.path.join(attack_path, "mia_train.csv"),
                    index=False)
    test_df.to_csv(os.path.join(attack_path, "mia_test.csv"),
                   index=False)


def load_mia_dataset(base_path):
    attack_path = os.path.join(base_path, "mia_attack")
    train_df = pd.read_csv(os.path.join(attack_path, "mia_train.csv"))
    test_df = pd.read_csv(os.path.join(attack_path, "mia_test.csv"))
    return train_df, test_df


def save_attack_results(base_path, results):
    attack_path = os.path.join(base_path, "mia_attack")
    os.makedirs(attack_path, exist_ok=True)

    with open(os.path.join(attack_path, "threshold_results.json"), "w") as f:
        json.dump(results, f, indent=4)


def load_attack_results(base_path):
    attack_path = os.path.join(base_path, "mia_attack")
    with open(os.path.join(attack_path, "threshold_results.json"), "r") as f:
        results = json.load(f)
    return results


def save_classifier_results(base_path, results):
    attack_path = os.path.join(base_path, "mia_attack")
    os.makedirs(attack_path, exist_ok=True)

    with open(os.path.join(attack_path, "classifier_results.json"), "w") as f:
        json.dump(results, f, indent=4)


def load_classifier_results(base_path):
    attack_path = os.path.join(base_path, "mia_attack")
    with open(os.path.join(attack_path, "classifier_results.json"), "r") as f:
        results = json.load(f)
    return results