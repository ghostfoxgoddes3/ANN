import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "data/dataset_p1.csv"
SPLIT_PATH = "results/split_assignments.csv"


def preprocess_split(df, split_name):
    train_mask = df[f"{split_name}_split"] == "train"
    val_mask = df[f"{split_name}_split"] == "val"
    test_mask = df[f"{split_name}_split"] == "test"

    x_train = df.loc[train_mask, ["x"]].to_numpy()
    x_val = df.loc[val_mask, ["x"]].to_numpy()
    x_test = df.loc[test_mask, ["x"]].to_numpy()

    y_train = df.loc[train_mask, ["y"]].to_numpy()
    y_val = df.loc[val_mask, ["y"]].to_numpy()
    y_test = df.loc[test_mask, ["y"]].to_numpy()

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # ajusta os parâmetros somente no conjunto de treinamento
    x_train_scaled = x_scaler.fit_transform(x_train)
    y_train_scaled = y_scaler.fit_transform(y_train)

    # aplica os mesmos parâmetros aos demais conjuntos
    x_val_scaled = x_scaler.transform(x_val)
    x_test_scaled = x_scaler.transform(x_test)

    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)

    return {
        "x_train": x_train_scaled,
        "x_val": x_val_scaled,
        "x_test": x_test_scaled,
        "y_train": y_train_scaled,
        "y_val": y_val_scaled,
        "y_test": y_test_scaled,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }


def save_split(data, split_name):
    np.savez(
        f"results/{split_name}_preprocessed.npz",
        x_train=data["x_train"],
        x_val=data["x_val"],
        x_test=data["x_test"],
        y_train=data["y_train"],
        y_val=data["y_val"],
        y_test=data["y_test"],
    )


df = pd.read_csv(DATASET_PATH)
split_df = pd.read_csv(SPLIT_PATH)

assert len(df) == len(split_df)

data_random = preprocess_split(split_df, "random")
data_systematic = preprocess_split(split_df, "systematic")

save_split(data_random, "random")
save_split(data_systematic, "systematic")

print("Preprocessing completed.")

for split_name, data in [
    ("random", data_random),
    ("systematic", data_systematic),
]:
    print(f"{split_name.capitalize()} split:")
    print(f"  train: {data['x_train'].shape[0]}")
    print(f"  validation: {data['x_val'].shape[0]}")
    print(f"  test: {data['x_test'].shape[0]}")
    print()