import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


SEED = 42


CONFIGS = {
    "random": {
        "model": "momentum",
        "hidden_layers": (128, 64),
        "activation": "ReLU",
    },
    "systematic": {
        "model": "baseline",
        "hidden_layers": (128, 64),
        "activation": "ReLU",
    },
}


class MLP(nn.Module):
    def __init__(self, hidden_layers, activation):
        super().__init__()

        layers = []
        input_size = 1

        for hidden_size in hidden_layers:
            layers.append(
                nn.Linear(input_size, hidden_size)
            )

            if activation == "ReLU":
                layers.append(nn.ReLU())
            elif activation == "Tanh":
                layers.append(nn.Tanh())

            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_y_scaler(split_df, split_name):
    train_mask = (
        split_df[f"{split_name}_split"] == "train"
    )

    y_train = split_df.loc[
        train_mask, ["y"]
    ].to_numpy()

    y_scaler = StandardScaler()
    y_scaler.fit(y_train)

    return y_scaler


def evaluate_split(split_name):
    set_seed(SEED)

    data = np.load(
        f"results/{split_name}_preprocessed.npz"
    )

    x_test = torch.tensor(
        data["x_test"], dtype=torch.float32
    )

    y_test = torch.tensor(
        data["y_test"], dtype=torch.float32
    )

    config = CONFIGS[split_name]

    model = MLP(
        config["hidden_layers"],
        config["activation"],
    )

    checkpoint_path = (
        f"results/{config['model']}_{split_name}.pt"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    with torch.no_grad():
        predictions_scaled = model(x_test).numpy()

    y_test_scaled = y_test.numpy()

    mse_scaled = mean_squared_error(
        y_test_scaled,
        predictions_scaled,
    )

    rmse_scaled = np.sqrt(mse_scaled)

    mae_scaled = mean_absolute_error(
        y_test_scaled,
        predictions_scaled,
    )

    r2_scaled = r2_score(
        y_test_scaled,
        predictions_scaled,
    )

    split_df = pd.read_csv(
        "results/split_assignments.csv"
    )

    test_mask = (
        split_df[f"{split_name}_split"] == "test"
    )

    test_df = split_df.loc[
        test_mask, ["index", "x", "y"]
    ].copy()

    y_scaler = get_y_scaler(
        split_df,
        split_name,
    )

    y_test_original = test_df[
        ["y"]
    ].to_numpy()

    y_pred_original = y_scaler.inverse_transform(
        predictions_scaled
    )

    mse_original = mean_squared_error(
        y_test_original,
        y_pred_original,
    )

    rmse_original = np.sqrt(mse_original)

    mae_original = mean_absolute_error(
        y_test_original,
        y_pred_original,
    )

    r2_original = r2_score(
        y_test_original,
        y_pred_original,
    )

    test_df["y_pred"] = y_pred_original.ravel()

    test_df["residual"] = (
        test_df["y"] - test_df["y_pred"]
    )

    test_df["model"] = config["model"]

    test_df.to_csv(
        f"results/test_predictions_{split_name}.csv",
        index=False,
    )

    return {
        "split": split_name,
        "model": config["model"],
        "mse_scaled": mse_scaled,
        "rmse_scaled": rmse_scaled,
        "mae_scaled": mae_scaled,
        "r2_scaled": r2_scaled,
        "mse_original": mse_original,
        "rmse_original": rmse_original,
        "mae_original": mae_original,
        "r2_original": r2_original,
        "n_test": len(test_df),
    }


results = []

for split_name in ["random", "systematic"]:
    result = evaluate_split(split_name)
    results.append(result)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/final_test_results.csv",
    index=False,
)

print("\nResultados finais no conjunto de teste:")

print(
    results_df[
        [
            "split",
            "model",
            "n_test",
            "mse_scaled",
            "rmse_scaled",
            "mae_scaled",
            "r2_scaled",
        ]
    ].to_string(index=False)
)

print("\nResultados no espaço original:")

print(
    results_df[
        [
            "split",
            "model",
            "n_test",
            "mse_original",
            "rmse_original",
            "mae_original",
            "r2_original",
        ]
    ].to_string(index=False)
)