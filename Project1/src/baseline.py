import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
EPOCHS = 1000
PATIENCE = 20


CONFIGS = {
    "random": {
        "hidden_layers": (128, 64),
        "learning_rate": 0.01,
        "batch_size": 16,
    },
    "systematic": {
        "hidden_layers": (128, 64),
        "learning_rate": 0.01,
        "batch_size": 32,
    },
}


#reprodutibilidade#
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#modelo mlp#
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super().__init__()

        layers = []
        previous_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


#treinamento do baseline#
def train_baseline(split_name, config):
    set_seed(SEED)

    data = np.load(
        f"results/{split_name}_preprocessed.npz"
    )

    x_train = torch.tensor(
        data["x_train"], dtype=torch.float32
    )
    y_train = torch.tensor(
        data["y_train"], dtype=torch.float32
    )
    x_val = torch.tensor(
        data["x_val"], dtype=torch.float32
    )
    y_val = torch.tensor(
        data["y_val"], dtype=torch.float32
    )

    dataset = TensorDataset(x_train, y_train)

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    model = MLP(
        input_size=1,
        hidden_layers=config["hidden_layers"],
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=0.0,
    )

    history = []

    best_val_mse = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()

        for x_batch, y_batch in loader:
            optimizer.zero_grad()

            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)

            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            train_prediction = model(x_train)
            val_prediction = model(x_val)

            train_mse = torch.mean(
                (train_prediction - y_train) ** 2
            ).item()

            val_mse = torch.mean(
                (val_prediction - y_val) ** 2
            ).item()

            train_mae = torch.mean(
                torch.abs(train_prediction - y_train)
            ).item()

            val_mae = torch.mean(
                torch.abs(val_prediction - y_val)
            ).item()

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "train_mae": train_mae,
                "val_mae": val_mae,
            }
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        f"results/baseline_{split_name}_history.csv",
        index=False,
    )

    torch.save(
        model.state_dict(),
        f"results/baseline_{split_name}.pt",
    )

    return history_df, best_epoch, best_val_mse


#curvas de treinamento#
def plot_history(history_df, split_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        history_df["epoch"],
        history_df["train_mse"],
        label="Treino",
    )
    axes[0].plot(
        history_df["epoch"],
        history_df["val_mse"],
        label="Validação",
    )

    axes[0].set_title(
        f"MSE - divisão {split_name}"
    )
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        history_df["epoch"],
        history_df["train_mae"],
        label="Treino",
    )
    axes[1].plot(
        history_df["epoch"],
        history_df["val_mae"],
        label="Validação",
    )

    axes[1].set_title(
        f"MAE - divisão {split_name}"
    )
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"figures/baseline_{split_name}_curves.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main():
    results = []

    for split_name, config in CONFIGS.items():

        history, best_epoch, best_val_mse = train_baseline(
            split_name,
            config,
        )

        plot_history(
            history,
            split_name,
        )

        final_row = history.iloc[-1]

        results.append(
            {
                "split": split_name,
                "hidden_layers": str(
                    config["hidden_layers"]
                ),
                "activation": "ReLU",
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "momentum": 0.0,
                "best_val_mse": best_val_mse,
                "best_epoch": best_epoch,
                "epochs": len(history),
                "final_train_mse": final_row["train_mse"],
                "final_val_mse": final_row["val_mse"],
                "final_train_mae": final_row["train_mae"],
                "final_val_mae": final_row["val_mae"],
            }
        )

        print(f"\n{split_name.upper()}")
        print(f"best epoch: {best_epoch}")
        print(f"best validation MSE: {best_val_mse:.6f}")
        print(f"epochs: {len(history)}")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/baseline_results.csv",
        index=False,
    )

    print("\nResultados:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()