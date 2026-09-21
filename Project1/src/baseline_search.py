import itertools
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
EPOCHS = 1000
PATIENCE = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, activation):
        super().__init__()

        layers = []
        previous_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(activation())
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(
    x_train,
    y_train,
    x_val,
    y_val,
    hidden_layers,
    activation,
    learning_rate,
    batch_size,
):
    set_seed(SEED)

    model = MLP(
        input_size=x_train.shape[1],
        hidden_layers=hidden_layers,
        activation=activation,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.0,
    )

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()

        batch_losses = []

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)

        model.eval()

        with torch.no_grad():
            val_predictions = model(x_val_tensor)
            val_loss = criterion(
                val_predictions,
                y_val_tensor,
            ).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs": len(train_losses),
    }


def load_data(split_name):
    data = np.load(
        f"results/{split_name}_preprocessed.npz"
    )

    return (
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        data["x_test"],
        data["y_test"],
    )


def run_search(split_name):
    x_train, y_train, x_val, y_val, _, _ = load_data(
        split_name
    )

    hidden_layer_options = [
        (32,),
        (64,),
        (128,),
        (32, 32),
        (64, 64),
        (128, 64),
    ]

    activation_options = [
        ("ReLU", nn.ReLU),
        ("Tanh", nn.Tanh),
    ]

    learning_rates = [0.001, 0.01]
    batch_sizes = [8, 16, 32]

    results = []

    combinations = itertools.product(
        hidden_layer_options,
        activation_options,
        learning_rates,
        batch_sizes,
    )

    for hidden_layers, (activation_name, activation), lr, batch_size in combinations:

        print(
            f"{split_name}: "
            f"layers={hidden_layers}, "
            f"activation={activation_name}, "
            f"lr={lr}, "
            f"batch={batch_size}"
        )

        result = train_model(
            x_train,
            y_train,
            x_val,
            y_val,
            hidden_layers,
            activation,
            lr,
            batch_size,
        )

        results.append(
            {
                "hidden_layers": str(hidden_layers),
                "activation": activation_name,
                "learning_rate": lr,
                "batch_size": batch_size,
                "best_val_mse": result["best_val_loss"],
                "best_epoch": result["best_epoch"],
                "epochs": result["epochs"],
                "final_train_mse": result["train_losses"][-1],
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "best_val_mse"
    ).reset_index(drop=True)

    results_df.to_csv(
        f"results/baseline_search_{split_name}.csv",
        index=False,
    )

    print()
    print(f"Melhor configuração para {split_name}:")
    print(results_df.iloc[0])

    return results_df


if __name__ == "__main__":
    run_search("random")
    run_search("systematic")