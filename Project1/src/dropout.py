import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


SEED = 42
EPOCHS = 1000
PATIENCE = 20
MOMENTUM = 0.0
DROPOUT_P = 0.2


CONFIGS = {
    "random": {
        "hidden_layers": (128, 64),
        "activation": "ReLU",
        "learning_rate": 0.01,
        "batch_size": 16,
    },
    "systematic": {
        "hidden_layers": (128, 64),
        "activation": "ReLU",
        "learning_rate": 0.01,
        "batch_size": 32,
    },
}


class MLP(nn.Module):
    def __init__(self, hidden_layers, activation):
        super().__init__()

        layers = []
        input_size = 1

        for i, hidden_size in enumerate(hidden_layers):
            layers.append(
                nn.Linear(input_size, hidden_size)
            )

            if activation == "ReLU":
                layers.append(nn.ReLU())
            elif activation == "Tanh":
                layers.append(nn.Tanh())

            layers.append(
                nn.Dropout(p=DROPOUT_P)
            )

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


def train_model(split_name):
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

    config = CONFIGS[split_name]

    model = MLP(
        config["hidden_layers"],
        config["activation"],
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=MOMENTUM,
    )

    criterion = nn.MSELoss()

    train_dataset = TensorDataset(
        x_train, y_train
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
    )

    history = []

    best_val_mse = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            train_predictions = model(x_train)
            val_predictions = model(x_val)

            train_mse = criterion(
                train_predictions, y_train
            ).item()

            val_mse = criterion(
                val_predictions, y_val
            ).item()

            train_mae = torch.mean(
                torch.abs(train_predictions - y_train)
            ).item()

            val_mae = torch.mean(
                torch.abs(val_predictions - y_val)
            ).item()

        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_mae": train_mae,
            "val_mae": val_mae,
        })

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch

            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)

    best_row = history_df[
        history_df["epoch"] == best_epoch
    ].iloc[0]

    torch.save(
        model.state_dict(),
        f"results/dropout_{split_name}.pt",
    )

    history_df.to_csv(
        f"results/dropout_{split_name}_history.csv",
        index=False,
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        history_df["epoch"],
        history_df["train_mse"],
        label="Train MSE",
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_mse"],
        label="Validation MSE",
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(
        f"Dropout - {split_name.capitalize()} split"
    )
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"figures/dropout_{split_name}_curves.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return {
        "split": split_name,
        "hidden_layers": str(config["hidden_layers"]),
        "activation": config["activation"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "momentum": MOMENTUM,
        "dropout_p": DROPOUT_P,
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "epochs": len(history_df),
        "best_train_mse": best_row["train_mse"],
        "best_train_mae": best_row["train_mae"],
        "best_val_mae": best_row["val_mae"],
    }


results = []

for split_name in ["random", "systematic"]:
    result = train_model(split_name)
    results.append(result)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/dropout_results.csv",
    index=False,
)

for result in results:
    print(f"\n{result['split'].upper()}")
    print(f"best epoch: {result['best_epoch']}")
    print(
        f"best validation MSE: "
        f"{result['best_val_mse']:.6f}"
    )
    print(
        f"validation MAE: "
        f"{result['best_val_mae']:.6f}"
    )
    print(f"epochs: {result['epochs']}")

print("\nResultados:")
print(results_df.to_string(index=False))