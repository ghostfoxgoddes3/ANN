import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


SEED = 42


CONFIGS = {
    "random": {
        "model": "momentum",
        "hidden_layers": (128, 64),
    },
    "systematic": {
        "model": "baseline",
        "hidden_layers": (128, 64),
    },
}


#reprodutibilidade#
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


#modelo mlp#
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super().__init__()

        layers = []
        previous_size = input_size

        for hidden_size in hidden_layers:
            layers.append(
                nn.Linear(
                    previous_size,
                    hidden_size,
                )
            )
            layers.append(nn.ReLU())
            previous_size = hidden_size

        layers.append(
            nn.Linear(
                previous_size,
                1,
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


#carrega o modelo treinado#
def load_model(split_name, model_name, hidden_layers):
    model = MLP(
        input_size=1,
        hidden_layers=hidden_layers,
    )

    model.load_state_dict(
        torch.load(
            f"results/{model_name}_{split_name}.pt",
            map_location="cpu",
        )
    )

    model.eval()

    return model


#preprocessamento usando somente o treino#
def prepare_data(split_name):
    df = pd.read_csv("data/dataset_p1.csv")
    split_df = pd.read_csv(
        "results/split_assignments.csv"
    )

    train_mask = (
        split_df[f"{split_name}_split"] == "train"
    )

    x_train = df.loc[
        train_mask,
        ["x"],
    ].to_numpy()

    y_train = df.loc[
        train_mask,
        ["y"],
    ].to_numpy()

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(x_train)
    y_scaler.fit(y_train)

    return (
        df,
        split_df,
        x_scaler,
        y_scaler,
    )


#gera as previsoes do modelo#
def predict_model(
    model,
    x,
    x_scaler,
    y_scaler,
):
    x_scaled = x_scaler.transform(
        x.reshape(-1, 1)
    )

    x_tensor = torch.tensor(
        x_scaled,
        dtype=torch.float32,
    )

    with torch.no_grad():
        prediction_scaled = model(
            x_tensor
        ).numpy()

    prediction = y_scaler.inverse_transform(
        prediction_scaled
    )

    return prediction.ravel()


#calcula metricas no conjunto de teste#
def calculate_metrics(
    model,
    df,
    split_df,
    split_name,
    x_scaler,
    y_scaler,
):
    test_mask = (
        split_df[f"{split_name}_split"] == "test"
    )

    x_test = df.loc[
        test_mask,
        "x",
    ].to_numpy()

    y_test = df.loc[
        test_mask,
        "y",
    ].to_numpy()

    prediction = predict_model(
        model,
        x_test,
        x_scaler,
        y_scaler,
    )

    error = y_test - prediction

    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum(
        (y_test - np.mean(y_test)) ** 2
    )

    r2 = 1 - ss_res / ss_tot

    return mse, rmse, mae, r2


#plot dos modelos finais#
def plot_final_models():
    set_seed(SEED)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharex=True,
        sharey=True,
    )

    results = []

    for ax, (split_name, config) in zip(
        axes,
        CONFIGS.items(),
    ):

        model = load_model(
            split_name,
            config["model"],
            config["hidden_layers"],
        )

        (
            df,
            split_df,
            x_scaler,
            y_scaler,
        ) = prepare_data(split_name)

        x = df["x"].to_numpy()
        y = df["y"].to_numpy()

        train_mask = (
            split_df[f"{split_name}_split"]
            == "train"
        )

        val_mask = (
            split_df[f"{split_name}_split"]
            == "val"
        )

        test_mask = (
            split_df[f"{split_name}_split"]
            == "test"
        )

        #pontos do dataset#
        ax.scatter(
            x,
            y,
            label="Dataset",
            alpha=0.35,
            s=18,
        )

        #pontos de treino#
        ax.scatter(
            x[train_mask],
            y[train_mask],
            label="Treino",
            s=35,
            marker="o",
        )

        #pontos de validação#
        ax.scatter(
            x[val_mask],
            y[val_mask],
            label="Validação",
            s=40,
            marker="s",
        )

        #pontos de teste#
        ax.scatter(
            x[test_mask],
            y[test_mask],
            label="Teste",
            s=22,
            marker="x",
        )

        #grade para representar a funcao aprendida#
        x_grid = np.linspace(
            x.min(),
            x.max(),
            1000,
        )

        y_grid = predict_model(
            model,
            x_grid,
            x_scaler,
            y_scaler,
        )

        ax.plot(
            x_grid,
            y_grid,
            linewidth=2.5,
            label="Modelo final",
        )

        #metricas no conjunto de teste#
        mse, rmse, mae, r2 = calculate_metrics(
            model,
            df,
            split_df,
            split_name,
            x_scaler,
            y_scaler,
        )

        results.append(
            {
                "split": split_name,
                "model": config["model"],
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )

        model_name = (
            "Momentum"
            if config["model"] == "momentum"
            else "Baseline"
        )

        split_title = (
            "Partição aleatória"
            if split_name == "random"
            else "Partição sistemática"
        )

        ax.set_title(
            f"Modelo final - {split_title}\n"
            f"{model_name}"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.grid(alpha=0.3)

        ax.text(
            0.03,
            0.97,
            f"MSE = {mse:.4f}\n"
            f"RMSE = {rmse:.4f}\n"
            f"MAE = {mae:.4f}\n"
            f"$R^2$ = {r2:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8,
            ),
        )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Modelos finais sobre o dataset_p1",
        fontsize=14,
    )

    plt.tight_layout(
        rect=(0, 0.08, 1, 1)
    )

    plt.savefig(
        "figures/final_models_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/final_models_metrics.csv",
        index=False,
    )

    print("\nResultados dos modelos finais:")
    print(
        results_df.to_string(
            index=False
        )
    )


def main():
    plot_final_models()


if __name__ == "__main__":
    main()