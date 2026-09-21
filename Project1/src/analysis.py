import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


SPLITS = ["random", "systematic"]


def analyze_split(split_name):
    df = pd.read_csv(
        f"results/test_predictions_{split_name}.csv"
    )

    y = df["y"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    x = df["x"].to_numpy()
    residual = df["residual"].to_numpy()

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # paridade
    plt.figure(figsize=(7, 7))

    plt.scatter(
        y,
        y_pred,
        alpha=0.7,
    )

    min_value = min(y.min(), y_pred.min())
    max_value = max(y.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Ideal",
    )

    plt.xlabel("Valor real")
    plt.ylabel("Valor predito")
    plt.title(
        f"Gráfico de paridade - "
        f"{split_name.capitalize()} split"
    )
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"figures/parity_{split_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # resíduos vs x
    plt.figure(figsize=(10, 5))

    plt.scatter(
        x,
        residual,
        alpha=0.7,
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel("x")
    plt.ylabel("Resíduo")
    plt.title(
        f"Resíduos vs. x - "
        f"{split_name.capitalize()} split"
    )
    plt.grid(True)

    plt.savefig(
        f"figures/residuals_x_{split_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # resíduos vs previsão
    plt.figure(figsize=(10, 5))

    plt.scatter(
        y_pred,
        residual,
        alpha=0.7,
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel("Valor predito")
    plt.ylabel("Resíduo")
    plt.title(
        f"Resíduos vs. valor predito - "
        f"{split_name.capitalize()} split"
    )
    plt.grid(True)

    plt.savefig(
        f"figures/residuals_pred_{split_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # estatísticas dos resíduos
    residual_summary = {
        "split": split_name,
        "mean_residual": np.mean(residual),
        "std_residual": np.std(residual),
        "min_residual": np.min(residual),
        "max_residual": np.max(residual),
        "median_residual": np.median(residual),
        "rmse": rmse,
    }
        # histograma dos resíduos
    plt.figure(figsize=(10, 5))

    plt.hist(
        residual,
        bins=25,
        edgecolor="black",
    )

    plt.axvline(
        0,
        linestyle="--",
        label="Zero",
    )

    plt.axvline(
        np.mean(residual),
        linestyle="--",
        label="Média",
    )

    plt.xlabel("Resíduo")
    plt.ylabel("Frequência")
    plt.title(
        f"Distribuição dos resíduos - "
        f"{split_name.capitalize()} split"
    )
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"figures/residual_hist_{split_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return residual_summary


results = []

for split_name in SPLITS:
    results.append(
        analyze_split(split_name)
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/residual_summary.csv",
    index=False,
)

print("\nResumo dos resíduos:")
print(results_df.to_string(index=False))

