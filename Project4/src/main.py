"""
Treino final dos modelos selecionados e geração das curvas de aprendizagem.

Lê a melhor configuração encontrada por search.py para cada estratégia de
partição, treina o modelo correspondente, salva o checkpoint, o scaler, o
histórico e a figura com as curvas de loss/F1.

Uso:
    python -m src.main
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.search import load_split, SPLITS
from src.train import train_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def best_config(split_name: str) -> dict:
    df = pd.read_csv(RESULTS_DIR / f"search_{split_name}.csv")
    row = df.sort_values("best_val_loss").iloc[0]
    return {
        "cell": str(row["cell"]),
        "hidden_size": int(row["hidden_size"]),
        "num_layers": int(row["num_layers"]),
        "lr": float(row["lr"]),
        "batch_size": int(row["batch_size"]),
        "dropout": float(row["dropout"]),
    }


def plot_history(history_df: pd.DataFrame, split_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Treino")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validação")
    axes[0].set_title(f"Loss (BCE) - {split_name}")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["train_f1"], label="Treino")
    axes[1].plot(history_df["epoch"], history_df["val_f1"], label="Validação")
    axes[1].set_title(f"F1 - {split_name}")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("F1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / f"curves_{split_name}.png", dpi=200,
                bbox_inches="tight")
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for split_name in SPLITS:
        config = best_config(split_name)
        x_tr, y_tr, x_val, y_val = load_split(split_name)

        model, scaler, history, best = train_model(
            x_tr, y_tr, x_val, y_val, config, device=device
        )

        hist_df = pd.DataFrame(history)
        hist_df.to_csv(RESULTS_DIR / f"history_{split_name}.csv", index=False)
        plot_history(hist_df, split_name)

        torch.save(
            {"state_dict": model.state_dict(),
             "config": config,
             "mean": scaler["mean"], "std": scaler["std"]},
            RESULTS_DIR / f"model_{split_name}.pt",
        )

        rows.append({"split": split_name, **config, **best})
        print(f"[{split_name}] {config['cell']} h={config['hidden_size']} "
              f"L={config['num_layers']} -> val_loss="
              f"{best['best_val_loss']:.4f} (época {best['best_epoch']})")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "final_models.csv", index=False)


if __name__ == "__main__":
    main()
