"""
Busca de hiperparâmetros por conjunto de validação.

Para cada estratégia de partição (cronológica e aleatória) percorremos um
grid de configurações variando a célula recorrente (LSTM/GRU), o tamanho do
estado oculto, o número de camadas, a taxa de aprendizado e o batch. O
critério de seleção é a menor loss de validação -- o conjunto de teste
permanece intocado.

A busca pode ser executada por partição (argumento opcional) para paralelizar
ou retomar o trabalho:

    python -m src.search                 # ambas as partições
    python -m src.search chronological   # apenas uma
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.train import train_model, evaluate_logits, apply_scaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SPLITS = ("chronological", "random")

GRID = {
    "cell": ["LSTM", "GRU"],
    "hidden_size": [32, 64],
    "num_layers": [1, 2],
    "lr": [1e-3],
    "batch_size": [128],
    "dropout": [0.0],
}


def load_split(split_name):
    data = np.load(DATA_DIR / "windows.npz", allow_pickle=True)
    idx = np.load(RESULTS_DIR / f"split_{split_name}.npz")
    x, y = data["x"], data["y"]
    return (
        x[idx["train"]], y[idx["train"]],
        x[idx["val"]], y[idx["val"]],
    )


def iter_configs():
    keys = list(GRID.keys())
    for values in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, values))


def search_split(split_name, device):
    x_tr, y_tr, x_val, y_val = load_split(split_name)
    rows = []
    print(f"\n=== busca [{split_name}] ===", flush=True)

    for config in iter_configs():
        model, scaler, history, best = train_model(
            x_tr, y_tr, x_val, y_val, config, device=device
        )
        x_val_s = apply_scaler(x_val, scaler["mean"], scaler["std"])
        val_acc, val_f1, _ = evaluate_logits(model, x_val_s, y_val, device)

        row = {**config,
               "best_val_loss": best["best_val_loss"],
               "best_epoch": best["best_epoch"],
               "epochs": best["epochs"],
               "val_acc": val_acc, "val_f1": val_f1}
        rows.append(row)
        print(f"  {config['cell']} h={config['hidden_size']} "
              f"L={config['num_layers']} -> "
              f"val_loss={best['best_val_loss']:.4f} f1={val_f1:.4f}",
              flush=True)

    df = pd.DataFrame(rows).sort_values("best_val_loss")
    df.to_csv(RESULTS_DIR / f"search_{split_name}.csv", index=False)
    print(f"melhor [{split_name}]: {df.iloc[0].to_dict()}", flush=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    which = sys.argv[1:] if len(sys.argv) > 1 else list(SPLITS)
    for split_name in which:
        search_split(split_name, device)


if __name__ == "__main__":
    main()
