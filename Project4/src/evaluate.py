"""
Avaliação final no conjunto de teste e comparação entre as partições.

Para cada estratégia, carrega o modelo treinado, aplica no conjunto de teste
(intocado até aqui) e calcula acurácia, precisão, revocação, F1, AUC e a
matriz de confusão. Em seguida compara explicitamente os resultados das duas
partições para quantificar o efeito do vazamento temporal.

Uso:
    python -m src.evaluate
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from src.model import RainClassifier
from src.train import apply_scaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

SPLITS = ("chronological", "random")


def load_test(split_name):
    data = np.load(DATA_DIR / "windows.npz", allow_pickle=True)
    idx = np.load(RESULTS_DIR / f"split_{split_name}.npz")
    return data["x"][idx["test"]], data["y"][idx["test"]]


def load_model(split_name, input_size):
    ckpt = torch.load(RESULTS_DIR / f"model_{split_name}.pt",
                      map_location="cpu", weights_only=False)
    model = RainClassifier(
        input_size=input_size,
        cell=ckpt["config"]["cell"],
        hidden_size=ckpt["config"]["hidden_size"],
        num_layers=ckpt["config"]["num_layers"],
        dropout=ckpt["config"].get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def plot_confusion(cm, split_name):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["sem chuva", "chuva"])
    ax.set_yticklabels(["sem chuva", "chuva"])
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusão - {split_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / f"confusion_{split_name}.png", dpi=200,
                bbox_inches="tight")
    plt.close()


def main() -> None:
    rows = []
    for split_name in SPLITS:
        x_test, y_test = load_test(split_name)
        model, ckpt = load_model(split_name, x_test.shape[-1])

        x_s = apply_scaler(x_test, ckpt["mean"], ckpt["std"])
        with torch.no_grad():
            logits = model(torch.tensor(x_s, dtype=torch.float32)).numpy()
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)

        cm = confusion_matrix(y_test, preds)
        plot_confusion(cm, split_name)

        rows.append({
            "split": split_name,
            "cell": ckpt["config"]["cell"],
            "n_test": len(y_test),
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "auc": roc_auc_score(y_test, probs),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "test_results.csv", index=False)
    print("\nResultados no conjunto de teste:")
    print(df.to_string(index=False))

    # Comparação explícita entre partições (efeito do vazamento temporal).
    chrono = df[df["split"] == "chronological"].iloc[0]
    rand = df[df["split"] == "random"].iloc[0]
    delta = {
        m: float(rand[m] - chrono[m])
        for m in ("accuracy", "precision", "recall", "f1", "auc")
    }
    comp = pd.DataFrame([
        {"metric": m, "chronological": float(chrono[m]),
         "random": float(rand[m]), "random_minus_chrono": delta[m]}
        for m in ("accuracy", "precision", "recall", "f1", "auc")
    ])
    comp.to_csv(RESULTS_DIR / "split_comparison.csv", index=False)
    print("\nComparação entre partições (random - cronológico):")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
