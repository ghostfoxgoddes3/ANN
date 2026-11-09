"""
Laço de treino para o classificador recorrente.

Responsabilidades:
  - padronização ajustada SOMENTE no treino (evita vazamento de escala);
  - balanceamento de classe via pos_weight no BCEWithLogitsLoss;
  - early stopping monitorando a loss de validação;
  - registro de histórico por época (loss e métricas).

As funções são reutilizadas pela busca de hiperparâmetros (search.py), pelo
treino final (main.py) e pela avaliação (evaluate.py).
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.model import RainClassifier

SEED = 42
MAX_EPOCHS = 100
PATIENCE = 10


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Média e desvio por variável, calculados sobre (N*WINDOW) do treino."""
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def make_loader(x, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate_logits(model, x, y, device):
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    logits = model(xt).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.float32)
    acc = float((preds == y).mean())
    f1 = float(f1_score(y, preds, zero_division=0))
    return acc, f1, probs


def train_model(
    x_train, y_train, x_val, y_val, config, device="cpu", verbose=False,
):
    """Treina um modelo e devolve (model, scaler, history, best)."""
    set_seed(config.get("seed", SEED))

    mean, std = fit_scaler(x_train)
    x_train_s = apply_scaler(x_train, mean, std)
    x_val_s = apply_scaler(x_val, mean, std)

    loader = make_loader(
        x_train_s, y_train, config["batch_size"], shuffle=True
    )

    model = RainClassifier(
        input_size=x_train.shape[-1],
        cell=config["cell"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config.get("dropout", 0.0),
    ).to(device)

    # pos_weight = negativos / positivos (compensa o desbalanceamento).
    n_pos = max(float(y_train.sum()), 1.0)
    n_neg = float(len(y_train) - y_train.sum())
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"],
    )

    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Loss de treino e validação ao final da época.
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(x_train_s, dtype=torch.float32, device=device)
            xv = torch.tensor(x_val_s, dtype=torch.float32, device=device)
            yt = torch.tensor(y_train, dtype=torch.float32, device=device)
            yv = torch.tensor(y_val, dtype=torch.float32, device=device)
            train_loss = criterion(model(xt), yt).item()
            val_loss = criterion(model(xv), yv).item()

        train_acc, train_f1, _ = evaluate_logits(model, x_train_s, y_train, device)
        val_acc, val_f1, _ = evaluate_logits(model, x_val_s, y_val, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
            "train_f1": train_f1, "val_f1": val_f1,
        })

        if verbose:
            print(f"  ep{epoch:3d} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(best_state)
    best = {"best_epoch": best_epoch, "best_val_loss": best_val,
            "epochs": len(history)}
    scaler = {"mean": mean, "std": std}
    return model, scaler, history, best
