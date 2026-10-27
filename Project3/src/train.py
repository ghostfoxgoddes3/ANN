"""Treinamento e busca de topologia para VAEs no Fashion-MNIST."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.vae import VAE, elbo_loss

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: VAE, loader, device, beta: float) -> tuple[float, float, float]:
    """Loss media (ELBO neg), reconstrucao e KL no conjunto fornecido."""
    model.eval()
    tot_loss = tot_rec = tot_kl = 0.0
    n = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, rec, kl = elbo_loss(recon, x, mu, logvar, beta)
            bs = x.size(0)
            tot_loss += loss.item() * bs
            tot_rec += rec.item() * bs
            tot_kl += kl.item() * bs
            n += bs
    return tot_loss / n, tot_rec / n, tot_kl / n


def train_model(model: VAE, train_loader, val_loader, device,
                epochs: int, lr: float, beta: float,
                patience: int = 6, verbose: bool = True):
    """Treina com Adam e early stopping pela loss de validacao."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    history = []
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, _, _ = elbo_loss(recon, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()

        tr_loss, tr_rec, tr_kl = evaluate(model, train_loader, device, beta)
        va_loss, va_rec, va_kl = evaluate(model, val_loader, device, beta)
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_rec": tr_rec, "train_kl": tr_kl,
            "val_loss": va_loss, "val_rec": va_rec, "val_kl": va_kl,
        })
        if verbose:
            print(f"  epoca {epoch:02d} | train {tr_loss:.3f} | val {va_loss:.3f} "
                  f"(rec {va_rec:.3f}, kl {va_kl:.3f})")

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  early stopping na epoca {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_val": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
    }
