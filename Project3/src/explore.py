"""Exploracao do espaco latente dos VAEs treinados.

Gera, para cada dimensao latente:
  - dispersao do espaco latente colorida pelos rotulos
    (2D direto; 8D projetado em 2D via PCA, com variancia explicada);
  - curvas de treino/validacao (loss total, reconstrucao e KL);
  - grade de reconstrucoes e, no caso 2D, varredura do espaco latente.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.data import CLASS_NAMES, build_splits
from src.vae import VAE

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

CMAP = plt.get_cmap("tab10")


def load_model(latent_dim: int) -> VAE:
    ckpt = torch.load(RESULTS_DIR / f"vae_latent{latent_dim}.pt", map_location="cpu")
    model = VAE(latent_dim=ckpt["latent_dim"], hidden=tuple(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def encode_set(model: VAE, x: np.ndarray) -> np.ndarray:
    """Retorna a media do posterior (mu) para cada amostra."""
    with torch.no_grad():
        mu, _ = model.encode(torch.from_numpy(x).float())
    return mu.numpy()


def scatter_latent(coords: np.ndarray, labels: np.ndarray, title: str,
                   out_path: Path, axis_labels=("z1", "z2")) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for c in range(10):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, alpha=0.5,
                   color=CMAP(c), label=CLASS_NAMES[c])
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.legend(markerscale=3, fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_history(latent_dim: int) -> None:
    path = RESULTS_DIR / f"vae_latent{latent_dim}_history.csv"
    epochs, tr, va, rec, kl = [], [], [], [], []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            epochs.append(int(row["epoch"]))
            tr.append(float(row["train_loss"]))
            va.append(float(row["val_loss"]))
            rec.append(float(row["val_rec"]))
            kl.append(float(row["val_kl"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(epochs, tr, label="treino")
    ax1.plot(epochs, va, label="validacao")
    ax1.set_title(f"ELBO negativo ({latent_dim}D)")
    ax1.set_xlabel("epoca"); ax1.set_ylabel("loss"); ax1.legend()

    ax2.plot(epochs, rec, label="reconstrucao (val)")
    ax2.plot(epochs, kl, label="KL (val)")
    ax2.set_title(f"Decomposicao da loss ({latent_dim}D)")
    ax2.set_xlabel("epoca"); ax2.set_ylabel("nats"); ax2.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"history_latent{latent_dim}.png", dpi=140)
    plt.close(fig)


def reconstruction_grid(model: VAE, x: np.ndarray, latent_dim: int, n: int = 10) -> None:
    with torch.no_grad():
        recon, _, _ = model(torch.from_numpy(x[:n]).float())
    recon = recon.numpy().reshape(n, 28, 28)
    orig = x[:n].reshape(n, 28, 28)

    fig, axes = plt.subplots(2, n, figsize=(n, 2.4))
    for i in range(n):
        axes[0, i].imshow(orig[i], cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(recon[i], cmap="gray"); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("orig")
    fig.suptitle(f"Reconstrucoes VAE {latent_dim}D (topo: original, base: reconstruido)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"reconstructions_latent{latent_dim}.png", dpi=140)
    plt.close(fig)


def latent_sweep_2d(model: VAE, out_path: Path, steps: int = 18, span: float = 2.5) -> None:
    """Varre uma grade no plano latente 2D e decodifica cada ponto."""
    grid = np.linspace(-span, span, steps)
    canvas = np.zeros((28 * steps, 28 * steps))
    with torch.no_grad():
        for i, zi in enumerate(grid):
            for j, zj in enumerate(grid):
                z = torch.tensor([[zj, zi]], dtype=torch.float32)
                img = model.decode(z).numpy().reshape(28, 28)
                canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = img
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canvas, cmap="gray", extent=[-span, span, -span, span])
    ax.set_title("Varredura do espaco latente 2D")
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    splits = build_splits(seed=42, val_fraction=0.1)
    x_test, y_test = splits["x_test"], splits["y_test"]

    summary_rows = []

    # --- Latente 2D ---
    model2 = load_model(2)
    z2 = encode_set(model2, x_test)
    scatter_latent(z2, y_test, "Espaco latente 2D (VAE) - conjunto de teste",
                   FIGURES_DIR / "latent2d_scatter.png")
    plot_history(2)
    reconstruction_grid(model2, x_test, 2)
    latent_sweep_2d(model2, FIGURES_DIR / "latent2d_sweep.png")
    summary_rows.append({"latent_dim": 2, "projection": "direta",
                         "explained_var_pc1": "", "explained_var_pc2": "",
                         "explained_var_2d": ""})

    # --- Latente 8D projetado em 2D via PCA ---
    model8 = load_model(8)
    z8 = encode_set(model8, x_test)
    pca = PCA(n_components=8)
    z8_proj_full = pca.fit_transform(z8)
    evr = pca.explained_variance_ratio_
    z8_2d = z8_proj_full[:, :2]
    title = (f"Espaco latente 8D -> PCA 2D "
             f"(var. explicada: PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%, "
             f"soma={100*(evr[0]+evr[1]):.1f}%)")
    scatter_latent(z8_2d, y_test, title,
                   FIGURES_DIR / "latent8d_pca_scatter.png",
                   axis_labels=("PC1", "PC2"))
    plot_history(8)
    reconstruction_grid(model8, x_test, 8)
    summary_rows.append({"latent_dim": 8, "projection": "PCA",
                         "explained_var_pc1": round(float(evr[0]), 4),
                         "explained_var_pc2": round(float(evr[1]), 4),
                         "explained_var_2d": round(float(evr[0] + evr[1]), 4)})

    # Grafico de variancia explicada acumulada do PCA (8D)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, 9), evr * 100, alpha=0.6, label="individual")
    ax.plot(range(1, 9), np.cumsum(evr) * 100, "o-", color="k", label="acumulada")
    ax.set_xlabel("componente principal"); ax.set_ylabel("variancia explicada (%)")
    ax.set_title("PCA do espaco latente 8D")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "latent8d_pca_variance.png", dpi=140)
    plt.close(fig)

    with (RESULTS_DIR / "latent_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Figuras geradas em figures/. Resumo em results/latent_summary.csv")
    print(f"PCA 8D: variancia explicada 2D = {100*(evr[0]+evr[1]):.1f}%")


if __name__ == "__main__":
    main()
