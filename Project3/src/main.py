"""Pipeline principal do Projeto 3: VAEs para Fashion-MNIST.

Para cada dimensao de espaco latente (2 e 8), realiza uma busca de topologia
e seleciona o melhor modelo pela loss de validacao (ELBO negativo). O melhor
modelo de cada dimensao e salvo em results/ junto com o historico de treino
e as tabelas de busca.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.data import build_splits, make_loader
from src.train import RESULTS_DIR, evaluate, set_seed, train_model
from src.vae import VAE

torch.set_num_threads(6)

SEED = 42
BATCH_SIZE = 256
SEARCH_EPOCHS = 8        # epocas por candidato na busca
FINAL_EPOCHS = 20        # epocas para re-treinar o melhor
SEARCH_SUBSET = 20000    # subamostra de treino usada apenas na busca
BETA = 1.0
LR = 1e-3

# Espaco de busca de topologia (hidden layers) por dimensao latente.
TOPOLOGIES = [
    (256,),
    (512, 256),
    (512, 256, 128),
]
LATENT_DIMS = (2, 8)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    print("Carregando Fashion-MNIST ...")
    splits = build_splits(seed=SEED, val_fraction=0.1)
    print(f"Treino: {len(splits['x_train'])} | Val: {len(splits['x_val'])} "
          f"| Teste: {len(splits['x_test'])}")

    train_loader = make_loader(splits["x_train"], splits["y_train"], BATCH_SIZE, True)
    val_loader = make_loader(splits["x_val"], splits["y_val"], BATCH_SIZE, False)
    test_loader = make_loader(splits["x_test"], splits["y_test"], BATCH_SIZE, False)

    # Subconjunto de treino usado apenas na busca de topologia (mais rapido).
    rng = np.random.default_rng(SEED)
    sub_idx = rng.choice(len(splits["x_train"]), size=SEARCH_SUBSET, replace=False)
    search_loader = make_loader(
        splits["x_train"][sub_idx], splits["y_train"][sub_idx], BATCH_SIZE, True
    )

    search_rows = []
    final_rows = []

    for latent_dim in LATENT_DIMS:
        print(f"\n=== Espaco latente {latent_dim}D ===")
        best = None

        for hidden in TOPOLOGIES:
            set_seed(SEED)
            model = VAE(latent_dim=latent_dim, hidden=hidden)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\nCandidato hidden={hidden} ({n_params} params)")
            info = train_model(
                model, search_loader, val_loader, device,
                epochs=SEARCH_EPOCHS, lr=LR, beta=BETA, patience=3, verbose=True,
            )
            search_rows.append({
                "latent_dim": latent_dim,
                "hidden": "-".join(map(str, hidden)),
                "params": n_params,
                "best_val_loss": round(info["best_val"], 4),
                "best_epoch": info["best_epoch"],
                "epochs_run": info["epochs_run"],
            })
            if best is None or info["best_val"] < best["val"]:
                best = {"hidden": hidden, "val": info["best_val"]}

        print(f"\nMelhor topologia {latent_dim}D: {best['hidden']} "
              f"(val loss {best['val']:.4f}). Re-treinando ...")

        set_seed(SEED)
        model = VAE(latent_dim=latent_dim, hidden=best["hidden"])
        info = train_model(
            model, train_loader, val_loader, device,
            epochs=FINAL_EPOCHS, lr=LR, beta=BETA, patience=6, verbose=True,
        )

        test_loss, test_rec, test_kl = evaluate(model, test_loader, device, BETA)
        print(f"Teste {latent_dim}D: loss {test_loss:.3f} "
              f"(rec {test_rec:.3f}, kl {test_kl:.3f})")

        ckpt = RESULTS_DIR / f"vae_latent{latent_dim}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "latent_dim": latent_dim,
            "hidden": list(best["hidden"]),
        }, ckpt)

        hist_path = RESULTS_DIR / f"vae_latent{latent_dim}_history.csv"
        write_csv(hist_path, info["history"], list(info["history"][0].keys()))

        final_rows.append({
            "latent_dim": latent_dim,
            "hidden": "-".join(map(str, best["hidden"])),
            "best_val_loss": round(info["best_val"], 4),
            "best_epoch": info["best_epoch"],
            "epochs_run": info["epochs_run"],
            "test_loss": round(test_loss, 4),
            "test_rec": round(test_rec, 4),
            "test_kl": round(test_kl, 4),
        })

    write_csv(RESULTS_DIR / "search.csv", search_rows, list(search_rows[0].keys()))
    write_csv(RESULTS_DIR / "final_models.csv", final_rows, list(final_rows[0].keys()))

    with (RESULTS_DIR / "config.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "seed": SEED, "batch_size": BATCH_SIZE,
            "search_epochs": SEARCH_EPOCHS, "final_epochs": FINAL_EPOCHS,
            "search_subset": SEARCH_SUBSET,
            "beta": BETA, "lr": LR,
            "topologies": [list(t) for t in TOPOLOGIES],
            "latent_dims": list(LATENT_DIMS),
        }, fh, indent=2)

    print("\nConcluido. Resultados em results/.")


if __name__ == "__main__":
    main()
