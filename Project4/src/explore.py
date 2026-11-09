"""
Figuras de exploração: contexto climático (El Niño / ONI) e vazamento temporal.

Gera material para a discussão ecológico-social e para a comparação entre as
partições:

  1. Série do índice ONI sobre o período, destacando a fase de El Niño forte
     de 2023-2024, sobreposta à precipitação mensal da estação.
  2. Taxa mensal de chuva (classe positiva) ao longo do tempo, evidenciando a
     não estacionariedade que torna o split cronológico mais difícil -- e mais
     honesto -- que o aleatório.
  3. Comparação visual das métricas de teste entre as duas partições.

Uso:
    python -m src.explore
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_raw, load_oni, build_features, build_label

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def climate_context() -> None:
    """ONI vs precipitação mensal: o contexto El Niño do período."""
    raw = load_raw()
    oni = load_oni()
    df = build_features(raw, oni)

    df = df.set_index("timestamp")
    monthly_precip = df["precip"].resample("MS").sum()
    monthly_oni = df["oni"].resample("MS").mean()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(monthly_precip.index, monthly_precip.values, width=20,
            color="#4c72b0", alpha=0.7, label="Precipitação mensal (mm)")
    ax1.set_ylabel("Precipitação mensal (mm)", color="#4c72b0")
    ax1.set_xlabel("Tempo")

    ax2 = ax1.twinx()
    ax2.plot(monthly_oni.index, monthly_oni.values, color="#c44e52",
             marker="o", label="ONI")
    ax2.axhline(0.5, ls="--", color="gray", lw=1)
    ax2.axhline(-0.5, ls="--", color="gray", lw=1)
    ax2.fill_between(monthly_oni.index, 0.5, monthly_oni.values,
                     where=(monthly_oni.values >= 0.5), color="#c44e52",
                     alpha=0.15)
    ax2.set_ylabel("Índice ONI (°C)", color="#c44e52")

    ax1.set_title("Contexto climático: El Niño (ONI) e precipitação mensal")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "climate_context.png", dpi=200,
                bbox_inches="tight")
    plt.close()


def rain_nonstationarity() -> None:
    """Taxa mensal da classe 'chuva' -> não estacionariedade da série."""
    raw = load_raw()
    oni = load_oni()
    df = build_features(raw, oni)
    df = build_label(df).set_index("timestamp")

    monthly_rate = df["label"].resample("MS").mean()

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(monthly_rate.index, monthly_rate.values, marker="o",
            color="#55a868")
    ax.axhline(df["label"].mean(), ls="--", color="gray",
               label="média global")
    ax.set_title("Taxa mensal da classe 'vai chover' (não estacionariedade)")
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Proporção de janelas positivas")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig(FIGURES_DIR / "rain_nonstationarity.png", dpi=200,
                bbox_inches="tight")
    plt.close()


def split_metric_comparison() -> None:
    """Barras comparando as métricas de teste entre as duas partições."""
    path = RESULTS_DIR / "test_results.csv"
    if not path.exists():
        print("test_results.csv ausente; rode src.evaluate antes.")
        return

    df = pd.read_csv(path).set_index("split")
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(metrics))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, df.loc["chronological", metrics].values, width,
           label="Cronológico (honesto)", color="#4c72b0")
    ax.bar(x + width / 2, df.loc["random", metrics].values, width,
           label="Aleatório (com vazamento)", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Efeito do vazamento temporal nas métricas de teste")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(FIGURES_DIR / "split_metric_comparison.png", dpi=200,
                bbox_inches="tight")
    plt.close()


def main() -> None:
    climate_context()
    rain_nonstationarity()
    split_metric_comparison()
    print("Figuras de exploração salvas em figures/.")


if __name__ == "__main__":
    main()
