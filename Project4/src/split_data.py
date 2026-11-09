"""
Partição dos dados em treino/validação/teste segundo duas estratégias.

Esta etapa é o centro conceitual do projeto. Em séries temporais, a forma de
particionar os dados muda a validade da avaliação:

  - CRONOLÓGICO (principal): treina no passado, valida no período intermediário
    e testa no futuro. Simula o uso real do modelo e é a avaliação honesta.

  - ALEATÓRIO (didático): embaralha as janelas e sorteia os conjuntos. Como
    janelas vizinhas se sobrepõem no tempo, janelas de validação/teste
    compartilham instantes com janelas de treino -> VAZAMENTO TEMPORAL. A
    métrica resultante tende a ser otimista e irreal.

Mantemos os dois para medir, de forma controlada, o quanto o vazamento infla
o desempenho aparente. No split cronológico aplicamos ainda um GAP (embargo)
entre os blocos, descartando as janelas da fronteira que se sobreporiam.

Uso:
    python -m src.split_data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# o restante (0.15) é teste

# Importado de data.py para dimensionar o embargo entre blocos cronológicos.
from src.data import WINDOW, HORIZON

GAP = WINDOW + HORIZON  # janelas descartadas na fronteira de cada bloco


def chronological_split(n: int) -> dict[str, np.ndarray]:
    """Blocos contíguos no tempo, com embargo (gap) entre eles."""
    i_train_end = int(n * TRAIN_FRAC)
    i_val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    train = np.arange(0, i_train_end - GAP)
    val = np.arange(i_train_end + GAP, i_val_end - GAP)
    test = np.arange(i_val_end + GAP, n)
    return {"train": train, "val": val, "test": test}


def random_split(n: int) -> dict[str, np.ndarray]:
    """Embaralhamento global das janelas (gera vazamento temporal)."""
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)

    i_train_end = int(n * TRAIN_FRAC)
    i_val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    return {
        "train": np.sort(perm[:i_train_end]),
        "val": np.sort(perm[i_train_end:i_val_end]),
        "test": np.sort(perm[i_val_end:]),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(DATA_DIR / "windows.npz", allow_pickle=True)
    n = len(data["y"])

    splits = {
        "chronological": chronological_split(n),
        "random": random_split(n),
    }

    lines = ["SPLIT SUMMARY", "=============", f"Total de janelas: {n}",
             f"Seed: {SEED} | gap cronológico: {GAP} janelas", ""]

    for name, idx in splits.items():
        np.savez(
            RESULTS_DIR / f"split_{name}.npz",
            train=idx["train"], val=idx["val"], test=idx["test"],
        )
        y = data["y"]
        lines.append(f"[{name}]")
        for subset in ("train", "val", "test"):
            sel = idx[subset]
            pos = float(y[sel].mean()) if len(sel) else float("nan")
            lines.append(
                f"  {subset:5s}: {len(sel):6d} janelas | chuva={pos:.4f}"
            )
        lines.append("")

    summary = "\n".join(lines)
    (RESULTS_DIR / "split_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
