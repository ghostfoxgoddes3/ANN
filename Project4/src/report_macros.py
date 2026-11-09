"""
Gera report/results_macros.tex a partir dos CSVs em results/.

Mantém os números do relatório sincronizados com os experimentos de fato
executados, evitando valores digitados à mão.

Uso:
    python -m src.report_macros
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    WINDOW, HORIZON, RAIN_THRESHOLD, FEATURE_COLUMNS, YEARS, STATION_CODE,
)
from src.split_data import GAP

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"

PT = {"chronological": "Cronol\\'ogica", "random": "Aleat\\'oria"}


def num(x, d=4):
    return f"{float(x):.{d}f}".replace(".", ",")


def load_npz_counts():
    data = np.load(ROOT / "data" / "windows.npz", allow_pickle=True)
    y = data["y"]
    rows = []
    for split in ("chronological", "random"):
        idx = np.load(RESULTS_DIR / f"split_{split}.npz")
        for subset in ("train", "val", "test"):
            sel = idx[subset]
            rows.append((split, subset, len(sel), float(y[sel].mean())))
    return len(y), float(y.mean()), rows


def split_rates_rows(rows):
    sub_pt = {"train": "Treino", "val": "Valida\\c c\\~ao", "test": "Teste"}
    out = []
    for split, subset, n, rate in rows:
        out.append(
            f"{PT[split]} & {sub_pt[subset]} & {n} & {num(rate)} \\\\"
        )
    return "\n".join(out)


def best_configs_rows():
    out = []
    for split in ("chronological", "random"):
        df = pd.read_csv(RESULTS_DIR / f"search_{split}.csv")
        r = df.sort_values("best_val_loss").iloc[0]
        out.append(
            f"{PT[split]} & {r['cell']} & {int(r['hidden_size'])} & "
            f"{int(r['num_layers'])} & {num(r['best_val_loss'])} & "
            f"{num(r['val_f1'])} \\\\"
        )
    return "\n".join(out)


def test_results_rows():
    df = pd.read_csv(RESULTS_DIR / "test_results.csv").set_index("split")
    out = []
    for split in ("chronological", "random"):
        r = df.loc[split]
        out.append(
            f"{PT[split]} & {r['cell']} & {num(r['accuracy'])} & "
            f"{num(r['precision'])} & {num(r['recall'])} & "
            f"{num(r['f1'])} & {num(r['auc'])} \\\\"
        )
    return "\n".join(out)


def split_comparison_rows():
    df = pd.read_csv(RESULTS_DIR / "split_comparison.csv")
    pt = {"accuracy": "Acur\\'acia", "precision": "Precis\\~ao",
          "recall": "Revoca\\c c\\~ao", "f1": "F1", "auc": "AUC"}
    out = []
    for _, r in df.iterrows():
        out.append(
            f"{pt[r['metric']]} & {num(r['chronological'])} & "
            f"{num(r['random'])} & {num(r['random_minus_chrono'])} \\\\"
        )
    return "\n".join(out)


def main() -> None:
    n_win, pos_rate, rows = load_npz_counts()
    raw = pd.read_parquet(
        ROOT / "data" / "raw" / f"{STATION_CODE}_{YEARS[0]}_{YEARS[-1]}.parquet"
    )

    macros = [
        f"\\newcommand{{\\WINDOW}}{{{WINDOW}}}",
        f"\\newcommand{{\\HORIZON}}{{{HORIZON}}}",
        f"\\newcommand{{\\GAP}}{{{GAP}}}",
        f"\\newcommand{{\\RAINTH}}{{{num(RAIN_THRESHOLD, 1)}}}",
        f"\\newcommand{{\\NREG}}{{{len(raw)}}}",
        f"\\newcommand{{\\NFEAT}}{{{len(FEATURE_COLUMNS)}}}",
        f"\\newcommand{{\\NWIN}}{{{n_win}}}",
        f"\\newcommand{{\\POSRATE}}{{{num(pos_rate)}}}",
        f"\\newcommand{{\\SPLITRATES}}{{%\n{split_rates_rows(rows)}\n}}",
        f"\\newcommand{{\\BESTCONFIGS}}{{%\n{best_configs_rows()}\n}}",
        f"\\newcommand{{\\TESTRESULTS}}{{%\n{test_results_rows()}\n}}",
        f"\\newcommand{{\\SPLITCOMPARISON}}{{%\n{split_comparison_rows()}\n}}",
    ]

    (REPORT_DIR / "results_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )
    print("report/results_macros.tex gerado.")


if __name__ == "__main__":
    main()
