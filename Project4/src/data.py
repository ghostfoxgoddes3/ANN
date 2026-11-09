"""
Aquisição e preparação dos dados do Projeto 4.

Fontes:
  - INMET (dados históricos horários, estação automática) -> variáveis
    meteorológicas de superfície.
  - NOAA/CPC Oceanic Niño Index (ONI) -> estado do El Niño / La Niña (ENOS),
    usado como variável exógena de contexto climático.

O módulo baixa os dados brutos, padroniza o formato do INMET (encoding
latin1, separador ";", decimal com vírgula, 8 linhas de metadados), constrói
um conjunto de variáveis por hora, deriva o rótulo binário "vai chover nas
próximas K horas" e monta janelas deslizantes prontas para a RNN.

Uso:
    python -m src.data
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT / "results"

# Estação e período. A701 = São Paulo (Mirante). O período cobre o forte
# El Niño de 2023-2024, o que torna a variável ONI informativa.
STATION_CODE = "A701"
STATION_NAME = "SAO PAULO - MIRANTE"
YEARS = (2022, 2023, 2024)

INMET_URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# Parâmetros da tarefa de classificação.
WINDOW = 24       # horas de histórico que a janela "enxerga"
HORIZON = 6       # K: prever chuva nas próximas K horas (t+1 .. t+K)
RAIN_THRESHOLD = 0.2  # mm acumulados no horizonte para rotular "chuva"

MISSING = -9999.0

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (Project4 ANN academic use)"}

# Mapeamento dos cabeçalhos longos do INMET para nomes curtos.
COLUMN_MAP = {
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": "precip",
    "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": "pressure",
    "RADIACAO GLOBAL (Kj/m²)": "radiation",
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "temp",
    "TEMPERATURA DO PONTO DE ORVALHO (°C)": "dew_point",
    "UMIDADE RELATIVA DO AR, HORARIA (%)": "humidity",
    "VENTO, DIREÇÃO HORARIA (gr) (° (gr))": "wind_dir",
    "VENTO, VELOCIDADE HORARIA (m/s)": "wind_speed",
}

# Variáveis meteorológicas usadas como entrada (antes de adicionar features
# cíclicas e o ONI).
METEO_FEATURES = [
    "precip",
    "pressure",
    "radiation",
    "temp",
    "dew_point",
    "humidity",
    "wind_dir",
    "wind_speed",
]


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def _download(url: str) -> bytes:
    response = requests.get(url, headers=HTTP_HEADERS, timeout=600)
    response.raise_for_status()
    return response.content


def _station_csv_from_zip(zip_bytes: bytes, year: int) -> pd.DataFrame:
    """Extrai o CSV da estação alvo de dentro do ZIP anual do INMET."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        target = None
        for name in zf.namelist():
            if f"_{STATION_CODE}_" in name and name.upper().endswith(".CSV"):
                target = name
                break

        if target is None:
            raise FileNotFoundError(
                f"Estação {STATION_CODE} não encontrada no ZIP de {year}."
            )

        with zf.open(target) as handle:
            raw = handle.read()

    return _parse_inmet_csv(raw)


def _parse_inmet_csv(raw: bytes) -> pd.DataFrame:
    """Lê um CSV horário do INMET (latin1, ';' , decimal ',', 8 metadados)."""
    df = pd.read_csv(
        io.BytesIO(raw),
        sep=";",
        skiprows=8,
        encoding="latin-1",
        decimal=",",
        na_values=["", " "],
    )

    # A última coluna costuma vir vazia por causa do ';' final.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Renomeia as colunas conhecidas (ignora as demais).
    rename = {}
    for original, short in COLUMN_MAP.items():
        for col in df.columns:
            if col.strip() == original:
                rename[col] = short
    df = df.rename(columns=rename)

    # Coluna de data (nome varia entre "Data" e "DATA (YYYY-MM-DD)").
    date_col = next(c for c in df.columns if c.strip().lower().startswith("data"))
    hour_col = next(c for c in df.columns if "hora" in c.strip().lower())

    hours = (
        df[hour_col].astype(str).str.extract(r"(\d{2})")[0].astype(int)
    )
    dates = pd.to_datetime(
        df[date_col].astype(str).str.replace("/", "-"),
        format="mixed",
        dayfirst=False,
    )
    df["timestamp"] = dates + pd.to_timedelta(hours, unit="h")

    keep = ["timestamp"] + [c for c in METEO_FEATURES if c in df.columns]
    df = df[keep].copy()

    # INMET usa -9999 para ausente em alguns anos.
    df = df.replace(MISSING, np.nan)

    return df


def load_raw(force: bool = False) -> pd.DataFrame:
    """Baixa (com cache) e concatena os anos da estação alvo."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{STATION_CODE}_{YEARS[0]}_{YEARS[-1]}.parquet"

    if cache.exists() and not force:
        return pd.read_parquet(cache)

    frames = []
    for year in YEARS:
        print(f"[INMET] baixando {year} ...")
        zip_bytes = _download(INMET_URL.format(year=year))
        frames.append(_station_csv_from_zip(zip_bytes, year))

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df.reset_index(drop=True)

    df.to_parquet(cache, index=False)
    print(f"[INMET] {len(df)} registros horários salvos em {cache.name}")
    return df


def load_oni() -> pd.DataFrame:
    """Baixa o índice ONI (anomalia de TSM, ENOS) por mês."""
    print("[NOAA] baixando índice ONI ...")
    text = _download(ONI_URL).decode("utf-8")
    df = pd.read_csv(io.StringIO(text), sep=r"\s+")

    seas_to_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    df["month"] = df["SEAS"].map(seas_to_month)
    df = df.rename(columns={"YR": "year", "ANOM": "oni"})
    return df[["year", "month", "oni"]]


# --------------------------------------------------------------------------- #
# Engenharia de variáveis
# --------------------------------------------------------------------------- #

def build_features(df: pd.DataFrame, oni: pd.DataFrame) -> pd.DataFrame:
    """Cria a grade horária contínua, trata faltantes e adiciona features."""
    df = df.set_index("timestamp").sort_index()

    # Grade horária regular (preenche buracos de tempo com NaN).
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_index)
    df.index.name = "timestamp"

    # Interpolação curta para lacunas pequenas; precipitação ausente -> 0.
    df["precip"] = df["precip"].fillna(0.0)
    df[METEO_FEATURES] = df[METEO_FEATURES].interpolate(limit=6).ffill().bfill()

    # Features cíclicas de tempo (hora do dia e dia do ano).
    hour = df.index.hour
    doy = df.index.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Mescla o ONI (contexto El Niño) por ano/mês.
    df["year"] = df.index.year
    df["month"] = df.index.month
    df = df.reset_index().merge(oni, on=["year", "month"], how="left")
    df["oni"] = df["oni"].ffill().bfill()
    df = df.drop(columns=["year", "month"])

    return df


FEATURE_COLUMNS = METEO_FEATURES + [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "oni",
]


def build_label(df: pd.DataFrame) -> pd.DataFrame:
    """Rótulo binário: chove acima do limiar nas próximas HORIZON horas."""
    # Soma da precipitação em t+1 .. t+HORIZON (olhando para o futuro).
    future = (
        df["precip"]
        .iloc[::-1]
        .rolling(window=HORIZON, min_periods=HORIZON)
        .sum()
        .iloc[::-1]
        .shift(-1)
    )
    df = df.copy()
    df["label"] = (future >= RAIN_THRESHOLD).astype("float32")
    df["future_precip"] = future
    # As últimas HORIZON linhas não têm horizonte completo.
    df = df.iloc[:-HORIZON].reset_index(drop=True)
    return df


def make_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera janelas deslizantes (N, WINDOW, F) e rótulos (N,).

    Retorna também os timestamps do instante t de cada janela, usados pelo
    split cronológico.
    """
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.float32)
    times = df["timestamp"].to_numpy()

    n = len(df) - WINDOW + 1
    x = np.stack([features[i:i + WINDOW] for i in range(n)])
    # Rótulo da janela = rótulo do último instante t da janela.
    y = labels[WINDOW - 1:WINDOW - 1 + n]
    t = times[WINDOW - 1:WINDOW - 1 + n]
    return x, y, t


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    oni = load_oni()

    df = build_features(raw, oni)
    df = build_label(df)

    x, y, t = make_windows(df)

    out = DATA_DIR / "windows.npz"
    np.savez_compressed(
        out,
        x=x,
        y=y,
        t=t.astype("datetime64[s]").astype(np.int64),
        feature_columns=np.array(FEATURE_COLUMNS),
    )

    positive = float(y.mean())
    summary = (
        f"Estação: {STATION_NAME} ({STATION_CODE})\n"
        f"Anos: {YEARS}\n"
        f"Janela (WINDOW): {WINDOW} h | Horizonte (HORIZON): {HORIZON} h\n"
        f"Limiar de chuva: {RAIN_THRESHOLD} mm\n"
        f"Total de janelas: {len(y)}\n"
        f"Proporção de chuva (classe 1): {positive:.4f}\n"
        f"Variáveis ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}\n"
    )
    (RESULTS_DIR / "dataset_summary.txt").write_text(summary, encoding="utf-8")

    print("\n" + summary)
    print(f"Janelas salvas em {out}")


if __name__ == "__main__":
    main()
