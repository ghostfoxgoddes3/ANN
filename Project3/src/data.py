"""Carregamento do Fashion-MNIST sem dependencia de torchvision.

Os arquivos IDX sao baixados (e cacheados) a partir do espelho publico
hospedado no Google Cloud Storage. As imagens sao retornadas como tensores
float32 normalizados em [0, 1] e achatados em vetores de 784 dimensoes.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

BASE_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


def _download(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / FILES[name]
    if not dest.exists():
        url = f"{BASE_URL}/{FILES[name]}"
        print(f"Baixando {FILES[name]} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def _read_idx(path: Path) -> np.ndarray:
    """Le um arquivo no formato IDX (MNIST) comprimido em gzip."""
    with gzip.open(path, "rb") as fh:
        zero, zero2, dtype_code, ndim = struct.unpack(">BBBB", fh.read(4))
        assert zero == 0 and zero2 == 0, "cabecalho IDX invalido"
        shape = struct.unpack(f">{ndim}I", fh.read(4 * ndim))
        buffer = fh.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        return data.reshape(shape)


def load_fashion_mnist() -> dict[str, np.ndarray]:
    """Retorna dict com train/test images e labels como numpy arrays."""
    paths = {name: _download(name) for name in FILES}
    train_images = _read_idx(paths["train_images"]).astype(np.float32) / 255.0
    test_images = _read_idx(paths["test_images"]).astype(np.float32) / 255.0
    train_labels = _read_idx(paths["train_labels"]).astype(np.int64)
    test_labels = _read_idx(paths["test_labels"]).astype(np.int64)

    train_images = train_images.reshape(len(train_images), -1)
    test_images = test_images.reshape(len(test_images), -1)
    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }


def build_splits(seed: int = 42, val_fraction: float = 0.1):
    """Separa treino em treino/validacao de forma reprodutivel."""
    data = load_fashion_mnist()
    rng = np.random.default_rng(seed)

    n = len(data["train_images"])
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    splits = {
        "x_train": data["train_images"][train_idx],
        "y_train": data["train_labels"][train_idx],
        "x_val": data["train_images"][val_idx],
        "y_val": data["train_labels"][val_idx],
        "x_test": data["test_images"],
        "y_test": data["test_labels"],
    }
    return splits


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    tensors = TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).long(),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)
