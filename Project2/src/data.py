
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(image_path: str | Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Load an image and return normalized RGB pixels and image dimensions."""
    image_path = Path(image_path)

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        pixels = np.asarray(rgb_image, dtype=np.float64) / 255.0
        height, width = pixels.shape[:2]

    return pixels.reshape(-1, 3), (height, width)


def reconstruct_image(
    pixels: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Reconstruct an RGB image from normalized pixel vectors."""
    height, width = image_shape
    pixels = np.asarray(pixels, dtype=np.float64)

    if pixels.shape != (height * width, 3):
        raise ValueError(
            f"Expected shape {(height * width, 3)}, got {pixels.shape}."
        )

    pixels = np.clip(pixels, 0.0, 1.0)
    pixels = np.rint(pixels * 255).astype(np.uint8)

    return pixels.reshape(height, width, 3)


def save_image(
    pixels: np.ndarray,
    image_shape: tuple[int, int],
    output_path: str | Path,
) -> None:
    """Save normalized RGB pixel vectors as an image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reconstructed = reconstruct_image(pixels, image_shape)
    Image.fromarray(reconstructed, mode="RGB").save(output_path)
