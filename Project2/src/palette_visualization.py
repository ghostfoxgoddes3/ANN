from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from src.data import load_image
from src.gng import GNG
from src.som import SOM

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "figures" / "resized"
OUTPUT_DIR = ROOT / "results" / "palettes"

IMAGES = ("berlim.png", "canudos.png", "debret1.png")
PALETTE_SIZES = (4, 8, 16, 32)
SEED = 42
TRAINING_SAMPLE_SIZE = 5000


def make_palette_image(pixels, prototypes, height, width, n_colors):
    """Cluster learned prototypes into a smaller palette and reconstruct."""
    if len(prototypes) < n_colors:
        raise ValueError(
            f"Only {len(prototypes)} prototypes available for {n_colors} colors."
        )

    clustering = KMeans(
        n_clusters=n_colors,
        random_state=SEED,
        n_init=10,
    )
    clustering.fit(prototypes)
    palette = clustering.cluster_centers_

    # Assign each pixel to its nearest color in the reduced palette.
    assignments = np.empty(len(pixels), dtype=np.int32)
    batch_size = 8192
    palette_norms = np.sum(palette**2, axis=1)

    for start in range(0, len(pixels), batch_size):
        batch = pixels[start:start + batch_size]
        distances = (
            np.sum(batch**2, axis=1)[:, None]
            + palette_norms[None, :]
            - 2 * batch @ palette.T
        )
        assignments[start:start + batch_size] = np.argmin(distances, axis=1)

    reconstructed = np.clip(palette[assignments], 0, 1)
    image = np.rint(reconstructed.reshape(height, width, 3) * 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_index, image_name in enumerate(IMAGES):
        image_path = IMAGE_DIR / image_name
        pixels, (height, width) = load_image(image_path)

        rng = np.random.default_rng(SEED)
        sample = pixels[
            rng.choice(
                len(pixels),
                size=min(TRAINING_SAMPLE_SIZE, len(pixels)),
                replace=False,
            )
        ]

        print(f"\n{image_name}: training models...", flush=True)

        models = []

        for side in (4, 8, 16):
            model = SOM(side, side, seed=SEED)
            model.fit(sample, epochs=5)
            models.append((f"SOM_{side}x{side}", model.weights))

        for max_neurons in (16, 64, 256):
            model = GNG(max_neurons=max_neurons, seed=SEED)
            model.fit(sample, n_steps=max(2000, max_neurons * 40))
            models.append((f"GNG_{max_neurons}", model.weights))

        # Combine prototypes from the trained configurations. Each method's
        # palette is still generated independently from its own prototypes.
        for model_name, prototypes in models:
            if prototypes is None:
                continue

            for n_colors in PALETTE_SIZES:
                if len(prototypes) < n_colors:
                    print(
                        f"  Skipping {model_name}, {n_colors} colors: "
                        f"only {len(prototypes)} prototypes."
                    )
                    continue

                result = make_palette_image(
                    pixels, prototypes, height, width, n_colors
                )
                output_path = (
                    OUTPUT_DIR
                    / f"{Path(image_name).stem}_{model_name}_{n_colors}colors.png"
                )
                result.save(output_path)

            print(f"  Finished {model_name}", flush=True)

    print(f"\nNew palette images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
