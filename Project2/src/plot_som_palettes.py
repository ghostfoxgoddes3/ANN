from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "palettes"

IMAGES = ("berlim", "canudos", "debret1")
GRID_SIZES = (4, 8, 16)
SEEDS = (42, 43, 44)


def save_palette(weights_path: Path, output_dir: Path) -> None:
    weights = np.load(weights_path)

    side = int(round(np.sqrt(len(weights))))
    if weights.shape != (side * side, 3) or side not in GRID_SIZES:
        raise ValueError(f"Unexpected weight shape in {weights_path}: {weights.shape}")

    # The SOM indexes neurons row by row, matching this reshape.
    grid = np.clip(weights.reshape(side, side, 3), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, interpolation="nearest")
    ax.set_title(f"SOM color map — {weights_path.stem}")
    ax.set_xticks(np.arange(-0.5, side, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, side, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.65)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    fig.tight_layout()
    output_path = output_dir / f"{weights_path.stem}_palette.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Also save the exact RGB values of the neurons as a text palette.
    colors = np.rint(grid.reshape(-1, 3) * 255).astype(np.uint8)
    txt_path = output_dir / f"{weights_path.stem}_colors.txt"
    with txt_path.open("w", encoding="utf-8") as file:
        for index, rgb in enumerate(colors):
            file.write(
                f"{index:3d}: RGB({rgb[0]:3d}, {rgb[1]:3d}, {rgb[2]:3d}) "
                f"{to_hex(rgb / 255)}\n"
            )

    print(f"Saved {output_path.relative_to(ROOT)}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    for image in IMAGES:
        for side in GRID_SIZES:
            for seed in SEEDS:
                weights_path = RESULTS_DIR / (
                    f"{image}_som_{side}x{side}_seed{seed}_weights.npy"
                )
                if not weights_path.is_file():
                    print(f"Missing: {weights_path.relative_to(ROOT)}")
                    continue

                save_palette(weights_path, OUTPUT_DIR)
                count += 1

    print(f"\nGenerated {count} palettes in {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
