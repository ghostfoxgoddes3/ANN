
from pathlib import Path
import csv

import numpy as np

from src.data import load_image, save_image
from src.gng import GNG
from src.metrics import quantization_error, topological_error
from src.som import SOM


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
RESIZED_DIR = FIGURES_DIR / "resized"
RESULTS_DIR = ROOT / "results"

SELECTED_IMAGES = ("berlim.png", "canudos.png", "debret1.png")
TRAINING_SAMPLE_SIZE = 5000
SEEDS = (42, 43, 44)
GNG_STEPS = 60000
SOM_PRESENTATIONS = 60000


def evaluate(
    pixels: np.ndarray,
    prototypes: np.ndarray,
    assignments: np.ndarray,
    neighbors: list[set[int]],
) -> tuple[float, float]:
    """Calculate quantization and topological errors."""
    q_error = quantization_error(pixels, prototypes, assignments[:, 0])
    t_error = topological_error(assignments, neighbors)
    return q_error, t_error


def run_experiment(
    image_path: Path,
    seed: int,
    sample: np.ndarray,
    pixels: np.ndarray,
    image_shape: tuple[int, int, int],
) -> list[dict]:
    rows = []

    print(f"\nImage: {image_path.name} | seed={seed}")

    # SOM: compare 4x4, 8x8, and 16x16 grids.
    for side in (4, 8, 16):
        model = SOM(side, side, seed=seed)

        # Keep the existing SOM training implementation and distribute
        # the presentation budget over complete epochs.
        epochs = max(1, SOM_PRESENTATIONS // len(sample))
        model.fit(sample, epochs=epochs)

        assignments = model.predict_two(pixels)
        q_error, t_error = evaluate(
            pixels, model.weights, assignments,
            model.topological_neighbors(),
        )

        output_name = (
            f"{image_path.stem}_som_{side}x{side}_seed{seed}.png"
        )
        output_path = RESULTS_DIR / output_name
        save_image(model.weights[assignments[:, 0]], image_shape, output_path)

        weights_path = RESULTS_DIR / (
            f"{image_path.stem}_som_{side}x{side}_seed{seed}_weights.npy"
        )
        np.save(weights_path, model.weights)

        rows.append({
            "image": image_path.name,
            "seed": seed,
            "model": "SOM",
            "configuration": f"{side}x{side}",
            "neurons": len(model.weights),
            "quantization_error": q_error,
            "topological_error": t_error,
            "output": output_name,
        })
        print(
            f"SOM {side}x{side}: QE={q_error:.6f}, "
            f"TE={t_error:.6f}"
        )

    # GNG: maximum sizes of 16, 64, and 256 neurons.
    for max_neurons in (16, 64, 256):
        model = GNG(max_neurons=max_neurons, seed=seed)
        model.fit(sample, n_steps=GNG_STEPS)

        assignments = model.predict_two(pixels)
        q_error, t_error = evaluate(
            pixels, model.weights, assignments,
            model.topological_neighbors(),
        )

        output_name = (
            f"{image_path.stem}_gng_{max_neurons}_seed{seed}.png"
        )
        output_path = RESULTS_DIR / output_name
        save_image(model.weights[assignments[:, 0]], image_shape, output_path)

        rows.append({
            "image": image_path.name,
            "seed": seed,
            "model": "GNG",
            "configuration": f"max_{max_neurons}",
            "neurons": len(model.weights),
            "quantization_error": q_error,
            "topological_error": t_error,
            "output": output_name,
        })
        print(
            f"GNG max={max_neurons} (actual={len(model.weights)}): "
            f"QE={q_error:.6f}, TE={t_error:.6f}"
        )

    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not RESIZED_DIR.exists():
        raise FileNotFoundError(
            f"Resized image directory not found: {RESIZED_DIR}"
        )

    available = {
        path.name: path
        for path in RESIZED_DIR.glob("*.png")
        if path.is_file()
    }
    missing = [name for name in SELECTED_IMAGES if name not in available]
    if missing:
        raise FileNotFoundError(
            f"Missing selected images in {RESIZED_DIR}: {missing}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_pixels = {}
    image_shapes = {}
    samples = {}

    # Use the same sampled pixels for all configurations and seeds.
    # The sample itself is reproducible and independent of model RNGs.
    for name in SELECTED_IMAGES:
        pixels, shape = load_image(available[name])
        all_pixels[name] = pixels
        image_shapes[name] = shape

        sample_rng = np.random.default_rng(2026)
        sample_size = min(TRAINING_SAMPLE_SIZE, len(pixels))
        sample_indices = sample_rng.choice(
            len(pixels), size=sample_size, replace=False
        )
        samples[name] = pixels[sample_indices]

    all_results = []
    for name in SELECTED_IMAGES:
        print(
            f"\nTraining sample: {len(samples[name])} / "
            f"{len(all_pixels[name])} pixels"
        )
        for seed in SEEDS:
            all_results.extend(
                run_experiment(
                    available[name],
                    seed,
                    samples[name],
                    all_pixels[name],
                    image_shapes[name],
                )
            )

    run_fields = [
        "image", "seed", "model", "configuration", "neurons",
        "quantization_error", "topological_error", "output",
    ]
    runs_path = RESULTS_DIR / "metrics_runs.csv"
    write_csv(runs_path, all_results, run_fields)

    # Summarize metrics over seeds, preserving the actual neuron count
    # as mean and sample standard deviation.
    grouped = {}
    for row in all_results:
        key = (row["image"], row["model"], row["configuration"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (image, model, configuration), group in sorted(grouped.items()):
        qes = np.array(
            [r["quantization_error"] for r in group], dtype=float
        )
        tes = np.array(
            [r["topological_error"] for r in group], dtype=float
        )
        neurons = np.array([r["neurons"] for r in group], dtype=float)

        summary_rows.append({
            "image": image,
            "model": model,
            "configuration": configuration,
            "neurons_mean": neurons.mean(),
            "neurons_sd": neurons.std(ddof=1),
            "qe_mean": qes.mean(),
            "qe_sd": qes.std(ddof=1),
            "te_mean": tes.mean(),
            "te_sd": tes.std(ddof=1),
            "n_runs": len(group),
        })

    summary_fields = [
        "image", "model", "configuration", "neurons_mean", "neurons_sd",
        "qe_mean", "qe_sd", "te_mean", "te_sd", "n_runs",
    ]
    summary_path = RESULTS_DIR / "metrics_summary.csv"
    write_csv(summary_path, summary_rows, summary_fields)

    print(f"\nIndividual results: {runs_path}")
    print(f"Summary statistics: {summary_path}")


if __name__ == "__main__":
    main()
