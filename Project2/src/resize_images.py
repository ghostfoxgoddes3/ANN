
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "figures"
OUTPUT_DIR = SOURCE_DIR / "resized"
MAX_SIZE = (500, 500)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png"}
    images = [
        path for path in SOURCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]

    for path in sorted(images):
        with Image.open(path) as image:
            image = image.convert("RGB")
            original_size = image.size
            image.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)

            output_path = OUTPUT_DIR / f"{path.stem}.png"
            image.save(output_path, optimize=True)

        print(
            f"{path.name}: {original_size[0]}x{original_size[1]} "
            f"-> {image.width}x{image.height} | {output_path.name}"
        )

    print(f"\nImagens processadas: {len(images)}")
    print(f"Destino: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()