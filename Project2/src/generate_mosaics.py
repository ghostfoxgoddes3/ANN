    panels.append(original)
    labels.append("Original")

    for config in configs:
        recon = load_reconstruction(image, model, config)
        if recon is not None:
            panels.append(fit_image_from_image(recon, (panel_w, panel_h)))
            labels.append(f"{model.upper()} {config.replace('x', ' × ')}")

    if len(panels) == 1:
        print(f"[aviso] Nenhuma reconstrução disponível para {image}/{model}.")
        return

    width = len(panels) * panel_w + (len(panels) + 1) * gap
    height = panel_h + label_h + 2 * gap
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    label_font = font(18)

    for i, (panel, label) in enumerate(zip(panels, labels)):
        x = gap + i * (panel_w + gap)
        y = gap + label_h
        canvas.paste(panel, (x + (panel_w - panel.width) // 2,
                             y + (panel_h - panel.height) // 2))
        bbox = draw.textbbox((0, 0), label, font=label_font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (panel_w - tw) // 2, gap), label, fill=TEXT, font=label_font)

    output = RESULTS_DIR / f"mosaic_{image}_{model}.png"
    canvas.save(output)
    print(f"Salvo: {output.relative_to(ROOT)}")


def fit_image_from_image(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    image = image.copy()
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def make_som_palette_mosaic(image: str) -> None:
    """Render the learned RGB weights in their actual SOM grid arrangement."""
    tile = 240
    title_h, gap, margin = 38, 20, 18
    grids = []
    labels = []

    for side in SOM_SIDES:
        weights_path = RESULTS_DIR / (
            f"{image}_som_{side}x{side}_seed{SEED}_weights.npy"
        )
        if not weights_path.exists():
            print(
                f"[aviso] Pesos SOM não encontrados: {weights_path.name}. "
                "Adicione np.save(...) ao treinamento e rode src/main.py."
            )
            continue

        weights = np.load(weights_path)
        if weights.shape != (side * side, 3):
            print(f"[aviso] Formato inesperado em {weights_path.name}: {weights.shape}")
            continue

        # The model stores the grid as row-major vectors. Clamp for safe RGB display.
        colors = np.clip(weights.reshape(side, side, 3), 0.0, 1.0)
        colors = np.round(colors * 255).astype(np.uint8)
        palette = Image.fromarray(colors, mode="RGB").resize(
            (tile, tile), Image.Resampling.NEAREST
        )
        grids.append(palette)
        labels.append(f"SOM {side * side} — {side} × {side}")

    if not grids:
        return

    width = len(grids) * tile + (len(grids) + 1) * gap
    height = tile + title_h + 2 * gap
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font = font(17)

    for i, (grid, label) in enumerate(zip(grids, labels)):
        x = gap + i * (tile + gap)
        y = gap + title_h
        canvas.paste(grid, (x, y))
        draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=CELL_BORDER, width=1)
        bbox = draw.textbbox((0, 0), label, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (tile - tw) // 2, gap), label, fill=TEXT, font=title_font)

    output = RESULTS_DIR / f"palettes_{image}_som.png"
    canvas.save(output)
    print(f"Salvo: {output.relative_to(ROOT)}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for image in IMAGES:
        make_reconstruction_mosaic(
            image, "som", tuple(f"{side}x{side}" for side in SOM_SIDES)
        )
        make_reconstruction_mosaic(
            image, "gng", tuple(str(size) for size in GNG_SIZES)
        )
        make_som_palette_mosaic(image)


if __name__ == "__main__":
    main()
