"""Create a four-panel original/ground-truth/prediction/overlay comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from checkpoint import DEFAULT_WEIGHTS, load_model
from predict import predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--panel-size", type=int, default=512)
    return parser.parse_args()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    args = parse_args()
    model, checkpoint, device = load_model(args.weights)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    with Image.open(args.mask) as source:
        ground_truth = source.convert("L").resize(image.size, Image.Resampling.NEAREST)

    prediction, overlay = predict_image(
        image,
        model,
        device,
        image_size=int(checkpoint.get("img_size", 640)),
        threshold=args.threshold,
    )

    expected = np.asarray(ground_truth) > 127
    predicted = np.asarray(prediction) > 127
    intersection = np.logical_and(expected, predicted).sum()
    union = np.logical_or(expected, predicted).sum()
    total = expected.sum() + predicted.sum()
    iou = intersection / union if union else 1.0
    dice = 2 * intersection / total if total else 1.0

    size = args.panel_size
    labels = ("Original", "Ground truth", "Prediction", "Prediction overlay")
    panels = (image, ground_truth.convert("RGB"), prediction.convert("RGB"), overlay)
    header_height = 58
    footer_height = 52
    gap = 4
    canvas = Image.new(
        "RGB",
        (size * 4 + gap * 3, header_height + size + footer_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    label_font = font(22)
    metric_font = font(20)

    for index, (label, panel) in enumerate(zip(labels, panels)):
        x = index * (size + gap)
        fitted = panel.resize((size, size), Image.Resampling.BILINEAR)
        if label in {"Ground truth", "Prediction"}:
            fitted = panel.resize((size, size), Image.Resampling.NEAREST)
        canvas.paste(fitted, (x, header_height))
        label_box = draw.textbbox((0, 0), label, font=label_font)
        label_width = label_box[2] - label_box[0]
        draw.text((x + (size - label_width) / 2, 16), label, fill="black", font=label_font)

    metrics = f"Threshold {args.threshold:.2f}   |   IoU {iou:.4f}   |   Dice {dice:.4f}"
    metrics_box = draw.textbbox((0, 0), metrics, font=metric_font)
    metrics_width = metrics_box[2] - metrics_box[0]
    draw.text(
        ((canvas.width - metrics_width) / 2, header_height + size + 14),
        metrics,
        fill="black",
        font=metric_font,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    print(f"Saved {output} (IoU={iou:.4f}, Dice={dice:.4f})")


if __name__ == "__main__":
    main()

