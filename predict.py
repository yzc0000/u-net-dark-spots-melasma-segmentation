"""Run U-Net inference and save a binary mask plus a red overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from checkpoint import DEFAULT_WEIGHTS, load_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def create_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    highlighted = base.copy()
    highlighted[mask] = (255, 45, 45)
    return Image.fromarray((0.65 * base + 0.35 * highlighted).astype(np.uint8))


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    threshold: float,
) -> tuple[Image.Image, Image.Image]:
    original = image.convert("RGB")
    resized = original.resize((image_size, image_size), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))

    mask = probabilities.squeeze().cpu().numpy() >= threshold
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    mask_image = mask_image.resize(original.size, Image.Resampling.NEAREST)
    overlay = create_overlay(original, np.asarray(mask_image) > 0)
    return mask_image, overlay


def find_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="An image or directory of images")
    parser.add_argument("--output-dir", default="outputs", help="Directory for masks and overlays")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1")

    images = find_images(Path(args.input))
    if not images:
        raise FileNotFoundError(f"No supported images found at {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint, device = load_model(args.weights)
    image_size = int(checkpoint.get("img_size", 640))

    for image_path in images:
        with Image.open(image_path) as source:
            mask, overlay = predict_image(
                source, model, device, image_size=image_size, threshold=args.threshold
            )
        mask.save(output_dir / f"{image_path.stem}_mask.png")
        overlay.save(output_dir / f"{image_path.stem}_overlay.png")
        print(f"Processed {image_path.name}")


if __name__ == "__main__":
    main()

