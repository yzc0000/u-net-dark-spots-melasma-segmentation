"""Evaluate a checkpoint on a held-out paired image-and-mask split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from checkpoint import DEFAULT_WEIGHTS, load_model
from dataset import SegmentationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset root")
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1")

    model, checkpoint, device = load_model(args.weights)
    image_size = int(checkpoint.get("img_size", 640))
    root = Path(args.data)
    dataset = SegmentationDataset(
        root / "images" / args.split,
        root / "masks" / args.split,
        image_size=image_size,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    samples = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        predictions = (torch.sigmoid(model(images)) >= args.threshold).float()
        intersection = (predictions * masks).sum(dim=(1, 2, 3))
        predicted = predictions.sum(dim=(1, 2, 3))
        expected = masks.sum(dim=(1, 2, 3))
        union = predicted + expected - intersection

        totals["dice"] += ((2 * intersection + 1) / (predicted + expected + 1)).sum().item()
        totals["iou"] += torch.where(union > 0, intersection / union, torch.ones_like(union)).sum().item()
        totals["precision"] += torch.where(
            predicted > 0, intersection / predicted, torch.ones_like(predicted)
        ).sum().item()
        totals["recall"] += torch.where(
            expected > 0, intersection / expected, torch.ones_like(expected)
        ).sum().item()
        samples += images.size(0)

    metrics = {name: value / samples for name, value in totals.items()}
    denominator = metrics["precision"] + metrics["recall"]
    metrics["f1"] = (
        2 * metrics["precision"] * metrics["recall"] / denominator if denominator else 0.0
    )
    print(f"samples:   {samples}")
    for name in ("dice", "iou", "precision", "recall", "f1"):
        print(f"{name:10}{metrics[name]:.4f}")


if __name__ == "__main__":
    main()

