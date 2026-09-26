"""Rank a dataset split by per-image IoU to inspect successes and failures."""

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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--count", type=int, default=5, help="Rows to show at each extreme")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    model, checkpoint, device = load_model(args.weights)
    root = Path(args.data)
    dataset = SegmentationDataset(
        root / "images" / args.split,
        root / "masks" / args.split,
        image_size=int(checkpoint.get("img_size", 640)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results: list[tuple[float, float, str]] = []
    for index, (image, mask) in enumerate(loader):
        image, mask = image.to(device), mask.to(device)
        prediction = (torch.sigmoid(model(image)) >= args.threshold).float()
        intersection = (prediction * mask).sum()
        predicted = prediction.sum()
        expected = mask.sum()
        union = predicted + expected - intersection
        iou = float(intersection / union) if union > 0 else 1.0
        dice = float((2 * intersection + 1) / (predicted + expected + 1))
        results.append((iou, dice, dataset.items[index][0].name))

    results.sort(key=lambda row: row[0], reverse=True)
    count = max(1, min(args.count, len(results)))
    print("BEST")
    for iou, dice, name in results[:count]:
        print(f"IoU={iou:.4f} Dice={dice:.4f} {name}")
    print("WORST")
    for iou, dice, name in reversed(results[-count:]):
        print(f"IoU={iou:.4f} Dice={dice:.4f} {name}")


if __name__ == "__main__":
    main()

