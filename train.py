"""Train the U-Net on paired facial images and binary pigmentation masks."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from unet_model import UNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    return 1 - ((2 * intersection + smooth) / (denominator + smooth)).mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets) + dice_loss(
        logits, targets
    )


@torch.inference_mode()
def validate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    samples = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total_loss += combined_loss(logits, masks).item() * images.size(0)
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        intersection = (predictions * masks).sum(dim=(1, 2, 3))
        denominator = predictions.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        total_dice += ((2 * intersection + 1) / (denominator + 1)).sum().item()
        samples += images.size(0)

    return total_loss / samples, total_dice / samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset root")
    parser.add_argument("--output-dir", default="runs/training", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_root = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_data = SegmentationDataset(
        data_root / "images" / "train",
        data_root / "masks" / "train",
        image_size=args.image_size,
        augment=True,
    )
    validation_data = SegmentationDataset(
        data_root / "images" / "valid",
        data_root / "masks" / "valid",
        image_size=args.image_size,
    )
    training_loader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(3, 1, base_channels=args.base_channels, bilinear=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in training_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = combined_loss(model(images), masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        validation_loss, validation_dice = validate(model, validation_loader, device)
        print(
            f"epoch={epoch:03d} train_loss={running_loss / len(training_data):.4f} "
            f"val_loss={validation_loss:.4f} val_dice={validation_dice:.4f}"
        )

        metadata = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "img_size": args.image_size,
            "base_channels": args.base_channels,
            "threshold": 0.5,
            "validation_loss": validation_loss,
            "validation_dice": validation_dice,
        }
        torch.save(metadata, output_dir / "last.pt")
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(metadata, output_dir / "best.pt")


if __name__ == "__main__":
    main()

