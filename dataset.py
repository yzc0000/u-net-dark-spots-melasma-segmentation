"""Paired image-and-mask dataset used by training and evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, images_dir: Path, masks_dir: Path, image_size: int, augment: bool = False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.augment = augment
        self.items = self._collect_items()

    def _collect_items(self) -> list[tuple[Path, Path]]:
        if not self.images_dir.is_dir() or not self.masks_dir.is_dir():
            raise FileNotFoundError(
                f"Expected image and mask directories: {self.images_dir}, {self.masks_dir}"
            )

        items: list[tuple[Path, Path]] = []
        for image_path in sorted(self.images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = self.masks_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
            items.append((image_path, mask_path))

        if not items:
            raise FileNotFoundError(f"No supported images found in {self.images_dir}")
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.items[index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        with Image.open(mask_path) as source_mask:
            mask = source_mask.convert("L")

        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        image_tensor = TF.to_tensor(image)
        mask_array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor

