"""Safe checkpoint loading helpers for the released U-Net weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from unet_model import UNet


DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "u-net-dark-spots-melasma.pt"


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a tensor-only checkpoint without allowing arbitrary pickle objects."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Expected a checkpoint containing a 'model_state' mapping")
    return checkpoint


def load_model(
    path: str | Path = DEFAULT_WEIGHTS,
    device: str | torch.device | None = None,
) -> tuple[UNet, dict[str, Any], torch.device]:
    """Construct the model, restore its parameters, and switch to evaluation mode."""
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = load_checkpoint(path)
    base_channels = int(checkpoint.get("base_channels", 64))
    model = UNet(in_channels=3, out_channels=1, base_channels=base_channels, bilinear=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(resolved_device).eval()
    return model, checkpoint, resolved_device

