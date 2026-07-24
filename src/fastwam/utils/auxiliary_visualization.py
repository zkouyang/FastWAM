"""Compact validation visualizations for offline auxiliary supervision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


def _gray_image(value: torch.Tensor) -> Image.Image:
    value = value.detach().float().cpu()
    finite = torch.isfinite(value)
    if finite.any():
        low = value[finite].quantile(0.01)
        high = value[finite].quantile(0.99)
        value = (value - low) / (high - low).clamp(min=1e-6)
    value = value.nan_to_num().clamp(0, 1)
    return Image.fromarray((value.numpy() * 255).astype(np.uint8), mode="L").convert("RGB")


def _pair(left: Image.Image, right: Image.Image) -> Image.Image:
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + right.width, height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def _draw_boxes(boxes: torch.Tensor, size: tuple[int, int], color: str) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for cx, cy, bw, bh in boxes.detach().float().cpu().tolist():
        draw.rectangle(
            (
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ),
            outline=color,
            width=2,
        )
    return image


def _draw_tracks(
    coords: torch.Tensor,
    visibility: torch.Tensor,
    point_valid: torch.Tensor,
    *,
    size: tuple[int, int] = (256, 256),
) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    palette = ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00")
    coords = coords.detach().float().cpu()
    visibility = visibility.detach().float().cpu()
    point_valid = point_valid.detach().bool().cpu()
    for index in torch.where(point_valid)[0].tolist():
        visible = visibility[index] > 0.5
        points = [
            (float(x) * width, float(y) * height)
            for (x, y), keep in zip(coords[index].tolist(), visible.tolist(), strict=True)
            if keep
        ]
        if len(points) >= 2:
            draw.line(points, fill=palette[index % len(palette)], width=2)
        for point in points:
            draw.ellipse(
                (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2),
                fill=palette[index % len(palette)],
            )
    return image


def save_auxiliary_visualizations(
    outputs: dict[str, Any],
    output_dir: str | Path,
    *,
    prefix: str,
) -> dict[str, str]:
    """Save first-sample/first-frame prediction-vs-target PNGs.

    The left half is prediction and the right half is ground truth.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if "depth" in outputs:
        item = outputs["depth"]
        image = _pair(
            _gray_image(item["prediction"][0, 0, 0]),
            _gray_image(item["target"][0, 0, 0]),
        )
        path = root / f"{prefix}_depth.png"
        image.save(path)
        paths["depth"] = str(path)

    if "bbox" in outputs:
        item = outputs["bbox"]
        prediction = item["prediction"]
        scores = prediction["pred_logits"][0, 0].sigmoid().amax(dim=-1)
        keep = scores > 0.5
        if not keep.any():
            keep[scores.argmax()] = True
        image = _pair(
            _draw_boxes(prediction["pred_boxes"][0, 0, keep], (256, 256), "red"),
            _draw_boxes(item["target"][0][0], (256, 256), "green"),
        )
        path = root / f"{prefix}_bbox.png"
        image.save(path)
        paths["bbox"] = str(path)

    if "mask" in outputs:
        item = outputs["mask"]
        pred = item["prediction"][0, 0].sigmoid().amax(dim=0)
        target_masks = item["target"][0][0]
        target = (
            target_masks.amax(dim=0)
            if len(target_masks)
            else torch.zeros_like(pred, device=target_masks.device)
        )
        image = _pair(_gray_image(pred), _gray_image(target))
        path = root / f"{prefix}_mask.png"
        image.save(path)
        paths["mask"] = str(path)

    if "trajectory" in outputs:
        item = outputs["trajectory"]
        pred = item["prediction"]
        target = item["target"]
        image = _pair(
            _draw_tracks(
                pred["pred_coords"][0],
                pred["pred_visibility"][0].sigmoid(),
                target["point_valid"][0],
            ),
            _draw_tracks(
                target["coords"][0],
                target["visibility"][0],
                target["point_valid"][0],
            ),
        )
        path = root / f"{prefix}_trajectory.png"
        image.save(path)
        paths["trajectory"] = str(path)

    return paths
