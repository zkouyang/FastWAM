"""Compact validation visualizations for offline auxiliary supervision."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw

from .video_io import save_mp4


DISPLAY_SIZE = (224, 224)
HEADER_HEIGHT = 32


def _gray_image(
    value: torch.Tensor,
    *,
    size: tuple[int, int] = DISPLAY_SIZE,
    value_range: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> Image.Image:
    value = value.detach().float().cpu()
    finite = torch.isfinite(value)
    if value_range is None:
        if finite.any():
            low = value[finite].quantile(0.01)
            high = value[finite].quantile(0.99)
        else:
            low = value.new_tensor(0.0)
            high = value.new_tensor(1.0)
    else:
        low, high = value_range
    value = (value - low) / (high - low).clamp(min=1e-6)
    value = value.nan_to_num().clamp(0, 1)
    image = Image.fromarray((value.numpy() * 255).astype(np.uint8), mode="L").convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image


def _depth_range(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = target.detach().float().cpu()
    finite = torch.isfinite(target)
    if not finite.any():
        return target.new_tensor(0.0), target.new_tensor(1.0)
    return target[finite].quantile(0.01), target[finite].quantile(0.99)


def _pair(left: Image.Image, right: Image.Image) -> Image.Image:
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + right.width, height + HEADER_HEIGHT), "white")
    canvas.paste(left, (0, HEADER_HEIGHT))
    canvas.paste(right, (left.width, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 9), "Label", fill="black")
    draw.text((left.width + 4, 9), "Inference decoding", fill="black")
    return canvas


def _rgb_frame(
    rgb_video: Optional[torch.Tensor],
    frame_index: int,
    *,
    size: tuple[int, int] = DISPLAY_SIZE,
) -> Image.Image:
    if rgb_video is None:
        return Image.new("RGB", size, "white")
    if rgb_video.ndim != 4 or rgb_video.shape[0] != 3:
        raise ValueError(f"rgb_video must be [3,T,H,W], got {tuple(rgb_video.shape)}")
    frame_index = min(max(int(frame_index), 0), rgb_video.shape[1] - 1)
    frame = rgb_video[:, frame_index].detach().float().cpu()
    frame = ((frame.clamp(-1, 1) + 1) * 0.5).permute(1, 2, 0).numpy()
    image = Image.fromarray((frame * 255).astype(np.uint8), mode="RGB")

    # LIBERO concatenates [agent-view, wrist-view] horizontally while the
    # auxiliary labels are agent-view-only. Crop the first matching view.
    target_aspect = size[0] / size[1]
    crop_width = min(image.width, max(1, round(image.height * target_aspect)))
    if crop_width < image.width:
        image = image.crop((0, 0, crop_width, image.height))
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image


def _draw_boxes(boxes: torch.Tensor, base: Image.Image, color: str) -> Image.Image:
    image = base.copy()
    width, height = image.size
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


def _overlay_mask(base: Image.Image, mask: torch.Tensor, color: str) -> Image.Image:
    mask = mask.detach().float().cpu().nan_to_num().clamp(0, 1)
    mask_image = Image.fromarray((mask.numpy() * 255).astype(np.uint8), mode="L")
    if mask_image.size != base.size:
        mask_image = mask_image.resize(base.size, Image.Resampling.BILINEAR)
    alpha = np.asarray(mask_image, dtype=np.float32)[..., None] / 255.0 * 0.55
    base_array = np.asarray(base.convert("RGB"), dtype=np.float32)
    color_array = np.asarray(ImageColor.getrgb(color), dtype=np.float32)
    blended = base_array * (1.0 - alpha) + color_array * alpha
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8), mode="RGB")


def _draw_tracks(
    coords: torch.Tensor,
    visibility: torch.Tensor,
    point_valid: torch.Tensor,
    *,
    base: Image.Image,
) -> Image.Image:
    width, height = base.size
    image = base.copy()
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


def _prediction_boxes(prediction: dict[str, torch.Tensor], frame_index: int) -> torch.Tensor:
    scores = prediction["pred_logits"][0, frame_index].sigmoid().amax(dim=-1)
    return prediction["pred_boxes"][0, frame_index, scores > 0.5]


def _depth_pair(item: dict[str, Any], frame_index: int) -> Image.Image:
    target = item["target"][0, frame_index, 0]
    prediction = item["prediction"][0, frame_index, 0]
    value_range = _depth_range(target)
    return _pair(
        _gray_image(target, value_range=value_range),
        _gray_image(prediction, value_range=value_range),
    )


def _bbox_pair(
    item: dict[str, Any], rgb_video: Optional[torch.Tensor], frame_index: int
) -> Image.Image:
    base = _rgb_frame(rgb_video, frame_index)
    return _pair(
        _draw_boxes(item["target"][0][frame_index], base, "#00ff00"),
        _draw_boxes(_prediction_boxes(item["prediction"], frame_index), base, "#ff0000"),
    )


def _mask_pair(
    item: dict[str, Any], rgb_video: Optional[torch.Tensor], frame_index: int
) -> Image.Image:
    pred = item["prediction"][0, frame_index].sigmoid().amax(dim=0)
    target_masks = item["target"][0][frame_index]
    target = (
        target_masks.amax(dim=0)
        if len(target_masks)
        else torch.zeros_like(pred, device=target_masks.device)
    )
    base = _rgb_frame(rgb_video, frame_index)
    return _pair(
        _overlay_mask(base, target, "#00ff00"),
        _overlay_mask(base, pred, "#ff0000"),
    )


def save_auxiliary_visualizations(
    outputs: dict[str, Any],
    output_dir: str | Path,
    *,
    prefix: str,
    rgb_video: Optional[torch.Tensor] = None,
    save_video: bool = False,
    video_fps: int = 8,
) -> dict[str, str]:
    """Save first-sample auxiliary label-vs-inference visualizations.

    Every image stores the label on the left and the corresponding module
    decoding on the right. Optional videos cover the complete depth, bbox,
    and mask horizon; the default remains a first-frame PNG only.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if "depth" in outputs:
        item = outputs["depth"]
        path = root / f"{prefix}_depth.png"
        _depth_pair(item, 0).save(path)
        paths["depth"] = str(path)
        if save_video:
            video_path = root / f"{prefix}_depth.mp4"
            frames = [_depth_pair(item, frame) for frame in range(item["prediction"].shape[1])]
            save_mp4(frames, str(video_path), fps=video_fps)
            paths["depth_video"] = str(video_path)

    if "bbox" in outputs:
        item = outputs["bbox"]
        path = root / f"{prefix}_bbox.png"
        _bbox_pair(item, rgb_video, 0).save(path)
        paths["bbox"] = str(path)
        if save_video:
            video_path = root / f"{prefix}_bbox.mp4"
            frames = [
                _bbox_pair(item, rgb_video, frame)
                for frame in range(item["prediction"]["pred_boxes"].shape[1])
            ]
            save_mp4(frames, str(video_path), fps=video_fps)
            paths["bbox_video"] = str(video_path)

    if "mask" in outputs:
        item = outputs["mask"]
        path = root / f"{prefix}_mask.png"
        _mask_pair(item, rgb_video, 0).save(path)
        paths["mask"] = str(path)
        if save_video:
            video_path = root / f"{prefix}_mask.mp4"
            frames = [_mask_pair(item, rgb_video, frame) for frame in range(item["prediction"].shape[1])]
            save_mp4(frames, str(video_path), fps=video_fps)
            paths["mask_video"] = str(video_path)

    if "trajectory" in outputs:
        item = outputs["trajectory"]
        pred = item["prediction"]
        target = item["target"]
        base = _rgb_frame(rgb_video, 0)
        image = _pair(
            _draw_tracks(
                target["coords"][0],
                target["visibility"][0],
                target["point_valid"][0],
                base=base,
            ),
            _draw_tracks(
                pred["pred_coords"][0],
                pred["pred_visibility"][0].sigmoid(),
                target["point_valid"][0],
                base=base,
            ),
        )
        path = root / f"{prefix}_trajectory.png"
        image.save(path)
        paths["trajectory"] = str(path)

    return paths
