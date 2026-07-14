#!/usr/bin/env python3
"""Render visualizations directly from existing LIBERO bbox cache files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from merge_libero_bbox_camera import unpack_cameras
from preprocess_libero_bbox import (
    episode_video_path,
    load_video_rgb,
    make_bbox_vis_frames,
    slugify,
    write_rgb_video_or_contact_sheet,
)


def render_cache(
    cache_path: Path,
    *,
    data_root: Path,
    vis_root: Path,
    camera_key: str,
    fps: int,
    max_frames: int,
) -> Path:
    with np.load(cache_path, allow_pickle=False) as cache:
        meta = json.loads(str(cache["meta_json"]))
        frame_indices = np.asarray(cache["frame_indices"], dtype=np.int32)
        camera_keys = [str(value) for value in cache["camera_keys"].tolist()]
        records = unpack_cameras(cache)

    if camera_key not in records:
        raise KeyError(f"Camera {camera_key!r} not found in {cache_path}")
    camera_index = camera_keys.index(camera_key)
    suite = str(meta["suite"])
    task = str(meta["task"])
    episode_index = int(meta["episode_index"])
    image_size = int(meta["image_size"])
    video_path = episode_video_path(data_root / suite, camera_key, episode_index)
    frames = load_video_rgb(video_path, resize=image_size)
    if len(frames) <= int(frame_indices[-1]):
        raise ValueError(
            f"Video has {len(frames)} frames but cache references frame {int(frame_indices[-1])}"
        )

    if max_frames > 0 and len(frame_indices) > max_frames:
        keep = np.linspace(0, len(frame_indices) - 1, max_frames, dtype=np.int32)
    else:
        keep = np.arange(len(frame_indices), dtype=np.int32)
    record = records[camera_key]
    output_dir = (
        vis_root
        / suite
        / slugify(task)
        / f"episode_{episode_index:06d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task.txt").write_text(task + "\n", encoding="utf-8")
    camera_dir = output_dir / f"camera_{camera_index:02d}_{slugify(camera_key)}"
    output_path = camera_dir / (
        "grounded_sam2_mask_map.mp4" if record["masks"] is not None else "bbox_map.mp4"
    )
    rendered = make_bbox_vis_frames(
        frames,
        record["boxes"],
        record["confidences"],
        record["labels"],
        frame_indices,
        keep,
        record["masks"],
    )
    write_rgb_video_or_contact_sheet(rendered, output_path, fps)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_paths", nargs="+", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/libero_mujoco3.3.2"))
    parser.add_argument(
        "--vis-root",
        type=Path,
        default=Path("data/libero_mujoco3.3.2_bbox_cache/visualizations"),
    )
    parser.add_argument("--camera-key", default="observation.images.image")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=48)
    args = parser.parse_args()

    for cache_path in args.cache_paths:
        output_path = render_cache(
            cache_path,
            data_root=args.data_root,
            vis_root=args.vis_root,
            camera_key=args.camera_key,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        print(f"rendered {cache_path} -> {output_path}")


if __name__ == "__main__":
    main()
