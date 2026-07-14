#!/usr/bin/env python3
"""Build per-suite contact sheets for reviewing LIBERO bbox/mask videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


AGENTVIEW_DIR = "camera_00_observation.images.image"
VIDEO_NAMES = ("grounded_sam2_mask_map.mp4", "bbox_map.mp4")


def read_sampled_frames(path: Path, count: int) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f"Video reports no frames: {path}")
        indices = np.linspace(0, frame_count - 1, min(count, frame_count), dtype=np.int32)
        sampled: list[tuple[int, np.ndarray]] = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {index} from {path}")
            sampled.append((int(index), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        return sampled
    finally:
        cap.release()


def task_label(task_dir: Path) -> str:
    task_file = next(task_dir.glob("episode_*/task.txt"), None)
    if task_file is not None:
        return task_file.read_text(encoding="utf-8").strip()
    return task_dir.name


def render_suite(video_paths: list[Path], output_path: Path, samples: int) -> None:
    rows: list[np.ndarray] = []
    for video_path in video_paths:
        task_dir = video_path.parents[2]
        frames = read_sampled_frames(video_path, samples)
        height, width = frames[0][1].shape[:2]
        header_height = 48
        row = np.full((height + header_height, width * len(frames), 3), 245, dtype=np.uint8)
        label = task_label(task_dir)
        cv2.putText(
            row,
            label[:110],
            (6, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            row,
            f"episode={video_path.parents[1].name}",
            (6, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        for column, (frame_index, frame) in enumerate(frames):
            x0 = column * width
            row[header_height:, x0 : x0 + width] = frame
            cv2.putText(
                row,
                f"f{frame_index}",
                (x0 + 4, header_height + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/libero_bbox_review"))
    parser.add_argument("--samples", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    vis_root = args.cache_root / "visualizations"
    suites = sorted(path for path in vis_root.iterdir() if path.is_dir())
    for suite_dir in suites:
        videos = sorted(
            [
                path
                for video_name in VIDEO_NAMES
                for path in suite_dir.glob(f"*/episode_*/{AGENTVIEW_DIR}/{video_name}")
            ],
            key=lambda path: task_label(path.parents[2]),
        )
        if not videos:
            continue
        output_path = args.output_dir / f"{suite_dir.name}.png"
        render_suite(videos, output_path, args.samples)
        print(f"{suite_dir.name}: {len(videos)} tasks -> {output_path}")


if __name__ == "__main__":
    main()
