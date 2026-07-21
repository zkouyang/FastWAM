#!/usr/bin/env python3
"""Validate LIBERO depth/bbox/mask/trajectory cache coverage and alignment.

The fast coverage pass checks every episode without decompressing large arrays.
A deterministic random subset is then opened to verify schema, camera order,
frame alignment, ragged offsets, instance-mask correspondence, and finite data.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SUITES = (
    "libero_spatial_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_10_no_noops_lerobot",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def expected_episodes(data_root: Path, suites: list[str]) -> list[tuple[str, int, int]]:
    episodes: list[tuple[str, int, int]] = []
    for suite in suites:
        path = data_root / suite / "meta" / "episodes.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing LIBERO episode metadata: {path}")
        for row in read_jsonl(path):
            episodes.append((suite, int(row["episode_index"]), int(row["length"])))
    return episodes


def cache_path(root: Path, suite: str, episode: int, suffix: str) -> Path:
    return root / suite / f"episode_{episode:06d}.{suffix}.npz"


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def inspect_episode(
    record: tuple[str, int, int],
    *,
    depth_root: Path,
    bbox_root: Path,
    trajectory_root: Path,
    require_masks: bool,
) -> list[str]:
    suite, episode, episode_length = record
    errors: list[str] = []
    paths = {
        "depth": cache_path(depth_root, suite, episode, "depth"),
        "bbox": cache_path(bbox_root, suite, episode, "bbox"),
        "trajectory": cache_path(trajectory_root, suite, episode, "motion"),
    }
    archives: dict[str, dict[str, np.ndarray]] = {}
    for name, path in paths.items():
        try:
            with np.load(path, allow_pickle=False) as archive:
                archives[name] = {key: archive[key] for key in archive.files}
        except Exception as exc:
            errors.append(f"{suite}/episode_{episode:06d} cannot read {name}: {exc}")
    if len(archives) != 3:
        return errors

    depth = archives["depth"]
    bbox = archives["bbox"]
    trajectory = archives["trajectory"]
    prefix = f"{suite}/episode_{episode:06d}"
    raw_frames = depth.get("frame_indices")
    raw_cameras = depth.get("camera_keys")
    if raw_frames is None or raw_cameras is None:
        errors.append(f"{prefix}: depth cache is missing frame_indices or camera_keys")
        return errors
    reference_frames = np.asarray(raw_frames)
    reference_cameras = np.asarray(raw_cameras)
    if reference_frames.ndim != 1 or reference_cameras.ndim != 1:
        errors.append(f"{prefix}: frame_indices and camera_keys must both be 1D")
        return errors
    require(
        len(reference_frames) == episode_length,
        f"{prefix}: label length {len(reference_frames)} != episode length {episode_length}",
        errors,
    )
    require(
        np.array_equal(reference_frames, np.arange(episode_length)),
        f"{prefix}: labels are not a complete stride-1 episode timeline",
        errors,
    )
    for name, archive in (("bbox", bbox), ("trajectory", trajectory)):
        require(
            np.array_equal(archive.get("frame_indices"), reference_frames),
            f"{prefix}: {name} frame_indices differ from depth",
            errors,
        )
        require(
            np.array_equal(archive.get("camera_keys"), reference_cameras),
            f"{prefix}: {name} camera order differs from depth",
            errors,
        )

    depth_values = depth.get("depth")
    depth_conf = depth.get("depth_conf")
    expected_prefix = (len(reference_cameras), len(reference_frames))
    require(
        depth_values is not None and depth_values.ndim == 4 and depth_values.shape[:2] == expected_prefix,
        f"{prefix}: invalid depth shape {getattr(depth_values, 'shape', None)}",
        errors,
    )
    require(
        depth_conf is not None and depth_conf.shape == getattr(depth_values, "shape", None),
        f"{prefix}: depth_conf does not match depth",
        errors,
    )
    if depth_values is not None:
        require(bool(np.isfinite(depth_values).all()), f"{prefix}: depth has NaN/Inf", errors)
    if depth_conf is not None:
        require(bool(np.isfinite(depth_conf).all()), f"{prefix}: depth_conf has NaN/Inf", errors)

    boxes = bbox.get("bbox_xyxy")
    scores = bbox.get("bbox_confidences")
    offsets = bbox.get("bbox_offsets")
    counts = bbox.get("bbox_counts")
    expected_offset_shape = (len(reference_cameras), len(reference_frames) + 1)
    boxes_valid = boxes is not None and boxes.ndim == 2 and boxes.shape[1:] == (4,)
    num_boxes = len(boxes) if boxes_valid else -1
    require(
        boxes_valid,
        f"{prefix}: invalid bbox_xyxy shape {getattr(boxes, 'shape', None)}",
        errors,
    )
    require(
        scores is not None and scores.shape == (num_boxes,),
        f"{prefix}: bbox confidence count mismatch",
        errors,
    )
    require(
        offsets is not None and offsets.shape == expected_offset_shape,
        f"{prefix}: invalid bbox_offsets shape {getattr(offsets, 'shape', None)}",
        errors,
    )
    require(
        counts is not None and counts.shape == expected_prefix,
        f"{prefix}: invalid bbox_counts shape {getattr(counts, 'shape', None)}",
        errors,
    )
    if offsets is not None and offsets.shape == expected_offset_shape:
        require(
            bool((np.diff(offsets, axis=1) >= 0).all()),
            f"{prefix}: bbox offsets are not monotonic",
            errors,
        )
        require(
            int(offsets[-1, -1]) == num_boxes,
            f"{prefix}: final bbox offset does not equal K",
            errors,
        )
        if counts is not None and counts.shape == expected_prefix:
            require(
                np.array_equal(np.diff(offsets, axis=1), counts),
                f"{prefix}: bbox counts differ from offsets",
                errors,
            )
    if boxes is not None:
        require(bool(np.isfinite(boxes).all()), f"{prefix}: bbox_xyxy has NaN/Inf", errors)
    masks = bbox.get("bbox_masks")
    if require_masks:
        require(masks is not None, f"{prefix}: bbox_masks is missing", errors)
    if masks is not None:
        require(
            masks.ndim == 3 and masks.shape[0] == num_boxes,
            f"{prefix}: mask/box instance count mismatch",
            errors,
        )

    points = trajectory.get("motion_points")
    visibility = trajectory.get("motion_visibility")
    source = trajectory.get("motion_point_source")
    require(
        points is not None and points.ndim == 4 and points.shape[:2] == expected_prefix and points.shape[-1] == 2,
        f"{prefix}: invalid motion_points shape {getattr(points, 'shape', None)}",
        errors,
    )
    if points is not None and points.ndim == 4:
        require(
            visibility is not None and visibility.shape == points.shape[:-1],
            f"{prefix}: motion_visibility does not match points",
            errors,
        )
        require(
            source is not None and source.shape == (points.shape[0], points.shape[2]),
            f"{prefix}: motion_point_source does not match points",
            errors,
        )
        require(bool(np.isfinite(points).all()), f"{prefix}: motion_points has NaN/Inf", errors)
    if visibility is not None:
        require(bool(np.isfinite(visibility).all()), f"{prefix}: motion_visibility has NaN/Inf", errors)
    return errors


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data/libero_mujoco3.3.2")
    parser.add_argument("--depth-cache-dir", default="./data/libero_mujoco3.3.2_depth_cache")
    parser.add_argument("--bbox-cache-dir", default="./data/libero_mujoco3.3.2_bbox_cache")
    parser.add_argument("--trajectory-cache-dir", default="./data/libero_mujoco3.3.2_motion_cache")
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing-masks", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.sample_count < 1:
        raise ValueError("--sample-count must be >= 1")
    data_root = Path(args.data_root)
    depth_root = Path(args.depth_cache_dir)
    bbox_root = Path(args.bbox_cache_dir)
    trajectory_root = Path(args.trajectory_cache_dir)
    records = expected_episodes(data_root, args.suites)

    missing: list[str] = []
    for suite, episode, _length in records:
        for name, root, suffix in (
            ("depth", depth_root, "depth"),
            ("bbox/mask", bbox_root, "bbox"),
            ("trajectory", trajectory_root, "motion"),
        ):
            path = cache_path(root, suite, episode, suffix)
            if not path.is_file():
                missing.append(f"{name}: {path}")
    if missing:
        preview = "\n".join(missing[:20])
        raise SystemExit(f"Missing {len(missing)} episode caches:\n{preview}")

    rng = random.Random(args.seed)
    sampled = rng.sample(records, min(args.sample_count, len(records)))
    errors: list[str] = []
    for record in sampled:
        errors.extend(
            inspect_episode(
                record,
                depth_root=depth_root,
                bbox_root=bbox_root,
                trajectory_root=trajectory_root,
                require_masks=not args.allow_missing_masks,
            )
        )
    if errors:
        raise SystemExit("Auxiliary cache validation failed:\n" + "\n".join(errors))

    per_suite = {
        suite: sum(record[0] == suite for record in records)
        for suite in args.suites
    }
    print(f"Validated cache coverage for {len(records)} episodes: {per_suite}")
    print(
        f"Deep-validated {len(sampled)} deterministic random episodes "
        f"(seed={args.seed}); depth/bbox/mask/trajectory timelines and cameras align."
    )


if __name__ == "__main__":
    main()
