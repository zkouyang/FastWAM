#!/usr/bin/env python3
"""Merge refreshed camera records into existing LIBERO bbox cache files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_meta(cache: Any) -> dict[str, Any]:
    return json.loads(str(cache["meta_json"]))


def unpack_cameras(cache: Any) -> dict[str, dict[str, Any]]:
    camera_keys = [str(value) for value in cache["camera_keys"].tolist()]
    frame_indices = np.asarray(cache["frame_indices"], dtype=np.int32)
    offsets = np.asarray(cache["bbox_offsets"], dtype=np.int64)
    labels = json.loads(str(cache["bbox_labels_json"]))
    boxes = np.asarray(cache["bbox_xyxy"], dtype=np.float32)
    confidences = np.asarray(cache["bbox_confidences"], dtype=np.float32)
    masks = np.asarray(cache["bbox_masks"], dtype=np.bool_) if "bbox_masks" in cache else None
    num_frames = len(frame_indices)
    if offsets.shape != (len(camera_keys), num_frames + 1):
        raise ValueError(f"Invalid bbox_offsets shape {offsets.shape} for {len(camera_keys)} cameras")
    if len(labels) != len(camera_keys) * num_frames:
        raise ValueError(f"Invalid flattened label count: {len(labels)}")

    result: dict[str, dict[str, Any]] = {}
    for camera_index, camera_key in enumerate(camera_keys):
        camera_boxes: list[np.ndarray] = []
        camera_confidences: list[np.ndarray] = []
        camera_masks: list[np.ndarray] | None = [] if masks is not None else None
        camera_labels: list[list[str]] = []
        for frame_index in range(num_frames):
            start = int(offsets[camera_index, frame_index])
            end = int(offsets[camera_index, frame_index + 1])
            camera_boxes.append(boxes[start:end].copy())
            camera_confidences.append(confidences[start:end].copy())
            camera_labels.append(list(labels[camera_index * num_frames + frame_index]))
            if camera_masks is not None:
                camera_masks.append(masks[start:end].copy())
        result[camera_key] = {
            "boxes": camera_boxes,
            "confidences": camera_confidences,
            "labels": camera_labels,
            "masks": camera_masks,
        }
    return result


def pack_cameras(
    camera_keys: list[str],
    records: dict[str, dict[str, Any]],
    *,
    num_frames: int,
    image_size: int,
) -> dict[str, np.ndarray]:
    offsets = np.zeros((len(camera_keys), num_frames + 1), dtype=np.int64)
    counts = np.zeros((len(camera_keys), num_frames), dtype=np.int32)
    flat_boxes: list[np.ndarray] = []
    flat_confidences: list[np.ndarray] = []
    flat_masks: list[np.ndarray] = []
    flat_labels: list[list[str]] = []
    cursor = 0
    have_masks = all(records[key]["masks"] is not None for key in camera_keys)

    for camera_index, camera_key in enumerate(camera_keys):
        record = records[camera_key]
        offsets[camera_index, 0] = cursor
        for frame_index in range(num_frames):
            boxes = np.asarray(record["boxes"][frame_index], dtype=np.float32).reshape(-1, 4)
            confidences = np.asarray(record["confidences"][frame_index], dtype=np.float32).reshape(-1)
            labels = list(record["labels"][frame_index])
            if not (len(boxes) == len(confidences) == len(labels)):
                raise ValueError(f"Record length mismatch for {camera_key} frame {frame_index}")
            counts[camera_index, frame_index] = len(boxes)
            if len(boxes):
                flat_boxes.append(boxes)
                flat_confidences.append(confidences)
            if have_masks:
                masks = np.asarray(record["masks"][frame_index], dtype=np.bool_).reshape(
                    -1, image_size, image_size
                )
                if len(masks) != len(boxes):
                    raise ValueError(f"Mask length mismatch for {camera_key} frame {frame_index}")
                if len(masks):
                    flat_masks.append(masks)
            flat_labels.append(labels)
            cursor += len(boxes)
            offsets[camera_index, frame_index + 1] = cursor

    arrays = {
        "bbox_xyxy": np.concatenate(flat_boxes) if flat_boxes else np.zeros((0, 4), np.float32),
        "bbox_confidences": (
            np.concatenate(flat_confidences) if flat_confidences else np.zeros((0,), np.float32)
        ),
        "bbox_offsets": offsets,
        "bbox_counts": counts,
        "bbox_labels_json": np.asarray(json.dumps(flat_labels, ensure_ascii=False)),
    }
    if have_masks:
        arrays["bbox_masks"] = (
            np.concatenate(flat_masks)
            if flat_masks
            else np.zeros((0, image_size, image_size), dtype=np.bool_)
        )
    return arrays


def preserve_matching_instances(
    refreshed: dict[str, Any],
    existing: dict[str, Any],
    label_fragments: list[str],
) -> dict[str, Any]:
    if not label_fragments:
        return refreshed
    result = {"boxes": [], "confidences": [], "labels": [], "masks": []}
    if refreshed["masks"] is None or existing["masks"] is None:
        result["masks"] = None
    for frame_index, refreshed_labels in enumerate(refreshed["labels"]):
        preserved_indices = [
            index
            for index, label in enumerate(existing["labels"][frame_index])
            if any(fragment.lower() in label.lower() for fragment in label_fragments)
        ]
        result["boxes"].append(
            np.concatenate(
                [refreshed["boxes"][frame_index], existing["boxes"][frame_index][preserved_indices]],
                axis=0,
            )
        )
        result["confidences"].append(
            np.concatenate(
                [
                    refreshed["confidences"][frame_index],
                    existing["confidences"][frame_index][preserved_indices],
                ],
                axis=0,
            )
        )
        result["labels"].append(
            list(refreshed_labels)
            + [existing["labels"][frame_index][index] for index in preserved_indices]
        )
        if result["masks"] is not None:
            result["masks"].append(
                np.concatenate(
                    [refreshed["masks"][frame_index], existing["masks"][frame_index][preserved_indices]],
                    axis=0,
                )
            )
    return result


def merge_cache(
    source_path: Path,
    destination_path: Path,
    camera_key: str,
    preserve_labels: list[str] | None = None,
) -> None:
    if not destination_path.exists():
        raise FileNotFoundError(f"Destination cache does not exist: {destination_path}")
    with np.load(source_path, allow_pickle=False) as source, np.load(
        destination_path, allow_pickle=False
    ) as destination:
        source_frame_indices = np.asarray(source["frame_indices"], dtype=np.int32)
        destination_frame_indices = np.asarray(destination["frame_indices"], dtype=np.int32)
        if not np.array_equal(source_frame_indices, destination_frame_indices):
            raise ValueError(f"Frame indices differ for {source_path.name}")
        source_records = unpack_cameras(source)
        destination_records = unpack_cameras(destination)
        if camera_key not in source_records or camera_key not in destination_records:
            raise KeyError(f"Camera {camera_key!r} missing while merging {source_path.name}")
        destination_records[camera_key] = preserve_matching_instances(
            source_records[camera_key],
            destination_records[camera_key],
            preserve_labels or [],
        )

        destination_camera_keys = [str(value) for value in destination["camera_keys"].tolist()]
        source_meta = load_meta(source)
        destination_meta = load_meta(destination)
        image_size = int(destination_meta["image_size"])
        packed = pack_cameras(
            destination_camera_keys,
            destination_records,
            num_frames=len(destination_frame_indices),
            image_size=image_size,
        )
        meta = dict(destination_meta)
        for key in (
            "task_prompt",
            "task_prompt_source",
            "task_key",
            "box_threshold",
            "text_threshold",
        ):
            meta[key] = source_meta.get(key)
        camera_refresh = dict(meta.get("camera_refresh", {}))
        camera_refresh[camera_key] = {
            "bbox_backend": source_meta.get("bbox_backend"),
            "grounded_sam2_chunk_size": source_meta.get("grounded_sam2_chunk_size"),
            "sam2_config": source_meta.get("sam2_config"),
        }
        meta["camera_refresh"] = camera_refresh
        meta["camera_keys"] = destination_camera_keys

        arrays = {
            "frame_indices": destination_frame_indices,
            "camera_keys": np.asarray(destination_camera_keys),
            **packed,
            "meta_json": np.asarray(json.dumps(meta, ensure_ascii=False)),
        }

    tmp_path = destination_path.with_suffix(destination_path.suffix + ".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(destination_path)


def update_manifest(source_root: Path, destination_root: Path, merged: list[Path], camera_key: str) -> None:
    destination_manifest = destination_root / "manifest.jsonl"
    rows = []
    if destination_manifest.exists():
        rows = [json.loads(line) for line in destination_manifest.read_text(encoding="utf-8").splitlines() if line]
    merged_keys = {(path.parent.name, int(path.stem.split("_")[1].split(".")[0])) for path in merged}
    source_rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    source_manifest = source_root / "manifest.jsonl"
    if source_manifest.exists():
        for line in source_manifest.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                source_rows_by_key[(row["suite"], int(row["episode_index"]))] = row
    rows = [row for row in rows if (row.get("suite"), int(row.get("episode_index", -1))) not in merged_keys]
    for key in sorted(merged_keys):
        row = dict(source_rows_by_key[key])
        row["path"] = str(destination_root / key[0] / f"episode_{key[1]:06d}.bbox.npz")
        row["status"] = "camera_merged"
        row["updated_camera_keys"] = [camera_key]
        rows.append(row)
    tmp_path = destination_manifest.with_suffix(".jsonl.tmp")
    tmp_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    tmp_path.replace(destination_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--camera-key", default="observation.images.image")
    parser.add_argument(
        "--preserve-label",
        action="append",
        default=[],
        help="Keep matching destination instances in addition to refreshed source instances.",
    )
    args = parser.parse_args()

    merged: list[Path] = []
    for source_path in sorted(args.source_root.glob("*/*.bbox.npz")):
        destination_path = args.destination_root / source_path.parent.name / source_path.name
        merge_cache(source_path, destination_path, args.camera_key, args.preserve_label)
        merged.append(source_path)
        print(f"merged {args.camera_key}: {source_path.name} -> {destination_path}")
    if not merged:
        raise RuntimeError(f"No .bbox.npz files found under {args.source_root}")
    update_manifest(args.source_root, args.destination_root, merged, args.camera_key)


if __name__ == "__main__":
    main()
