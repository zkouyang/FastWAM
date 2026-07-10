#!/usr/bin/env python3
"""Offline depth-label preprocessing for the LIBERO FastWAM dataset.

This depth-only preprocessor is extracted from `preprocess_libero_ssi.py` in
the same spirit as `preprocess_libero_bbox.py`: it keeps the self-contained
LIBERO episode/video loading and manifest writing, but only computes monocular
depth grid labels.

The produced `.depth.npz` cache stores episode-level depth labels:

    depth:      [C, T_label, G, G]
    depth_conf: [C, T_label, G, G]

`C` is the number of camera keys, `T_label` is controlled by `--frame-stride`,
and `G` is controlled by `--grid-size`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import imageio.v2 as iio_v2
import imageio.v3 as iio
import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_CAMERA_KEYS = ("observation.images.image", "observation.images.wrist_image")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DA2_REPO_DIR = REPO_ROOT / "third_party" / "Depth_Anything_V2"
DEFAULT_DA2_CHECKPOINT = DEFAULT_DA2_REPO_DIR / "checkpoints" / "depth_anything_v2_vitl.pth"
DEFAULT_VDA_REPO_DIR = REPO_ROOT / "third_party" / "Video-Depth-Anything"
DEFAULT_VDA_CHECKPOINT = DEFAULT_VDA_REPO_DIR / "checkpoints" / "video_depth_anything_vitl.pth"
VDA_MODEL_CONFIGS = {
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


@dataclass(frozen=True)
class EpisodeRecord:
    suite: str
    suite_root: Path
    episode_index: int
    task: str
    length: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_data_root(path: str) -> Path:
    root = Path(path)
    if root.exists():
        return root

    alt = Path(str(root).replace("libero_mujuco3.3.2", "libero_mujoco3.3.2"))
    if alt.exists():
        warnings.warn(f"Data root {root} does not exist; using {alt} instead.")
        return alt

    raise FileNotFoundError(
        f"LIBERO data root not found: {root}. "
        "The FastWAM configs use ./data/libero_mujoco3.3.2."
    )


def discover_suite_roots(data_root: Path, suite_names: list[str] | None) -> list[Path]:
    if suite_names:
        roots = []
        for name in suite_names:
            p = data_root / name
            if not p.exists() and not name.endswith("_lerobot"):
                p = data_root / f"{name}_no_noops_lerobot"
            if not p.exists():
                raise FileNotFoundError(f"Suite not found under {data_root}: {name}")
            roots.append(p)
        return roots

    roots = sorted(p for p in data_root.glob("*_lerobot") if p.is_dir())
    if not roots:
        raise FileNotFoundError(f"No '*_lerobot' suites found under {data_root}")
    return roots


def iter_episodes(
    suite_roots: Iterable[Path],
    *,
    episode_start: int | None,
    episode_end: int | None,
    max_episodes: int | None,
) -> list[EpisodeRecord]:
    out: list[EpisodeRecord] = []
    for suite_root in suite_roots:
        suite = suite_root.name
        episodes_path = suite_root / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(f"Missing {episodes_path}")

        for row in read_jsonl(episodes_path):
            ep_idx = int(row["episode_index"])
            if episode_start is not None and ep_idx < episode_start:
                continue
            if episode_end is not None and ep_idx >= episode_end:
                continue

            tasks = row.get("tasks") or [""]
            out.append(
                EpisodeRecord(
                    suite=suite,
                    suite_root=suite_root,
                    episode_index=ep_idx,
                    task=str(tasks[0]),
                    length=int(row["length"]),
                )
            )
            if max_episodes is not None and len(out) >= max_episodes:
                return out
    return out


def episode_video_path(suite_root: Path, camera_key: str, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return (
        suite_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / camera_key
        / f"episode_{episode_index:06d}.mp4"
    )


def load_video_rgb(path: Path, *, resize: int | None = 224) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing video: {path}")
    errors: list[str] = []
    try:
        frames = iio.imread(path, plugin="FFMPEG")
    except Exception as exc:
        errors.append(f"imageio.v3/FFMPEG: {type(exc).__name__}: {exc}")
        try:
            reader = iio_v2.get_reader(str(path), format="ffmpeg")
            try:
                frames = np.stack([frame for frame in reader], axis=0)
            finally:
                reader.close()
        except Exception as exc:
            errors.append(f"imageio.v2/ffmpeg: {type(exc).__name__}: {exc}")
            try:
                import av

                container = av.open(str(path))
                try:
                    decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
                finally:
                    container.close()
                if decoded:
                    frames = np.stack(decoded, axis=0)
                else:
                    raise RuntimeError("PyAV decoded no frames")
            except Exception as exc:
                errors.append(f"PyAV: {type(exc).__name__}: {exc}")
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    raise ImportError(
                        "Could not decode mp4 with imageio-ffmpeg, PyAV, or OpenCV. "
                        "LIBERO videos in this dataset are AV1 encoded; install a working AV1 decoder "
                        "(for example `pip install av imageio[ffmpeg]`, or a system ffmpeg with libdav1d/libaom), "
                        f"then retry. Video: {path}. Tried backends: {' | '.join(errors)}"
                    )
                decoded = []
                try:
                    while True:
                        ok, bgr = cap.read()
                        if not ok:
                            break
                        decoded.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                finally:
                    cap.release()
                if not decoded:
                    errors.append("OpenCV: opened video but decoded no frames")
                    raise RuntimeError(
                        "Could not decode mp4 frames. LIBERO videos in this dataset are AV1 encoded; "
                        "install a working AV1 decoder (for example `pip install av imageio[ffmpeg]`, "
                        "or a system ffmpeg with libdav1d/libaom), then retry. "
                        f"Video: {path}. Tried backends: {' | '.join(errors)}"
                    )
                frames = np.stack(decoded, axis=0)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB video [T,H,W,3], got {frames.shape} from {path}")
    frames = np.asarray(frames, dtype=np.uint8)
    if resize is not None and (frames.shape[1] != resize or frames.shape[2] != resize):
        frames = np.stack(
            [cv2.resize(f, (resize, resize), interpolation=cv2.INTER_AREA) for f in frames],
            axis=0,
        )
    return frames


def robust_normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth, dtype=np.float32)
    valid = depth[finite]
    lo, hi = np.percentile(valid, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        return np.zeros_like(depth, dtype=np.float32)
    norm = (depth - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def resize_grid(arr: np.ndarray, grid_size: int, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    return cv2.resize(arr.astype(np.float32), (grid_size, grid_size), interpolation=interpolation)


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "unknown")[:max_len]


def ensure_depth_batch(arr: Any, *, expected_len: int, name: str) -> np.ndarray:
    """Normalize DA3 single-frame or batched maps to [N,H,W]."""
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 2:
        out = out[None]
    if out.ndim != 3:
        raise ValueError(f"Expected {name} to have shape [H,W] or [N,H,W], got {out.shape}")
    if out.shape[0] != expected_len:
        raise ValueError(f"Expected {name} batch length {expected_len}, got shape {out.shape}")
    return out


def ensure_conf_batch(conf: Any, *, depth: np.ndarray, expected_len: int) -> np.ndarray:
    """Normalize optional DA3 confidence maps; missing confidence means all-ones."""
    if conf is None:
        return np.ones_like(depth, dtype=np.float32)

    out = np.asarray(conf, dtype=np.float32)
    if out.ndim == 0:
        return np.ones_like(depth, dtype=np.float32) * float(out)
    if out.ndim == 2:
        out = out[None]
    if out.ndim != 3:
        raise ValueError(f"Expected DA3 confidence to have shape [H,W] or [N,H,W], got {out.shape}")
    if out.shape[0] != expected_len:
        raise ValueError(f"Expected DA3 confidence batch length {expected_len}, got shape {out.shape}")
    if out.shape != depth.shape:
        raise ValueError(f"Expected DA3 confidence shape to match depth {depth.shape}, got {out.shape}")
    return out


class DepthAnnotator:
    def __init__(
        self,
        backend: str,
        *,
        model_id: str,
        da2_repo_dir: str,
        da2_checkpoint: str,
        video_depth_anything_repo_dir: str,
        video_depth_anything_checkpoint: str,
        video_depth_anything_encoder: str,
        video_depth_anything_input_size: int,
        video_depth_anything_fp32: bool,
        device: str,
        grid_size: int,
    ):
        self.backend = backend
        self.grid_size = grid_size
        self.device = device
        self.model: Any | None = None
        self.video_depth_anything_input_size = video_depth_anything_input_size
        self.video_depth_anything_fp32 = video_depth_anything_fp32

        if backend == "da3":
            try:
                import torch
                from depth_anything_3.api import DepthAnything3
            except Exception as exc:  # pragma: no cover - optional teacher dependency
                raise ImportError(
                    "Depth backend 'da3' requires the official Depth Anything 3 package. "
                    "Install it from https://github.com/bytedance-seed/depth-anything-3."
                ) from exc

            torch_device = torch.device(device)
            self.model = DepthAnything3.from_pretrained(model_id).to(device=torch_device)
            self.model.eval()
            return
        if backend == "da2":
            try:
                import torch
            except Exception as exc:  # pragma: no cover - optional teacher dependency
                raise ImportError("Depth backend 'da2' requires torch.") from exc

            repo_dir = Path(da2_repo_dir).expanduser()
            if not repo_dir.exists():
                raise FileNotFoundError(f"Depth Anything V2 repo not found: {repo_dir}")
            if str(repo_dir) not in sys.path:
                sys.path.insert(0, str(repo_dir))

            try:
                from depth_anything_v2.dpt import DepthAnythingV2
            except Exception as exc:  # pragma: no cover - optional teacher dependency
                raise ImportError(
                    "Depth backend 'da2' requires a local Depth Anything V2 repo containing "
                    "depth_anything_v2/dpt.py. Pass it with --da2-repo-dir."
                ) from exc

            checkpoint = Path(da2_checkpoint).expanduser()
            if not checkpoint.exists():
                raise FileNotFoundError(f"Depth Anything V2 ViT-L checkpoint not found: {checkpoint}")

            model_config = {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            }
            self.model = DepthAnythingV2(**model_config)
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state)
            self.model = self.model.to(torch.device(device)).eval()
            return
        if backend == "video_depth_anything":
            try:
                import torch
            except Exception as exc:  # pragma: no cover - optional teacher dependency
                raise ImportError("Depth backend 'video_depth_anything' requires torch.") from exc

            repo_dir = Path(video_depth_anything_repo_dir).expanduser()
            if not repo_dir.exists():
                raise FileNotFoundError(f"Video Depth Anything repo not found: {repo_dir}")
            if str(repo_dir) not in sys.path:
                sys.path.insert(0, str(repo_dir))

            try:
                from video_depth_anything.video_depth import VideoDepthAnything
            except Exception as exc:  # pragma: no cover - optional teacher dependency
                raise ImportError(
                    "Depth backend 'video_depth_anything' requires a local Video-Depth-Anything repo "
                    "containing video_depth_anything/video_depth.py. Pass it with "
                    "--video-depth-anything-repo-dir."
                ) from exc

            if video_depth_anything_encoder not in VDA_MODEL_CONFIGS:
                raise ValueError(
                    f"Unsupported Video Depth Anything encoder: {video_depth_anything_encoder}. "
                    f"Choose one of {sorted(VDA_MODEL_CONFIGS)}."
                )
            checkpoint = Path(video_depth_anything_checkpoint).expanduser()
            if not checkpoint.exists():
                raise FileNotFoundError(f"Video Depth Anything checkpoint not found: {checkpoint}")

            self.model = VideoDepthAnything(**VDA_MODEL_CONFIGS[video_depth_anything_encoder])
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=True)
            self.model = self.model.to(torch.device(device)).eval()
            return

        raise ValueError(f"Unsupported depth backend: {backend}")

    def annotate(self, frames: np.ndarray, frame_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected = frames[frame_indices]
        if self.backend == "da2":
            return self._annotate_da2(selected)
        if self.backend == "video_depth_anything":
            return self._annotate_video_depth_anything(selected)

        assert self.backend == "da3"
        assert self.model is not None
        return self._annotate_da3(selected)

    def _annotate_da2(self, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.model is not None
        depth_grids = []
        conf_grids = []
        raw_depths = []
        for frame in selected:
            bgr = np.ascontiguousarray(frame[..., ::-1])
            depth = np.asarray(self.model.infer_image(bgr), dtype=np.float32)
            depth_norm = robust_normalize_depth(depth)
            raw_depths.append(depth_norm)
            depth_grids.append(resize_grid(depth_norm, self.grid_size))
            conf_grids.append(np.ones((self.grid_size, self.grid_size), dtype=np.float32))
        return (
            np.stack(depth_grids).astype(np.float16),
            np.stack(conf_grids).astype(np.float16),
            np.stack(raw_depths).astype(np.float16),
        )

    def _annotate_video_depth_anything(self, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.model is not None
        import torch

        torch_device = torch.device(self.device)
        device_type = torch_device.type
        fp32 = self.video_depth_anything_fp32 or device_type == "cpu"
        with torch.inference_mode():
            depth, _ = self.model.infer_video_depth(
                selected,
                target_fps=-1,
                input_size=self.video_depth_anything_input_size,
                device=device_type,
                fp32=fp32,
            )

        depth_grids = []
        conf_grids = []
        raw_depths = []
        for d in ensure_depth_batch(depth, expected_len=len(selected), name="Video Depth Anything depth"):
            depth_norm = robust_normalize_depth(d)
            raw_depths.append(depth_norm)
            depth_grids.append(resize_grid(depth_norm, self.grid_size))
            conf_grids.append(np.ones((self.grid_size, self.grid_size), dtype=np.float32))
        return (
            np.stack(depth_grids).astype(np.float16),
            np.stack(conf_grids).astype(np.float16),
            np.stack(raw_depths).astype(np.float16),
        )

    def _annotate_da3(self, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.model is not None
        with tempfile.TemporaryDirectory(prefix="fastwam_da3_") as tmp:
            tmpdir = Path(tmp)
            paths: list[str] = []
            for i, frame in enumerate(selected):
                p = tmpdir / f"{i:06d}.png"
                Image.fromarray(frame).save(p)
                paths.append(str(p))

            prediction = self.model.inference(paths)
            depth = ensure_depth_batch(prediction.depth, expected_len=len(selected), name="DA3 depth")
            conf = ensure_conf_batch(getattr(prediction, "conf", None), depth=depth, expected_len=len(selected))

        depth_grids = []
        conf_grids = []
        raw_depths = []
        for d, c in zip(depth, conf, strict=True):
            depth_norm = robust_normalize_depth(d)
            raw_depths.append(depth_norm)
            depth_grids.append(resize_grid(depth_norm, self.grid_size))
            conf_grids.append(resize_grid(np.clip(c, 0.0, 1.0), self.grid_size))
        return (
            np.stack(depth_grids).astype(np.float16),
            np.stack(conf_grids).astype(np.float16),
            np.stack(raw_depths).astype(np.float16),
        )


def write_manifest_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def colorize_scalar_map(values: np.ndarray, size_hw: tuple[int, int], cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    h, w = size_hw
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    arr_u8 = (arr * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(arr_u8, cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_depth_vis_frames(
    frames: np.ndarray,
    depth: np.ndarray,
    raw_depth: np.ndarray,
    frame_indices: np.ndarray,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i, frame_idx in enumerate(frame_indices):
        frame = frames[int(frame_idx)]
        depth_rgb = colorize_scalar_map(depth[i], frame.shape[:2])
        raw_depth_rgb = colorize_scalar_map(raw_depth[i], frame.shape[:2])
        out.append(np.concatenate([frame, depth_rgb, raw_depth_rgb], axis=1))
    return out


def write_rgb_video_or_contact_sheet(frames: list[np.ndarray], path: Path, fps: int, *, max_sheet_frames: int = 12) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    ok = False
    try:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (w, h),
        )
        if writer.isOpened():
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            ok = True
        writer.release()
    except Exception as exc:
        warnings.warn(f"Failed to write visualization video {path}: {exc}")

    if ok:
        return

    sheet_path = path.with_suffix(".png")
    select = frames
    if len(select) > max_sheet_frames:
        select = [frames[i] for i in np.linspace(0, len(frames) - 1, max_sheet_frames, dtype=np.int32)]
    cols = min(4, len(select))
    rows = int(np.ceil(len(select) / cols))
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, frame in enumerate(select):
        r, c = divmod(idx, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = frame
    Image.fromarray(canvas).save(sheet_path)


def save_episode_visualizations(
    *,
    vis_root: Path,
    ep: EpisodeRecord,
    camera_keys: tuple[str, ...],
    frames_by_camera: list[np.ndarray],
    frame_indices: np.ndarray,
    depth_all: list[np.ndarray],
    raw_depth_all: list[np.ndarray],
    fps: int,
    max_frames: int,
) -> None:
    task_dir = vis_root / ep.suite / slugify(ep.task) / f"episode_{ep.episode_index:06d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.txt").write_text(ep.task + "\n", encoding="utf-8")

    if max_frames > 0 and len(frame_indices) > max_frames:
        keep = np.linspace(0, len(frame_indices) - 1, max_frames, dtype=np.int32)
        vis_frame_indices = frame_indices[keep]
    else:
        keep = np.arange(len(frame_indices), dtype=np.int32)
        vis_frame_indices = frame_indices

    for cam_idx, camera_key in enumerate(camera_keys):
        cam_dir = task_dir / f"camera_{cam_idx:02d}_{slugify(camera_key)}"
        frames = frames_by_camera[cam_idx]
        depth = depth_all[cam_idx][keep]
        raw_depth = raw_depth_all[cam_idx][keep]
        write_rgb_video_or_contact_sheet(
            make_depth_vis_frames(frames, depth, raw_depth, vis_frame_indices),
            cam_dir / "depth_map.mp4",
            fps,
        )


def process_episode(
    ep: EpisodeRecord,
    *,
    output_root: Path,
    camera_keys: tuple[str, ...],
    frame_stride: int,
    image_size: int,
    depth_annotator: DepthAnnotator,
    overwrite: bool,
    save_visualization: bool,
    visualization_root: Path,
    visualization_fps: int,
    visualization_max_frames: int,
) -> dict[str, Any]:
    suite_out = output_root / ep.suite
    suite_out.mkdir(parents=True, exist_ok=True)
    out_path = suite_out / f"episode_{ep.episode_index:06d}.depth.npz"
    save_cache = overwrite or not out_path.exists()
    if out_path.exists() and not overwrite and not save_visualization:
        return {
            "suite": ep.suite,
            "episode_index": ep.episode_index,
            "task": ep.task,
            "path": str(out_path),
            "status": "skipped_exists",
        }

    frames_by_camera = [
        load_video_rgb(episode_video_path(ep.suite_root, cam, ep.episode_index), resize=image_size)
        for cam in camera_keys
    ]

    min_len = min(len(x) for x in frames_by_camera)
    if min_len <= 0:
        raise ValueError(f"Empty video for {ep.suite} episode {ep.episode_index}")
    if any(len(x) != min_len for x in frames_by_camera):
        warnings.warn(
            f"Camera length mismatch in {ep.suite} episode {ep.episode_index}; truncating to {min_len} frames."
        )
        frames_by_camera = [x[:min_len] for x in frames_by_camera]

    frame_indices = np.arange(0, min_len, frame_stride, dtype=np.int32)
    depth_all = []
    depth_conf_all = []
    raw_depth_all = []

    desc = f"{ep.suite}/ep{ep.episode_index:06d} cameras"
    for frames in tqdm(frames_by_camera, desc=desc, leave=False):
        depth, depth_conf, raw_depth = depth_annotator.annotate(frames, frame_indices)
        depth_all.append(depth)
        depth_conf_all.append(depth_conf)
        raw_depth_all.append(raw_depth)

    meta = {
        "schema_version": "fastwam_depth_v1_episode_labels",
        "suite": ep.suite,
        "episode_index": ep.episode_index,
        "task": ep.task,
        "camera_keys": camera_keys,
        "image_size": image_size,
        "grid_size": depth_annotator.grid_size,
        "frame_stride": frame_stride,
        "depth_backend": depth_annotator.backend,
        "depth_convention": "depth is robust-normalized to [0,1] per labeled frame and downsampled to the grid",
        "label_stage_note": "Training code should slice frame_indices/depth according to the desired FastWAM horizon.",
    }

    if save_cache:
        np.savez_compressed(
            out_path,
            frame_indices=frame_indices,
            camera_keys=np.asarray(camera_keys),
            depth=np.stack(depth_all, axis=0).astype(np.float16),
            depth_conf=np.stack(depth_conf_all, axis=0).astype(np.float16),
            meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
        )

    if save_visualization:
        save_episode_visualizations(
            vis_root=visualization_root,
            ep=ep,
            camera_keys=camera_keys,
            frames_by_camera=frames_by_camera,
            frame_indices=frame_indices,
            depth_all=depth_all,
            raw_depth_all=raw_depth_all,
            fps=visualization_fps,
            max_frames=visualization_max_frames,
        )

    return {
        "suite": ep.suite,
        "episode_index": ep.episode_index,
        "task": ep.task,
        "path": str(out_path),
        "status": "ok" if save_cache else "visualized_existing",
        "num_frames": int(min_len),
        "num_frame_labels": int(len(frame_indices)),
        "depth_backend": depth_annotator.backend,
        "visualization_saved": bool(save_visualization),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", default="./data/libero_mujoco3.3.2")
    parser.add_argument("--output-root", default="./data/libero_mujoco3.3.2_depth_cache")
    parser.add_argument("--suites", nargs="*", default=None, help="Suite directories or short names. Default: all *_lerobot.")
    parser.add_argument("--camera-keys", nargs="+", default=list(DEFAULT_CAMERA_KEYS))
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--depth-backend", choices=["da3", "da2", "video_depth_anything"], default="da3")
    parser.add_argument("--da3-model-id", default="depth-anything/DA3MONO-LARGE")
    parser.add_argument(
        "--da2-repo-dir",
        default=str(DEFAULT_DA2_REPO_DIR),
        help="Local Depth Anything V2 repo directory for --depth-backend da2.",
    )
    parser.add_argument(
        "--da2-checkpoint",
        default=str(DEFAULT_DA2_CHECKPOINT),
        help="Depth Anything V2 ViT-L checkpoint for --depth-backend da2.",
    )
    parser.add_argument(
        "--video-depth-anything-repo-dir",
        default=str(DEFAULT_VDA_REPO_DIR),
        help="Local Video Depth Anything repo directory for --depth-backend video_depth_anything.",
    )
    parser.add_argument(
        "--video-depth-anything-checkpoint",
        default=str(DEFAULT_VDA_CHECKPOINT),
        help="Video Depth Anything checkpoint for --depth-backend video_depth_anything.",
    )
    parser.add_argument(
        "--video-depth-anything-encoder",
        choices=sorted(VDA_MODEL_CONFIGS),
        default="vitl",
        help="Video Depth Anything encoder variant. The default matches the Large checkpoint.",
    )
    parser.add_argument(
        "--video-depth-anything-input-size",
        type=int,
        default=518,
        help="Video Depth Anything inference input size; use a smaller multiple of 14 for CPU smoke tests.",
    )
    parser.add_argument(
        "--video-depth-anything-fp32",
        action="store_true",
        help="Run Video Depth Anything in fp32 instead of autocast. CPU always uses fp32.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Per-camera teacher input resolution.")
    parser.add_argument("--grid-size", type=int, default=8, help="Depth grid size.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Depth frame stride. Use 1 for full coverage.")

    parser.add_argument("--vis-num-demos-per-task", type=int, default=1)
    parser.add_argument("--vis-output-dir", default=None)
    parser.add_argument("--vis-fps", type=int, default=10)
    parser.add_argument("--vis-max-frames", type=int, default=48)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.grid_size < 1:
        raise ValueError("--grid-size must be a positive integer")
    if args.video_depth_anything_input_size < 14:
        raise ValueError("--video-depth-anything-input-size must be >= 14")
    if args.vis_num_demos_per_task < 0:
        raise ValueError("--vis-num-demos-per-task must be >= 0")
    if args.vis_fps < 1:
        raise ValueError("--vis-fps must be a positive integer")
    if args.vis_max_frames < 0:
        raise ValueError("--vis-max-frames must be >= 0")

    data_root = resolve_data_root(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visualization_root = Path(args.vis_output_dir) if args.vis_output_dir else output_root / "visualizations"
    manifest_path = output_root / "manifest.jsonl"
    if args.overwrite and manifest_path.exists():
        manifest_path.unlink()

    suite_roots = discover_suite_roots(data_root, args.suites)
    episodes = iter_episodes(
        suite_roots,
        episode_start=args.episode_start,
        episode_end=args.episode_end,
        max_episodes=args.max_episodes,
    )
    if not episodes:
        raise RuntimeError("No episodes selected.")

    print(f"[DEPTH] data_root={data_root}")
    print(f"[DEPTH] output_root={output_root}")
    print(f"[DEPTH] suites={[p.name for p in suite_roots]}")
    print(f"[DEPTH] selected_episodes={len(episodes)}")
    print(f"[DEPTH] backend={args.depth_backend}")
    print(f"[DEPTH] visualizations: first {args.vis_num_demos_per_task} demos/task -> {visualization_root}")

    depth_annotator = DepthAnnotator(
        args.depth_backend,
        model_id=args.da3_model_id,
        da2_repo_dir=args.da2_repo_dir,
        da2_checkpoint=args.da2_checkpoint,
        video_depth_anything_repo_dir=args.video_depth_anything_repo_dir,
        video_depth_anything_checkpoint=args.video_depth_anything_checkpoint,
        video_depth_anything_encoder=args.video_depth_anything_encoder,
        video_depth_anything_input_size=args.video_depth_anything_input_size,
        video_depth_anything_fp32=args.video_depth_anything_fp32,
        device=args.device,
        grid_size=args.grid_size,
    )

    visualization_counts: dict[tuple[str, str], int] = {}
    for ep in tqdm(episodes, desc="episodes"):
        vis_key = (ep.suite, ep.task)
        cur_vis_count = visualization_counts.get(vis_key, 0)
        save_visualization = args.vis_num_demos_per_task > 0 and cur_vis_count < args.vis_num_demos_per_task
        if save_visualization:
            visualization_counts[vis_key] = cur_vis_count + 1

        row = process_episode(
            ep,
            output_root=output_root,
            camera_keys=tuple(args.camera_keys),
            frame_stride=args.frame_stride,
            image_size=args.image_size,
            depth_annotator=depth_annotator,
            overwrite=args.overwrite,
            save_visualization=save_visualization,
            visualization_root=visualization_root,
            visualization_fps=args.vis_fps,
            visualization_max_frames=args.vis_max_frames,
        )
        write_manifest_row(manifest_path, row)

    print(f"[DEPTH] done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
