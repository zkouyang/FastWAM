#!/usr/bin/env python3
"""Offline trajectory-label preprocessing for the LIBERO FastWAM dataset.

This motion-only preprocessor is extracted from `preprocess_libero_ssi.py` in
the same self-contained style as `preprocess_libero_bbox.py`: it keeps the
LIBERO episode/video loading and manifest writing, but only computes point
trajectory labels.

The produced `.motion.npz` cache stores episode-level trajectory labels:

    motion_points:       [C, T_episode, N, 2]
    motion_visibility:   [C, T_episode, N]
    motion_point_source: [C, N]

`motion_points` are normalized xy coordinates in `[0, 1]` over the resized
teacher video. `motion_point_source` uses `0` for ATM-style grid queries and
`1` for random dynamic queries after variance filtering and re-tracking.
"""

from __future__ import annotations

import argparse
import json
import re
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


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "unknown")[:max_len]


class MotionAnnotator:
    def __init__(
        self,
        backend: str,
        *,
        device: str,
        repo_or_dir: str,
        num_random_points: int,
        grid_size: int,
        var_threshold: float,
    ):
        self.backend = backend
        self.device = device
        self.repo_or_dir = repo_or_dir
        self.num_random_points = num_random_points
        self.grid_size = grid_size
        self.var_threshold = var_threshold
        self.model: Any | None = None

        if backend in {"cotracker2", "cotracker3"}:
            try:
                import torch
            except Exception as exc:  # pragma: no cover
                raise ImportError(f"Motion backend '{backend}' requires torch.") from exc

            source = "local" if Path(repo_or_dir).exists() else "github"
            hub_entry = "cotracker2" if backend == "cotracker2" else "cotracker3_offline"
            self.model = torch.hub.load(repo_or_dir, hub_entry, source=source)
            self.model = self.model.to(torch.device(device))
            self.model.eval()
            return

        raise ValueError(f"Unsupported motion backend: {backend}")

    def annotate(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Track a fixed set of points over the full episode.

        Returns:
            motion_points: [T, N, 2], normalized xy in [0, 1]
            motion_visibility: [T, N]
            motion_point_source: [N], 0=grid, 1=random_dynamic
        """

        _t, height, width = frames.shape[:3]
        tracks, visibility, point_source = self._cotracker_atm_style_tracks(frames)

        tracks_norm = tracks.astype(np.float32)
        tracks_norm[..., 0] /= max(width - 1, 1)
        tracks_norm[..., 1] /= max(height - 1, 1)
        tracks_norm = np.clip(tracks_norm, 0.0, 1.0)

        return tracks_norm.astype(np.float16), visibility.astype(np.float16), point_source

    def _call_tracker(self, video: Any, queries: Any) -> tuple[Any, Any]:
        assert self.model is not None

        try:
            return self.model(video, queries=queries, backward_tracking=True)
        except TypeError:
            return self.model(video, queries=queries)

    def _track_and_remove(self, video: Any, queries: Any, var_threshold: float) -> tuple[Any, Any, Any]:
        """ATM-style dynamic point filtering and re-tracking."""

        import torch

        _, _, _, height, width = video.shape
        pred_tracks, _pred_vis = self._call_tracker(video, queries)

        var = torch.var(pred_tracks, dim=1)
        var = torch.sum(var, dim=-1)[0]
        dynamic_idx = torch.where(var > var_threshold)[0]
        if len(dynamic_idx) == 0:
            warnings.warn(
                "No CoTracker query passed the trajectory variance threshold; "
                "falling back to the original queries for this episode."
            )
            new_queries = queries.clone()
        else:
            new_queries = queries[:, dynamic_idx].clone()
            rep = queries.shape[1] // new_queries.shape[1] + 1
            new_queries = torch.tile(new_queries, (1, rep, 1))[:, : queries.shape[1]]

            noise = torch.randn_like(new_queries[:, :, 1:])
            scale = torch.tensor([width, height], device=new_queries.device, dtype=new_queries.dtype)
            new_queries[:, :, 1:] += noise * 0.05 * scale
            new_queries[:, :, 1] = torch.clamp(new_queries[:, :, 1], 0, width - 1)
            new_queries[:, :, 2] = torch.clamp(new_queries[:, :, 2], 0, height - 1)

        pred_tracks, pred_vis = self._call_tracker(video, new_queries)
        return pred_tracks, pred_vis, new_queries

    def _cotracker_atm_style_tracks(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        t, height, width = frames.shape[:3]
        video = torch.as_tensor(frames).permute(0, 3, 1, 2)[None].float().to(self.device)

        random_xy = sample_random_points((height, width), self.num_random_points)
        random_queries = np.zeros((self.num_random_points, 3), dtype=np.float32)
        random_queries[:, 0] = np.random.randint(0, t, size=(self.num_random_points,))
        random_queries[:, 1:] = random_xy
        random_queries_t = torch.as_tensor(random_queries)[None].float().to(self.device)

        grid_xy = double_grid_points((height, width), self.grid_size)
        grid_queries = np.zeros((len(grid_xy), 3), dtype=np.float32)
        grid_queries[:, 0] = np.random.randint(0, t, size=(len(grid_xy),))
        grid_queries[:, 1:] = grid_xy
        grid_queries_t = torch.as_tensor(grid_queries)[None].float().to(self.device)

        pred_tracks, pred_vis, _ = self._track_and_remove(video, random_queries_t, self.var_threshold)
        pred_grid_tracks, pred_grid_vis, _ = self._track_and_remove(video, grid_queries_t, 0.0)

        pred_tracks = torch.cat([pred_grid_tracks, pred_tracks], dim=2)
        pred_vis = torch.cat([pred_grid_vis, pred_vis], dim=2)

        tracks = pred_tracks[0].detach().cpu().numpy().astype(np.float32)
        vis = pred_vis[0].detach().cpu().numpy().astype(np.float32)
        if vis.ndim == 3:
            vis = vis[..., 0]
        point_source = np.concatenate(
            [
                np.zeros((len(grid_xy),), dtype=np.int16),
                np.ones((self.num_random_points,), dtype=np.int16),
            ],
            axis=0,
        )
        return tracks, vis, point_source


def sample_random_points(image_hw: tuple[int, int], num_points: int) -> np.ndarray:
    h, w = image_hw
    xy = np.empty((num_points, 2), dtype=np.float32)
    xy[:, 0] = np.random.uniform(0, max(w - 1, 1), size=(num_points,))
    xy[:, 1] = np.random.uniform(0, max(h - 1, 1), size=(num_points,))
    return xy


def double_grid_points(image_hw: tuple[int, int], grid_size: int) -> np.ndarray:
    h, w = image_hw

    def _grid(left: tuple[float, float], right: tuple[float, float]) -> np.ndarray:
        xs = np.linspace(left[0] * w, right[0] * w, grid_size, dtype=np.float32)
        ys = np.linspace(left[1] * h, right[1] * h, grid_size, dtype=np.float32)
        return np.asarray([(x, y) for y in ys for x in xs], dtype=np.float32)

    return np.concatenate(
        [
            _grid((0.05, 0.05), (0.85, 0.85)),
            _grid((0.15, 0.15), (0.95, 0.95)),
        ],
        axis=0,
    )


def write_manifest_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_trajectory_vis_frames(
    frames: np.ndarray,
    motion_points: np.ndarray,
    motion_visibility: np.ndarray,
    frame_indices: np.ndarray,
    *,
    max_tracks: int,
    future_frames: int,
) -> list[np.ndarray]:
    if motion_points.shape[1] == 0:
        return [frames[int(i)].copy() for i in frame_indices]

    n = motion_points.shape[1]
    if n > max_tracks:
        track_indices = np.linspace(0, n - 1, max_tracks, dtype=np.int32)
    else:
        track_indices = np.arange(n, dtype=np.int32)

    h, w = frames.shape[1:3]
    palette = cv2.applyColorMap(np.linspace(0, 255, len(track_indices), dtype=np.uint8)[:, None], cv2.COLORMAP_HSV)
    palette = cv2.cvtColor(palette, cv2.COLOR_BGR2RGB).reshape(len(track_indices), 3)

    out: list[np.ndarray] = []
    for frame_idx in frame_indices:
        ti = int(frame_idx)
        canvas = frames[ti].copy()
        end = min(ti + future_frames + 1, motion_points.shape[0])
        for pi, track_idx in enumerate(track_indices):
            color = tuple(int(x) for x in palette[pi])
            pts = motion_points[ti:end, track_idx].astype(np.float32)
            vis = motion_visibility[ti:end, track_idx] > 0.5
            if not vis.any():
                continue
            xy = pts.copy()
            xy[:, 0] *= max(w - 1, 1)
            xy[:, 1] *= max(h - 1, 1)
            xy = xy.astype(np.int32)
            prev = None
            for p, visible in zip(xy, vis, strict=True):
                if not visible:
                    prev = None
                    continue
                cur = (int(p[0]), int(p[1]))
                if prev is not None:
                    cv2.line(canvas, prev, cur, color, 1, cv2.LINE_AA)
                prev = cur
            if vis[0]:
                cv2.circle(canvas, (int(xy[0, 0]), int(xy[0, 1])), 2, color, -1, cv2.LINE_AA)
        out.append(canvas)
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
    motion_points_all: list[np.ndarray],
    motion_vis_all: list[np.ndarray],
    fps: int,
    max_frames: int,
    max_tracks: int,
    future_frames: int,
) -> None:
    task_dir = vis_root / ep.suite / slugify(ep.task) / f"episode_{ep.episode_index:06d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.txt").write_text(ep.task + "\n", encoding="utf-8")

    if max_frames > 0 and len(frame_indices) > max_frames:
        vis_frame_indices = frame_indices[np.linspace(0, len(frame_indices) - 1, max_frames, dtype=np.int32)]
    else:
        vis_frame_indices = frame_indices

    for cam_idx, camera_key in enumerate(camera_keys):
        cam_dir = task_dir / f"camera_{cam_idx:02d}_{slugify(camera_key)}"
        write_rgb_video_or_contact_sheet(
            make_trajectory_vis_frames(
                frames_by_camera[cam_idx],
                motion_points_all[cam_idx],
                motion_vis_all[cam_idx],
                vis_frame_indices,
                max_tracks=max_tracks,
                future_frames=future_frames,
            ),
            cam_dir / "trajectory.mp4",
            fps,
        )


def process_episode(
    ep: EpisodeRecord,
    *,
    output_root: Path,
    camera_keys: tuple[str, ...],
    frame_stride: int,
    image_size: int,
    motion_annotator: MotionAnnotator,
    overwrite: bool,
    save_visualization: bool,
    visualization_root: Path,
    visualization_fps: int,
    visualization_max_frames: int,
    visualization_max_tracks: int,
    visualization_future_frames: int,
) -> dict[str, Any]:
    suite_out = output_root / ep.suite
    suite_out.mkdir(parents=True, exist_ok=True)
    out_path = suite_out / f"episode_{ep.episode_index:06d}.motion.npz"
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
    motion_points_all = []
    motion_vis_all = []
    motion_source_all = []

    desc = f"{ep.suite}/ep{ep.episode_index:06d} cameras"
    for frames in tqdm(frames_by_camera, desc=desc, leave=False):
        motion_points, motion_vis, motion_source = motion_annotator.annotate(frames)
        motion_points_all.append(motion_points)
        motion_vis_all.append(motion_vis)
        motion_source_all.append(motion_source)

    meta = {
        "schema_version": "fastwam_motion_v1_episode_labels",
        "suite": ep.suite,
        "episode_index": ep.episode_index,
        "task": ep.task,
        "camera_keys": camera_keys,
        "image_size": image_size,
        "frame_stride": frame_stride,
        "motion_backend": motion_annotator.backend,
        "motion_sampling": "atm_random_plus_grid",
        "motion_num_random_points": motion_annotator.num_random_points,
        "motion_grid_size": motion_annotator.grid_size,
        "motion_num_points": int(motion_points_all[0].shape[1]),
        "motion_var_threshold": motion_annotator.var_threshold,
        "motion_point_source_convention": "0=ATM grid query, 1=ATM random dynamic query after variance filtering/retracking",
        "coordinate_convention": "motion_points are normalized xy in [0,1] over the resized per-camera teacher image",
        "label_stage_note": "Training code should slice frame_indices/motion_points according to the desired FastWAM horizon.",
    }

    if save_cache:
        np.savez_compressed(
            out_path,
            frame_indices=frame_indices,
            camera_keys=np.asarray(camera_keys),
            motion_points=np.stack(motion_points_all, axis=0).astype(np.float16),
            motion_visibility=np.stack(motion_vis_all, axis=0).astype(np.float16),
            motion_point_source=np.stack(motion_source_all, axis=0).astype(np.int16),
            meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
        )

    if save_visualization:
        save_episode_visualizations(
            vis_root=visualization_root,
            ep=ep,
            camera_keys=camera_keys,
            frames_by_camera=frames_by_camera,
            frame_indices=frame_indices,
            motion_points_all=motion_points_all,
            motion_vis_all=motion_vis_all,
            fps=visualization_fps,
            max_frames=visualization_max_frames,
            max_tracks=visualization_max_tracks,
            future_frames=visualization_future_frames,
        )

    return {
        "suite": ep.suite,
        "episode_index": ep.episode_index,
        "task": ep.task,
        "path": str(out_path),
        "status": "ok" if save_cache else "visualized_existing",
        "num_frames": int(min_len),
        "num_frame_labels": int(len(frame_indices)),
        "num_motion_points": int(motion_points_all[0].shape[1]),
        "motion_backend": motion_annotator.backend,
        "visualization_saved": bool(save_visualization),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", default="./data/libero_mujoco3.3.2")
    parser.add_argument("--output-root", default="./data/libero_mujoco3.3.2_motion_cache")
    parser.add_argument("--suites", nargs="*", default=None, help="Suite directories or short names. Default: all *_lerobot.")
    parser.add_argument("--camera-keys", nargs="+", default=list(DEFAULT_CAMERA_KEYS))
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--image-size", type=int, default=224, help="Per-camera teacher input resolution.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Frame stride for frame_indices and visualization.")
    parser.add_argument("--motion-backend", choices=["cotracker3", "cotracker2"], default="cotracker3")
    parser.add_argument("--cotracker-repo-or-dir", default="facebookresearch/co-tracker")
    parser.add_argument("--motion-num-random-points", type=int, default=1000)
    parser.add_argument("--motion-grid-size", type=int, default=7)
    parser.add_argument("--motion-var-threshold", type=float, default=10.0)

    parser.add_argument("--vis-num-demos-per-task", type=int, default=1)
    parser.add_argument("--vis-output-dir", default=None)
    parser.add_argument("--vis-fps", type=int, default=10)
    parser.add_argument("--vis-max-frames", type=int, default=48)
    parser.add_argument("--vis-max-tracks", type=int, default=128)
    parser.add_argument(
        "--vis-future-frames",
        type=int,
        default=16,
        help="Draw each visualization point trajectory from the current frame through this many future frames.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.motion_num_random_points < 1:
        raise ValueError("--motion-num-random-points must be a positive integer")
    if args.motion_grid_size < 1:
        raise ValueError("--motion-grid-size must be a positive integer")
    if args.vis_num_demos_per_task < 0:
        raise ValueError("--vis-num-demos-per-task must be >= 0")
    if args.vis_fps < 1:
        raise ValueError("--vis-fps must be a positive integer")
    if args.vis_max_frames < 0:
        raise ValueError("--vis-max-frames must be >= 0")
    if args.vis_max_tracks < 1:
        raise ValueError("--vis-max-tracks must be a positive integer")
    if args.vis_future_frames < 0:
        raise ValueError("--vis-future-frames must be >= 0")

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

    print(f"[MOTION] data_root={data_root}")
    print(f"[MOTION] output_root={output_root}")
    print(f"[MOTION] suites={[p.name for p in suite_roots]}")
    print(f"[MOTION] selected_episodes={len(episodes)}")
    print(f"[MOTION] backend={args.motion_backend}")
    print(f"[MOTION] visualizations: first {args.vis_num_demos_per_task} demos/task -> {visualization_root}")

    motion_annotator = MotionAnnotator(
        args.motion_backend,
        device=args.device,
        repo_or_dir=args.cotracker_repo_or_dir,
        num_random_points=args.motion_num_random_points,
        grid_size=args.motion_grid_size,
        var_threshold=args.motion_var_threshold,
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
            motion_annotator=motion_annotator,
            overwrite=args.overwrite,
            save_visualization=save_visualization,
            visualization_root=visualization_root,
            visualization_fps=args.vis_fps,
            visualization_max_frames=args.vis_max_frames,
            visualization_max_tracks=args.vis_max_tracks,
            visualization_future_frames=args.vis_future_frames,
        )
        write_manifest_row(manifest_path, row)

    print(f"[MOTION] done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
