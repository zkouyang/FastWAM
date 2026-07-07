#!/usr/bin/env python3
"""Offline SSI annotation preprocessing for the LIBERO FastWAM dataset.

The script builds episode-level SSI caches for the auxiliary SSI branch:

1. MSGE / geometry: monocular depth grid supervision from Depth Anything 3.
2. LGOL / layout: language-grounded role layout maps from Grounding DINO-T by default,
   with SAM3 kept as an optional backend.
3. ICMP / motion: point trajectories from CoTracker3.

The produced cache is intentionally independent from the original LeRobot
parquet files. It reads LIBERO's `meta/episodes.jsonl` and per-camera mp4
videos, then writes compressed `.npz` files plus a `manifest.jsonl`.

Important: this is a label-stage preprocessor. It saves full episode timelines
instead of FastWAM training windows. The training dataset should later slice
these labels according to `num_frames`, `action_video_freq_ratio`, and the SSI
loss horizon.

Expected default LIBERO root:
    ./data/libero_mujoco3.3.2

Example smoke test without heavy teachers:
    python scripts/preprocess_libero_ssi.py \
        --data-root ./data/libero_mujoco3.3.2 \
        --max-episodes 1 \
        --depth-backend heuristic \
        --layout-backend heuristic \
        --motion-backend opencv

Example full teacher run, assuming DA3, Grounding DINO-T and CoTracker are installed:
    python scripts/preprocess_libero_ssi.py \
        --data-root ./data/libero_mujoco3.3.2 \
        --depth-backend da3 \
        --layout-backend grounding_dino_t \
        --motion-backend cotracker3 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
DEFAULT_GROUNDING_DINO_T_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
ROLE_NAMES = ("source", "target", "robot", "other")
ROLE_COLORS = {
    "source": (255, 80, 80),
    "target": (80, 220, 120),
    "robot": (80, 160, 255),
    "other": (255, 210, 80),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM3_CHECKPOINT_CANDIDATES = (
    REPO_ROOT / "third_party" / "sam3" / "checkpoints" / "sam3.pt",
    REPO_ROOT / "third_party" / "sam3" / "sam3.pt",
    REPO_ROOT / "checkpoints" / "sam3.pt",
    REPO_ROOT / "data" / "checkpoints" / "sam3.pt",
)


def resolve_sam3_checkpoint_path(path: str | None) -> Path | None:
    """Resolve a SAM3 checkpoint path, preferring local files by default."""
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    env_path = os.environ.get("SAM3_CHECKPOINT_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(DEFAULT_SAM3_CHECKPOINT_CANDIDATES)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def format_sam3_checkpoint_candidates() -> str:
    paths = [str(p) for p in DEFAULT_SAM3_CHECKPOINT_CANDIDATES]
    return "\n".join(f"  - {p}" for p in paths)


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

    # Common typo in this project context: "mujuco" vs "mujoco".
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
        # Some environments install imageio-ffmpeg but do not expose the v3
        # plugin name. The v2 reader is slower but more broadly compatible.
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


def masks_to_role_grid(
    masks: list[np.ndarray],
    scores: list[float],
    *,
    image_hw: tuple[int, int],
    grid_size: int,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_hw
    if not masks:
        z = np.zeros((grid_size, grid_size), dtype=np.float32)
        return z, z

    dense = np.zeros((h, w), dtype=np.float32)
    dense_conf = np.zeros((h, w), dtype=np.float32)
    for mask, score in zip(masks, scores, strict=False):
        score = float(score)
        if score < score_threshold:
            continue
        m = np.asarray(mask)
        if m.ndim > 2:
            m = np.squeeze(m)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        m = m.astype(bool)
        dense[m] = np.maximum(dense[m], score)
        dense_conf[m] = np.maximum(dense_conf[m], score)
    grid = resize_grid(dense, grid_size)
    conf = resize_grid(dense_conf, grid_size)
    return np.clip(grid, 0.0, 1.0), np.clip(conf, 0.0, 1.0)


def boxes_to_role_grid(
    boxes_xyxy: list[np.ndarray],
    scores: list[float],
    *,
    image_hw: tuple[int, int],
    grid_size: int,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert language-grounded boxes into the same SSI role-grid schema as masks.

    Grounding DINO-T provides boxes rather than masks. For the SSI label stage we
    rasterize accepted boxes into a dense confidence map, then downsample to the
    role grid. This keeps the downstream `layout/layout_conf` tensor contract
    identical to SAM3 while using a non-gated default teacher.
    """
    h, w = image_hw
    if not boxes_xyxy:
        z = np.zeros((grid_size, grid_size), dtype=np.float32)
        return z, z

    dense = np.zeros((h, w), dtype=np.float32)
    dense_conf = np.zeros((h, w), dtype=np.float32)
    for box, score in zip(boxes_xyxy, scores, strict=False):
        score = float(score)
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = np.asarray(box, dtype=np.float32).reshape(-1)[:4]
        x1 = int(np.floor(np.clip(x1, 0, max(w - 1, 0))))
        y1 = int(np.floor(np.clip(y1, 0, max(h - 1, 0))))
        x2 = int(np.ceil(np.clip(x2, 0, w)))
        y2 = int(np.ceil(np.clip(y2, 0, h)))
        if x2 <= x1 or y2 <= y1:
            continue
        dense[y1:y2, x1:x2] = np.maximum(dense[y1:y2, x1:x2], score)
        dense_conf[y1:y2, x1:x2] = np.maximum(dense_conf[y1:y2, x1:x2], score)

    grid = resize_grid(dense, grid_size)
    conf = resize_grid(dense_conf, grid_size)
    return np.clip(grid, 0.0, 1.0), np.clip(conf, 0.0, 1.0)


def compact_phrase(text: str) -> str:
    text = re.sub(r"\b(the|a|an)\b", " ", text.lower())
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;")


def extract_role_prompts(task: str) -> dict[str, list[str]]:
    """Heuristic LIBERO instruction parser for SAM text prompts.

    This is intentionally transparent and deterministic. It gives SAM 3.1
    short noun-like phrases rather than requiring an online LLM.
    """

    t = compact_phrase(task)
    source: list[str] = []
    target: list[str] = []
    other: list[str] = []

    # Common LIBERO patterns: "pick up X and place it in/on Y".
    m = re.search(r"(?:pick up|grab|move|put|place)\s+(.+?)(?:\s+and\s+|\s+from\s+|\s+in\s+|\s+on\s+|$)", t)
    if m:
        source.append(compact_phrase(m.group(1)))

    m = re.search(r"(?:place it|put it|place .+?|put .+?)\s+(?:in|on|inside|into|to)\s+(.+?)(?:$| and |,)", t)
    if m:
        target.append(compact_phrase(m.group(1)))

    m = re.search(r"(?:from|between|next to|on|in)\s+(.+?)(?:\s+and\s+place|\s+and\s+put|$)", t)
    if m:
        other.append(compact_phrase(m.group(1)))

    # Extra task words that often appear as objects or receptacles.
    known = (
        "bowl",
        "plate",
        "ramekin",
        "basket",
        "drawer",
        "cabinet",
        "stove",
        "mug",
        "microwave",
        "moka pot",
        "sauce",
        "butter",
        "alphabet soup",
        "cream cheese",
        "tomato sauce",
        "book",
        "box",
        "cup",
        "table",
    )
    for noun in known:
        if noun in t and noun not in source and noun not in target:
            other.append(noun)

    source = [p for p in dict.fromkeys(source) if p]
    target = [p for p in dict.fromkeys(target) if p]
    other = [p for p in dict.fromkeys(other) if p]

    # Conservative fallbacks keep every role non-empty for a stable schema.
    if not source:
        source = [t]
    if not target:
        target = ["target object", "receptacle", "container"]
    if not other:
        other = ["table", "workspace"]

    return {
        "source": source[:4],
        "target": target[:4],
        "robot": ["robot gripper", "robot hand", "gripper"],
        "other": other[:6],
    }


class DepthAnnotator:
    def __init__(self, backend: str, *, model_id: str, device: str, grid_size: int):
        self.backend = backend
        self.grid_size = grid_size
        self.device = device
        self.model: Any | None = None

        if backend == "none":
            return
        if backend == "heuristic":
            return
        if backend == "da3":
            try:
                import torch
                from depth_anything_3.api import DepthAnything3
            except Exception as exc:  # pragma: no cover - optional teacher dependency
                raise ImportError(
                    "Depth backend 'da3' requires the official Depth Anything 3 package. "
                    "Install it from https://github.com/bytedance-seed/depth-anything-3 "
                    "or use --depth-backend heuristic for a pipeline smoke test."
                ) from exc

            torch_device = torch.device(device)
            self.model = DepthAnything3.from_pretrained(model_id).to(device=torch_device)
            self.model.eval()
            return

        raise ValueError(f"Unsupported depth backend: {backend}")

    def annotate(self, frames: np.ndarray, frame_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.backend == "none":
            t = len(frame_indices)
            z = np.zeros((t, self.grid_size, self.grid_size), dtype=np.float16)
            return z, z

        selected = frames[frame_indices]
        if self.backend == "heuristic":
            depth_grids = []
            conf_grids = []
            for frame in selected:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                # A deterministic proxy: darker regions are treated as closer.
                depth = robust_normalize_depth(1.0 - gray)
                depth_grids.append(resize_grid(depth, self.grid_size))
                conf_grids.append(np.ones((self.grid_size, self.grid_size), dtype=np.float32) * 0.25)
            return np.stack(depth_grids).astype(np.float16), np.stack(conf_grids).astype(np.float16)

        assert self.backend == "da3"
        assert self.model is not None
        with tempfile.TemporaryDirectory(prefix="fastwam_da3_") as tmp:
            tmpdir = Path(tmp)
            paths: list[str] = []
            for i, frame in enumerate(selected):
                p = tmpdir / f"{i:06d}.png"
                Image.fromarray(frame).save(p)
                paths.append(str(p))

            prediction = self.model.inference(paths)
            depth = np.asarray(prediction.depth, dtype=np.float32)
            conf = np.asarray(getattr(prediction, "conf", np.ones_like(depth)), dtype=np.float32)

        depth_grids = []
        conf_grids = []
        for d, c in zip(depth, conf, strict=True):
            depth_grids.append(resize_grid(robust_normalize_depth(d), self.grid_size))
            conf_grids.append(resize_grid(np.clip(c, 0.0, 1.0), self.grid_size))
        return np.stack(depth_grids).astype(np.float16), np.stack(conf_grids).astype(np.float16)


class LayoutAnnotator:
    def __init__(
        self,
        backend: str,
        *,
        device: str,
        grid_size: int,
        score_threshold: float,
        grounding_dino_model_id: str,
        grounding_dino_text_threshold: float,
        grounding_dino_local_files_only: bool,
        sam3_checkpoint_path: str | None,
        sam3_load_from_hf: bool,
        sam3_compile: bool,
    ):
        self.backend = backend
        self.grid_size = grid_size
        self.score_threshold = score_threshold
        self.text_threshold = grounding_dino_text_threshold
        self.processor: Any | None = None
        self.model: Any | None = None
        self.torch: Any | None = None

        if backend == "none":
            return
        if backend == "heuristic":
            return
        if backend == "grounding_dino_t":
            try:
                import torch
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            except Exception as exc:  # pragma: no cover - optional teacher dependency
                raise ImportError(
                    "Layout backend 'grounding_dino_t' requires torch and transformers. "
                    "Install them in the fastwam environment or use --layout-backend heuristic "
                    "for a pipeline smoke test."
                ) from exc

            torch_device = torch.device(device)
            try:
                self.processor = AutoProcessor.from_pretrained(
                    grounding_dino_model_id,
                    local_files_only=grounding_dino_local_files_only,
                )
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    grounding_dino_model_id,
                    local_files_only=grounding_dino_local_files_only,
                )
            except Exception as exc:  # pragma: no cover - optional teacher dependency
                raise RuntimeError(
                    "Failed to load Grounding DINO-T. If this is the first run, download "
                    "the public checkpoint with:\n"
                    "  conda run -n fastwam python -c \"from huggingface_hub import "
                    "hf_hub_download; files=['config.json','preprocessor_config.json',"
                    "'tokenizer_config.json','tokenizer.json','vocab.txt',"
                    "'special_tokens_map.json','model.safetensors']; "
                    "[hf_hub_download('IDEA-Research/grounding-dino-tiny', f) for f in files]\"\n"
                    "or pass --no-grounding-dino-local-files-only to allow an online load."
                ) from exc
            self.model = self.model.to(torch_device)
            self.model.eval()
            self.torch = torch
            print(f"[SSI] Grounding DINO-T model: {grounding_dino_model_id}")
            return
        if backend == "sam3":
            try:
                import torch
                from sam3.model.sam3_image_processor import Sam3Processor
                from sam3.model_builder import build_sam3_image_model
            except Exception as exc:  # pragma: no cover - optional teacher dependency
                raise ImportError(
                    "Layout backend 'sam3' requires the official SAM 3 / SAM 3.1 package. "
                    "Install it from https://github.com/facebookresearch/sam3, clone it to "
                    "third_party/sam3 and install it in the active environment, or use "
                    "--layout-backend heuristic for a pipeline smoke test."
                ) from exc

            if not str(device).startswith("cuda"):
                raise ValueError(
                    "Layout backend 'sam3' currently requires --device cuda. "
                    "The official SAM3 image builder precomputes some tensors on CUDA "
                    "during initialization. Use --layout-backend heuristic for CPU-only "
                    "pipeline tests."
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Layout backend 'sam3' requires a visible CUDA GPU, but "
                    "torch.cuda.is_available() is false in the active environment. "
                    "Check CUDA visibility/driver setup, or use --layout-backend "
                    "heuristic for a CPU-only smoke test."
                )

            requested_checkpoint_path = sam3_checkpoint_path
            resolved_checkpoint_path = resolve_sam3_checkpoint_path(sam3_checkpoint_path)
            if requested_checkpoint_path is not None and resolved_checkpoint_path is None:
                raise FileNotFoundError(
                    "SAM3 local checkpoint was requested but does not exist: "
                    f"{Path(requested_checkpoint_path).expanduser()}"
                )
            if resolved_checkpoint_path is not None:
                sam3_checkpoint_path = str(resolved_checkpoint_path)
                sam3_load_from_hf = False
                print(f"[SSI] SAM3 checkpoint: {sam3_checkpoint_path}")

            if sam3_checkpoint_path is None and not sam3_load_from_hf:
                raise ValueError(
                    "Layout backend 'sam3' needs pretrained SAM3 weights, and the "
                    "default policy is local-only.\n\n"
                    "Put the official checkpoint at one of these default locations:\n"
                    f"{format_sam3_checkpoint_candidates()}\n\n"
                    "Alternatively set SAM3_CHECKPOINT_PATH=/path/to/sam3.pt, pass "
                    "--sam3-checkpoint-path /path/to/sam3.pt, or explicitly enable "
                    "Hugging Face download with --sam3-load-from-hf after logging in "
                    "to an account that has access to facebook/sam3."
                )

            try:
                model = build_sam3_image_model(
                    device=device,
                    checkpoint_path=sam3_checkpoint_path,
                    load_from_HF=sam3_load_from_hf,
                    compile=sam3_compile,
                )
            except Exception as exc:  # pragma: no cover - optional teacher dependency
                message = str(exc)
                if (
                    "GatedRepoError" in exc.__class__.__name__
                    or "Cannot access gated repo" in message
                    or "401 Client Error" in message
                    or "Unauthorized" in message
                ):
                    raise RuntimeError(
                        "SAM3 package is installed, but its checkpoint download failed "
                        "because facebook/sam3 is a gated Hugging Face repository.\n\n"
                        "Fix options:\n"
                        "  1. Run `huggingface-cli login` in the fastwam environment "
                        "with an account that has accepted access to facebook/sam3, "
                        "then rerun the command; or\n"
                        "  2. Download the official SAM3 image checkpoint manually and "
                        "rerun with `--sam3-checkpoint-path /path/to/sam3.pt`; or\n"
                        "  3. Use `--layout-backend heuristic` for a dependency/pipeline "
                        "smoke test without SAM3 labels.\n\n"
                        "No SSI label files were written by the SAM3 layout stage before "
                        "this initialization failure."
                    ) from exc
                raise RuntimeError(
                    "Failed to initialize SAM3 layout annotator. If you are running "
                    "offline, pass a local checkpoint with --sam3-checkpoint-path; "
                    "otherwise make sure Hugging Face access is configured. "
                    f"Original error: {exc.__class__.__name__}: {exc}"
                ) from exc
            model.to(torch.device(device))
            model.eval()
            self.processor = Sam3Processor(model)
            return

        raise ValueError(f"Unsupported layout backend: {backend}")

    def annotate(
        self,
        frames: np.ndarray,
        frame_indices: np.ndarray,
        role_prompts: dict[str, list[str]],
    ) -> tuple[np.ndarray, np.ndarray]:
        t = len(frame_indices)
        r = len(ROLE_NAMES)
        if self.backend == "none":
            z = np.zeros((t, r, self.grid_size, self.grid_size), dtype=np.float16)
            return z, z

        selected = frames[frame_indices]
        if self.backend == "heuristic":
            layout = np.zeros((t, r, self.grid_size, self.grid_size), dtype=np.float32)
            conf = np.zeros_like(layout)
            yy, xx = np.mgrid[0 : self.grid_size, 0 : self.grid_size]
            center = np.exp(-(((xx - (self.grid_size - 1) / 2) ** 2 + (yy - (self.grid_size - 1) / 2) ** 2) / 8.0))
            bottom = np.clip((yy + 1) / self.grid_size, 0.0, 1.0)
            for i in range(t):
                layout[i, 0] = center
                layout[i, 1] = bottom
                layout[i, 2, -2:, self.grid_size // 3 : 2 * self.grid_size // 3] = 1.0
                layout[i, 3] = 0.15
                conf[i] = 0.20
            return layout.astype(np.float16), conf.astype(np.float16)

        if self.backend == "grounding_dino_t":
            assert self.processor is not None
            assert self.model is not None
            assert self.torch is not None
            layout = np.zeros((t, r, self.grid_size, self.grid_size), dtype=np.float32)
            conf = np.zeros_like(layout)
            device = next(self.model.parameters()).device

            for ti, frame in enumerate(selected):
                image = Image.fromarray(frame)
                h, w = frame.shape[:2]
                for ri, role in enumerate(ROLE_NAMES):
                    role_boxes: list[np.ndarray] = []
                    role_scores: list[float] = []
                    for prompt in role_prompts.get(role, []):
                        text_prompt = prompt.strip()
                        if not text_prompt:
                            continue
                        if not text_prompt.endswith("."):
                            text_prompt = f"{text_prompt}."
                        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
                        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
                        with self.torch.inference_mode():
                            outputs = self.model(**inputs)
                        results = self.processor.post_process_grounded_object_detection(
                            outputs,
                            input_ids=inputs.get("input_ids"),
                            threshold=self.score_threshold,
                            text_threshold=self.text_threshold,
                            target_sizes=[(h, w)],
                        )
                        if not results:
                            continue
                        boxes = results[0].get("boxes", [])
                        scores = results[0].get("scores", [])
                        if hasattr(boxes, "detach"):
                            boxes = boxes.detach().cpu().numpy()
                        if hasattr(scores, "detach"):
                            scores = scores.detach().cpu().numpy()
                        boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
                        scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
                        for bi, box in enumerate(boxes_arr):
                            role_boxes.append(box)
                            role_scores.append(float(scores_arr[bi]) if bi < len(scores_arr) else 1.0)
                    layout[ti, ri], conf[ti, ri] = boxes_to_role_grid(
                        role_boxes,
                        role_scores,
                        image_hw=(h, w),
                        grid_size=self.grid_size,
                        score_threshold=self.score_threshold,
                    )
            return layout.astype(np.float16), conf.astype(np.float16)

        assert self.backend == "sam3"
        assert self.processor is not None
        layout = np.zeros((t, r, self.grid_size, self.grid_size), dtype=np.float32)
        conf = np.zeros_like(layout)

        for ti, frame in enumerate(selected):
            image = Image.fromarray(frame)
            state = self.processor.set_image(image)
            h, w = frame.shape[:2]
            for ri, role in enumerate(ROLE_NAMES):
                role_masks: list[np.ndarray] = []
                role_scores: list[float] = []
                for prompt in role_prompts.get(role, []):
                    output = self.processor.set_text_prompt(state=state, prompt=prompt)
                    masks = output.get("masks", [])
                    scores = output.get("scores", [])
                    if hasattr(masks, "detach"):
                        masks = masks.detach().cpu().numpy()
                    if hasattr(scores, "detach"):
                        scores = scores.detach().cpu().numpy()
                    masks_arr = np.asarray(masks)
                    scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
                    if masks_arr.size == 0:
                        continue
                    if masks_arr.ndim == 2:
                        masks_arr = masks_arr[None]
                    for mi, m in enumerate(masks_arr):
                        role_masks.append(m)
                        score = float(scores_arr[mi]) if mi < len(scores_arr) else 1.0
                        role_scores.append(score)
                layout[ti, ri], conf[ti, ri] = masks_to_role_grid(
                    role_masks,
                    role_scores,
                    image_hw=(h, w),
                    grid_size=self.grid_size,
                    score_threshold=self.score_threshold,
                )
        return layout.astype(np.float16), conf.astype(np.float16)


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

        if backend in {"none", "opencv"}:
            return
        if backend == "cotracker3":
            try:
                import torch
            except Exception as exc:  # pragma: no cover
                raise ImportError("CoTracker3 backend requires torch.") from exc

            source = "local" if Path(repo_or_dir).exists() else "github"
            self.model = torch.hub.load(repo_or_dir, "cotracker3_offline", source=source)
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

        t, height, width = frames.shape[:3]

        if self.backend == "none":
            n_grid_points = 2 * self.grid_size * self.grid_size
            n_points = n_grid_points + self.num_random_points
            tracks = np.zeros((t, n_points, 2), dtype=np.float32)
            visibility = np.zeros((t, n_points), dtype=np.float32)
            point_source = np.concatenate(
                [
                    np.zeros((n_grid_points,), dtype=np.int16),
                    np.ones((self.num_random_points,), dtype=np.int16),
                ],
                axis=0,
            )
            return tracks.astype(np.float16), visibility.astype(np.float16), point_source

        if self.backend == "opencv":
            # OpenCV LK is only a lightweight smoke-test fallback. It tracks
            # deterministic grid points from frame 0 and does not reproduce
            # ATM's random-time backward tracking.
            init_points = double_grid_points((height, width), self.grid_size)
            point_source = np.zeros((len(init_points),), dtype=np.int16)
            tracks, visibility = opencv_point_tracks(frames, init_points)
        else:
            tracks, visibility, point_source = self._cotracker_atm_style_tracks(frames)

        # Convert pixel coordinates to [0,1] normalized coordinates.
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
        """ATM-style dynamic point filtering and re-tracking.

        First track sampled points, keep points whose trajectory variance is
        above `var_threshold`, then repeat them with small spatial noise and
        track again. This suppresses many static/background points.
        """

        import torch

        _, _, _, height, width = video.shape
        pred_tracks, _pred_vis = self._call_tracker(video, queries)

        var = torch.var(pred_tracks, dim=1)  # [B,N,2]
        var = torch.sum(var, dim=-1)[0]  # [N]
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

        tracks = pred_tracks[0].detach().cpu().numpy().astype(np.float32)  # [T,N,2]
        vis = pred_vis[0].detach().cpu().numpy().astype(np.float32)  # [T,N] or [T,N,1]
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


def uniform_grid_points(image_hw: tuple[int, int], grid_size: int) -> np.ndarray:
    h, w = image_hw
    xs = np.linspace(0.5 * w / grid_size, w - 0.5 * w / grid_size, grid_size)
    ys = np.linspace(0.5 * h / grid_size, h - 0.5 * h / grid_size, grid_size)
    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.float32)


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

    # Same structure as ATM's sample_double_grid(n): two interleaved nxn grids
    # with offsets [0.05, 0.85] and [0.15, 0.95].
    return np.concatenate(
        [
            _grid((0.05, 0.05), (0.85, 0.85)),
            _grid((0.15, 0.15), (0.95, 0.95)),
        ],
        axis=0,
    )


def opencv_point_tracks(clip: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sparse LK fallback for smoke tests; CoTracker3 should be used for real labels."""

    t, h, w = clip.shape[:3]
    n = len(points)
    tracks = np.zeros((t, n, 2), dtype=np.float32)
    vis = np.zeros((t, n), dtype=np.float32)
    tracks[0] = points
    vis[0] = 1.0

    prev_gray = cv2.cvtColor(clip[0], cv2.COLOR_RGB2GRAY)
    prev_points = points.reshape(-1, 1, 2)
    alive = np.ones(n, dtype=bool)
    for ti in range(1, t):
        gray = cv2.cvtColor(clip[ti], cv2.COLOR_RGB2GRAY)
        nxt, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if nxt is None or status is None:
            nxt = prev_points.copy()
            status = np.zeros((n, 1), dtype=np.uint8)
        status = status.reshape(-1).astype(bool)
        alive &= status
        cur = nxt.reshape(-1, 2)
        tracks[ti] = cur
        vis[ti] = alive.astype(np.float32)
        prev_gray = gray
        prev_points = cur.reshape(-1, 1, 2)
    return tracks, vis


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


def make_depth_vis_frames(frames: np.ndarray, depth: np.ndarray, frame_indices: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i, frame_idx in enumerate(frame_indices):
        frame = frames[int(frame_idx)]
        depth_rgb = colorize_scalar_map(depth[i], frame.shape[:2])
        blended = cv2.addWeighted(frame, 0.45, depth_rgb, 0.55, 0.0)
        panel = np.concatenate([frame, depth_rgb, blended], axis=1)
        out.append(panel)
    return out


def bbox_from_grid(role_grid: np.ndarray, image_hw: tuple[int, int], threshold: float) -> tuple[int, int, int, int] | None:
    grid = np.asarray(role_grid, dtype=np.float32)
    if grid.size == 0 or float(grid.max()) <= 0.0:
        return None
    mask = grid >= max(threshold, 0.25 * float(grid.max()))
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    h, w = image_hw
    gh, gw = grid.shape
    x1 = int(np.floor(xs.min() * w / gw))
    x2 = int(np.ceil((xs.max() + 1) * w / gw))
    y1 = int(np.floor(ys.min() * h / gh))
    y2 = int(np.ceil((ys.max() + 1) * h / gh))
    return max(x1, 0), max(y1, 0), min(x2, w - 1), min(y2, h - 1)


def make_bbox_vis_frames(
    frames: np.ndarray,
    layout: np.ndarray,
    layout_conf: np.ndarray,
    frame_indices: np.ndarray,
    threshold: float,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i, frame_idx in enumerate(frame_indices):
        frame = frames[int(frame_idx)].copy()
        heat = np.zeros_like(frame, dtype=np.float32)
        for role_idx, role in enumerate(ROLE_NAMES):
            role_grid = layout[i, role_idx]
            bbox = bbox_from_grid(role_grid, frame.shape[:2], threshold)
            color = ROLE_COLORS.get(role, (255, 255, 255))
            role_heat = cv2.resize(role_grid.astype(np.float32), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            for ch, c in enumerate(color):
                heat[..., ch] += role_heat * c
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            conf = float(layout_conf[i, role_idx].max()) if layout_conf.size else float(role_grid.max())
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{role}:{conf:.2f}",
                (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        heat = np.clip(heat, 0, 255).astype(np.uint8)
        blended = cv2.addWeighted(frame, 0.72, heat, 0.28, 0.0)
        out.append(blended)
    return out


def make_trajectory_vis_frames(
    frames: np.ndarray,
    motion_points: np.ndarray,
    motion_visibility: np.ndarray,
    frame_indices: np.ndarray,
    *,
    max_tracks: int,
    trail: int = 12,
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
        start = max(0, ti - trail)
        for pi, track_idx in enumerate(track_indices):
            color = tuple(int(x) for x in palette[pi])
            pts = motion_points[start : ti + 1, track_idx].astype(np.float32)
            vis = motion_visibility[start : ti + 1, track_idx] > 0.5
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
            if vis[-1]:
                cv2.circle(canvas, (int(xy[-1, 0]), int(xy[-1, 1])), 2, color, -1, cv2.LINE_AA)
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
    depth_all: list[np.ndarray],
    layout_all: list[np.ndarray],
    layout_conf_all: list[np.ndarray],
    motion_points_all: list[np.ndarray],
    motion_vis_all: list[np.ndarray],
    fps: int,
    max_frames: int,
    max_tracks: int,
    bbox_threshold: float,
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
        layout = layout_all[cam_idx][keep]
        layout_conf = layout_conf_all[cam_idx][keep]
        motion_points = motion_points_all[cam_idx]
        motion_vis = motion_vis_all[cam_idx]

        write_rgb_video_or_contact_sheet(
            make_depth_vis_frames(frames, depth, vis_frame_indices),
            cam_dir / "depth_map.mp4",
            fps,
        )
        write_rgb_video_or_contact_sheet(
            make_bbox_vis_frames(frames, layout, layout_conf, vis_frame_indices, bbox_threshold),
            cam_dir / "bbox_map.mp4",
            fps,
        )
        write_rgb_video_or_contact_sheet(
            make_trajectory_vis_frames(
                frames,
                motion_points,
                motion_vis,
                vis_frame_indices,
                max_tracks=max_tracks,
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
    depth_annotator: DepthAnnotator,
    layout_annotator: LayoutAnnotator,
    motion_annotator: MotionAnnotator,
    overwrite: bool,
    save_visualization: bool,
    visualization_root: Path,
    visualization_fps: int,
    visualization_max_frames: int,
    visualization_max_tracks: int,
    visualization_bbox_threshold: float,
) -> dict[str, Any]:
    suite_out = output_root / ep.suite
    suite_out.mkdir(parents=True, exist_ok=True)
    out_path = suite_out / f"episode_{ep.episode_index:06d}.ssi.npz"
    save_cache = overwrite or not out_path.exists()
    if out_path.exists() and not overwrite and not save_visualization:
        return {
            "suite": ep.suite,
            "episode_index": ep.episode_index,
            "task": ep.task,
            "path": str(out_path),
            "status": "skipped_exists",
        }

    role_prompts = extract_role_prompts(ep.task)
    frames_by_camera = []
    for cam in camera_keys:
        frames_by_camera.append(load_video_rgb(episode_video_path(ep.suite_root, cam, ep.episode_index), resize=image_size))

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
    layout_all = []
    layout_conf_all = []
    motion_points_all = []
    motion_vis_all = []
    motion_source_all = []

    for frames in tqdm(frames_by_camera, desc=f"{ep.suite}/ep{ep.episode_index:06d} cameras", leave=False):
        depth, depth_conf = depth_annotator.annotate(frames, frame_indices)
        layout, layout_conf = layout_annotator.annotate(frames, frame_indices, role_prompts)
        motion_points, motion_vis, motion_source = motion_annotator.annotate(frames)

        depth_all.append(depth)
        depth_conf_all.append(depth_conf)
        layout_all.append(layout)
        layout_conf_all.append(layout_conf)
        motion_points_all.append(motion_points)
        motion_vis_all.append(motion_vis)
        motion_source_all.append(motion_source)

    meta = {
        "schema_version": "fastwam_ssi_v2_episode_labels",
        "suite": ep.suite,
        "episode_index": ep.episode_index,
        "task": ep.task,
        "role_names": ROLE_NAMES,
        "role_prompts": role_prompts,
        "camera_keys": camera_keys,
        "image_size": image_size,
        "grid_size": depth_annotator.grid_size,
        "frame_stride": frame_stride,
        "motion_backend": motion_annotator.backend,
        "motion_sampling": "atm_random_plus_grid" if motion_annotator.backend == "cotracker3" else f"{motion_annotator.backend}_fallback",
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
            role_names=np.asarray(ROLE_NAMES),
            depth=np.stack(depth_all, axis=0).astype(np.float16),  # [C,T,G,G]
            depth_conf=np.stack(depth_conf_all, axis=0).astype(np.float16),
            layout=np.stack(layout_all, axis=0).astype(np.float16),  # [C,T,R,G,G]
            layout_conf=np.stack(layout_conf_all, axis=0).astype(np.float16),
            motion_points=np.stack(motion_points_all, axis=0).astype(np.float16),  # [C,T,N,2]
            motion_visibility=np.stack(motion_vis_all, axis=0).astype(np.float16),  # [C,T,N]
            motion_point_source=np.stack(motion_source_all, axis=0).astype(np.int16),  # [C,N]
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
            layout_all=layout_all,
            layout_conf_all=layout_conf_all,
            motion_points_all=motion_points_all,
            motion_vis_all=motion_vis_all,
            fps=visualization_fps,
            max_frames=visualization_max_frames,
            max_tracks=visualization_max_tracks,
            bbox_threshold=visualization_bbox_threshold,
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
        "visualization_saved": bool(save_visualization),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", default="./data/libero_mujoco3.3.2")
    parser.add_argument("--output-root", default="./data/libero_mujoco3.3.2_ssi_cache")
    parser.add_argument("--suites", nargs="*", default=None, help="Suite directories or short names. Default: all *_lerobot.")
    parser.add_argument("--camera-keys", nargs="+", default=list(DEFAULT_CAMERA_KEYS))
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--image-size", type=int, default=224, help="Per-camera teacher input resolution.")
    parser.add_argument("--grid-size", type=int, default=8, help="SSI depth/layout grid size.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Depth/layout frame stride. Use 1 for full training coverage.")
    parser.add_argument(
        "--motion-num-random-points",
        type=int,
        default=1000,
        help="ATM-style number of random CoTracker query points before variance filtering.",
    )
    parser.add_argument(
        "--motion-grid-size",
        type=int,
        default=7,
        help="ATM-style double-grid query size; adds 2 * motion_grid_size^2 extra points.",
    )
    parser.add_argument(
        "--motion-var-threshold",
        type=float,
        default=10.0,
        help="ATM-style variance threshold for filtering static random points before re-tracking.",
    )

    parser.add_argument("--depth-backend", choices=["da3", "heuristic", "none"], default="da3")
    parser.add_argument("--da3-model-id", default="depth-anything/DA3MONO-LARGE")
    parser.add_argument(
        "--layout-backend",
        choices=["grounding_dino_t", "sam3", "heuristic", "none"],
        default="grounding_dino_t",
    )
    parser.add_argument("--sam-score-threshold", type=float, default=0.30)
    parser.add_argument(
        "--grounding-dino-model-id",
        default=DEFAULT_GROUNDING_DINO_T_MODEL_ID,
        help="Grounding DINO-T model id or local directory used by --layout-backend grounding_dino_t.",
    )
    parser.add_argument(
        "--grounding-dino-text-threshold",
        type=float,
        default=0.25,
        help="Text threshold for Grounding DINO-T grounded object detection post-processing.",
    )
    parser.add_argument(
        "--grounding-dino-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load Grounding DINO-T from local Hugging Face cache only. Keep enabled for "
            "reproducible offline preprocessing after downloading the checkpoint once."
        ),
    )
    parser.add_argument(
        "--sam3-checkpoint-path",
        default=None,
        help=(
            "Optional local SAM3 image checkpoint path. If omitted, the script "
            "checks SAM3_CHECKPOINT_PATH and common local paths such as "
            "third_party/sam3/checkpoints/sam3.pt."
        ),
    )
    parser.add_argument(
        "--sam3-load-from-hf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow SAM3 to download its checkpoint from Hugging Face when no local "
            "checkpoint is found. Disabled by default because facebook/sam3 is gated."
        ),
    )
    parser.add_argument(
        "--sam3-compile",
        action="store_true",
        help="Enable torch.compile for SAM3 image model. Usually leave disabled for preprocessing startup latency.",
    )
    parser.add_argument("--motion-backend", choices=["cotracker3", "opencv", "none"], default="cotracker3")
    parser.add_argument(
        "--cotracker-repo-or-dir",
        default="facebookresearch/co-tracker",
        help="Torch Hub repo or local co-tracker checkout.",
    )
    parser.add_argument(
        "--vis-num-demos-per-task",
        type=int,
        default=1,
        help="Save SSI preprocessing visualizations for the first N demos of each task. Use 0 to disable.",
    )
    parser.add_argument(
        "--vis-output-dir",
        default=None,
        help="Visualization output directory. Default: <output-root>/visualizations.",
    )
    parser.add_argument("--vis-fps", type=int, default=10, help="FPS for visualization videos.")
    parser.add_argument(
        "--vis-max-frames",
        type=int,
        default=48,
        help="Maximum labeled frames to render per visualization. Use 0 to render all.",
    )
    parser.add_argument(
        "--vis-max-tracks",
        type=int,
        default=128,
        help="Maximum trajectories to draw in trajectory visualizations.",
    )
    parser.add_argument(
        "--vis-bbox-threshold",
        type=float,
        default=0.20,
        help="Role-grid threshold for drawing bbox visualizations.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.grid_size < 1:
        raise ValueError("--grid-size must be a positive integer")
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

    print(f"[SSI] data_root={data_root}")
    print(f"[SSI] output_root={output_root}")
    print(f"[SSI] suites={[p.name for p in suite_roots]}")
    print(f"[SSI] selected_episodes={len(episodes)}")
    print(
        "[SSI] backends="
        f"depth:{args.depth_backend}, layout:{args.layout_backend}, motion:{args.motion_backend}"
    )
    print(f"[SSI] visualizations: first {args.vis_num_demos_per_task} demos/task -> {visualization_root}")

    depth_annotator = DepthAnnotator(
        args.depth_backend,
        model_id=args.da3_model_id,
        device=args.device,
        grid_size=args.grid_size,
    )
    layout_annotator = LayoutAnnotator(
        args.layout_backend,
        device=args.device,
        grid_size=args.grid_size,
        score_threshold=args.sam_score_threshold,
        grounding_dino_model_id=args.grounding_dino_model_id,
        grounding_dino_text_threshold=args.grounding_dino_text_threshold,
        grounding_dino_local_files_only=args.grounding_dino_local_files_only,
        sam3_checkpoint_path=args.sam3_checkpoint_path,
        sam3_load_from_hf=args.sam3_load_from_hf,
        sam3_compile=args.sam3_compile,
    )
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
            depth_annotator=depth_annotator,
            layout_annotator=layout_annotator,
            motion_annotator=motion_annotator,
            overwrite=args.overwrite,
            save_visualization=save_visualization,
            visualization_root=visualization_root,
            visualization_fps=args.vis_fps,
            visualization_max_frames=args.vis_max_frames,
            visualization_max_tracks=args.vis_max_tracks,
            visualization_bbox_threshold=args.vis_bbox_threshold,
        )
        write_manifest_row(manifest_path, row)

    print(f"[SSI] done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
