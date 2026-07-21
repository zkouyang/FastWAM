"""Training-time readers for precomputed LIBERO spatial supervision.

The offline annotation scripts write one compressed cache per episode.  This
module is the single place where those episode timelines are sliced, aligned
with a Fast-WAM video window, and transformed into the spatial coordinate
system used by the returned RGB video.

No teacher model is imported here.  Auxiliary labels are targets only.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate


_INSTANCE_KEYS = {
    "boxes",
    "box_labels",
    "box_scores",
    "box_camera_indices",
    "masks",
}
_TRAJECTORY_KEYS = {
    "trajectories",
    "traj_visibility",
    "traj_query_points",
    "traj_point_source",
    "traj_camera_indices",
}


class AuxiliaryLabelLoadingError(RuntimeError):
    """A sample cannot be paired with its required auxiliary targets."""


def _to_plain_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(config, DictConfig):
            return dict(OmegaConf.to_container(config, resolve=True))
    except ImportError:
        pass
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError(f"auxiliary_labels must be a mapping, got {type(config)}")


def _scalar_int(value: Any, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar, got shape {tuple(value.shape)}")
        return int(value.item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"{name} must be scalar, got shape {value.shape}")
        return int(value.item())
    return int(value)


class _EpisodeCache:
    """Small per-worker LRU for decompressed episode arrays."""

    def __init__(self, max_episodes: int):
        if max_episodes < 0:
            raise ValueError(f"cache_size must be non-negative, got {max_episodes}")
        self.max_episodes = max_episodes
        self._items: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def load(
        self, path: Path, *, keys: set[str] | None = None
    ) -> dict[str, np.ndarray]:
        cached = self._items.pop(path, None)
        if cached is not None:
            self._items[path] = cached
            return cached

        if not path.is_file():
            raise FileNotFoundError(f"Missing auxiliary label cache: {path}")
        with np.load(path, allow_pickle=False) as archive:
            selected = archive.files if keys is None else [key for key in archive.files if key in keys]
            arrays = {key: archive[key] for key in selected}

        if self.max_episodes > 0:
            self._items[path] = arrays
            while len(self._items) > self.max_episodes:
                self._items.popitem(last=False)
        return arrays


class LiberoAuxiliaryLabelLoader:
    """Load aligned depth, bbox, mask, and trajectory targets for one sample.

    Returned conventions:

    * ``depth`` and ``depth_confidence``: ``[T, 1, H, W]``.
    * ``boxes``: a length-``T`` list of normalized ``cx,cy,w,h`` tensors.
    * ``masks``: a length-``T`` list of ``[N_t,H,W]`` binary tensors.  Its
      instance order is identical to ``boxes`` and ``box_labels``.
    * ``trajectories``: ``[N,T,2]`` normalized xy coordinates.
    * ``traj_query_points``: trajectory coordinates at the first window frame.
    """

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | Any,
        dataset_dirs: Sequence[str],
        camera_keys: Sequence[str],
        video_size: Sequence[int],
        concat_multi_camera: str | None,
    ):
        cfg = _to_plain_dict(config)
        self.enabled = bool(cfg.get("enabled", False))
        self.load_depth = bool(cfg.get("load_depth", True))
        self.load_bbox = bool(cfg.get("load_bbox", True))
        self.load_mask = bool(cfg.get("load_mask", True))
        self.load_trajectory = bool(cfg.get("load_trajectory", True))
        self.dataset_dirs = [Path(path) for path in dataset_dirs]
        self.suite_names = [path.name for path in self.dataset_dirs]
        self.camera_keys = list(camera_keys)
        self.target_hw = (int(video_size[0]), int(video_size[1]))
        self.concat_multi_camera = concat_multi_camera

        if not self.enabled:
            return
        if not any((self.load_depth, self.load_bbox, self.load_mask, self.load_trajectory)):
            raise ValueError("auxiliary_labels.enabled=true but every modality is disabled")
        if len(self.camera_keys) > 1 and self.concat_multi_camera not in {"horizontal", "vertical"}:
            raise ValueError(
                "LIBERO auxiliary labels currently support horizontal or vertical multi-camera "
                f"composition, got {self.concat_multi_camera!r}"
            )

        self.depth_root = self._required_root(cfg, "depth_cache_dir", self.load_depth)
        self.bbox_root = self._required_root(
            cfg, "bbox_cache_dir", self.load_bbox or self.load_mask
        )
        self.trajectory_root = self._required_root(
            cfg, "trajectory_cache_dir", self.load_trajectory
        )
        cache_size = int(cfg.get("cache_size", 1))
        # Keep independent LRUs: loading bbox after depth must not immediately
        # evict the same episode's depth archive.
        self._depth_cache = _EpisodeCache(cache_size)
        self._bbox_cache = _EpisodeCache(cache_size)
        self._trajectory_cache = _EpisodeCache(cache_size)

    @staticmethod
    def _required_root(cfg: dict[str, Any], key: str, required: bool) -> Path | None:
        value = cfg.get(key)
        if not required:
            return Path(value) if value else None
        if not value:
            raise ValueError(f"auxiliary_labels.{key} is required for the enabled modality")
        root = Path(value)
        if not root.is_dir():
            raise FileNotFoundError(f"Auxiliary cache directory does not exist: {root}")
        return root

    def _episode_path(self, root: Path, suite: str, episode: int, suffix: str) -> Path:
        return root / suite / f"episode_{episode:06d}.{suffix}.npz"

    def _validate_identity(
        self,
        arrays: dict[str, np.ndarray],
        *,
        path: Path,
        suite: str,
        episode: int,
    ) -> None:
        if "meta_json" not in arrays:
            raise ValueError(f"Cache is missing meta_json: {path}")
        import json

        meta = json.loads(str(arrays["meta_json"].item()))
        if meta.get("suite") != suite or int(meta.get("episode_index", -1)) != episode:
            raise ValueError(
                f"Cache identity mismatch in {path}: expected ({suite}, {episode}), "
                f"got ({meta.get('suite')}, {meta.get('episode_index')})"
            )

    def _camera_positions(self, arrays: dict[str, np.ndarray], path: Path) -> list[int]:
        stored = [str(key) for key in arrays["camera_keys"].tolist()]
        missing = [key for key in self.camera_keys if key not in stored]
        if missing:
            raise ValueError(f"Cache {path} is missing cameras {missing}; stored cameras are {stored}")
        return [stored.index(key) for key in self.camera_keys]

    @staticmethod
    def _frame_positions(
        arrays: dict[str, np.ndarray], requested: np.ndarray, path: Path
    ) -> np.ndarray:
        available = np.asarray(arrays["frame_indices"], dtype=np.int64)
        if available.ndim != 1 or len(available) == 0:
            raise ValueError(f"Invalid frame_indices in {path}: shape {available.shape}")
        positions = np.searchsorted(available, requested)
        valid = positions < len(available)
        exact = np.zeros_like(valid)
        exact[valid] = available[positions[valid]] == requested[valid]
        if not bool(exact.all()):
            missing = requested[~exact].tolist()
            raise ValueError(
                f"Cache {path} has no exact labels for RGB frame indices {missing}. "
                "Use label frame_stride=1 (recommended) or a sampler aligned to the label stride."
            )
        return positions.astype(np.int64)

    def _layout(self, height: int, width: int) -> tuple[int, int, list[tuple[int, int]]]:
        num_cameras = len(self.camera_keys)
        if num_cameras == 1:
            return height, width, [(0, 0)]
        if self.concat_multi_camera == "horizontal":
            return height, width * num_cameras, [(0, i * width) for i in range(num_cameras)]
        if self.concat_multi_camera == "vertical":
            return height * num_cameras, width, [(i * height, 0) for i in range(num_cameras)]
        raise AssertionError("multi-camera layout was validated in __init__")

    def _geometry(self, source_hw: tuple[int, int]) -> tuple[int, int, int, int, float, float]:
        source_h, source_w = source_hw
        target_h, target_w = self.target_hw
        scale = max(target_w / source_w, target_h / source_h)
        resized_h = int(scale * source_h + 0.5)
        resized_w = int(scale * source_w + 0.5)
        crop_top = int(round((resized_h - target_h) / 2.0))
        crop_left = int(round((resized_w - target_w) / 2.0))
        return resized_h, resized_w, crop_top, crop_left, resized_h / source_h, resized_w / source_w

    def _resize_crop(self, tensor: torch.Tensor, *, mode: str) -> torch.Tensor:
        source_hw = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        resized_h, resized_w, top, left, _sy, _sx = self._geometry(source_hw)
        kwargs = {"size": (resized_h, resized_w), "mode": mode}
        if mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        out = F.interpolate(tensor.float(), **kwargs)
        target_h, target_w = self.target_hw
        return out[..., top : top + target_h, left : left + target_w]

    def _load_depth_targets(
        self, suite: str, episode: int, requested: np.ndarray
    ) -> dict[str, torch.Tensor]:
        assert self.depth_root is not None
        path = self._episode_path(self.depth_root, suite, episode, "depth")
        arrays = self._depth_cache.load(
            path,
            keys={"frame_indices", "camera_keys", "depth", "depth_conf", "meta_json"},
        )
        self._validate_identity(arrays, path=path, suite=suite, episode=episode)
        cameras = self._camera_positions(arrays, path)
        frames = self._frame_positions(arrays, requested, path)
        depth_array = arrays["depth"]
        conf_array = arrays.get("depth_conf", np.ones_like(depth_array))
        depth = torch.from_numpy(depth_array[np.ix_(cameras, frames)].astype(np.float32))
        confidence = torch.from_numpy(conf_array[np.ix_(cameras, frames)].astype(np.float32))
        # [C,T,H,W] -> RGB camera composition -> [T,1,H_out,W_out]
        cat_dim = -1 if self.concat_multi_camera == "horizontal" else -2
        depth = torch.cat([depth[i] for i in range(len(cameras))], dim=cat_dim).unsqueeze(1)
        confidence = torch.cat([confidence[i] for i in range(len(cameras))], dim=cat_dim).unsqueeze(1)
        return {
            # Preserve the cache's depth convention (relative, inverse, or
            # metric).  Only confidence is intrinsically bounded to [0, 1].
            "depth": self._resize_crop(depth, mode="bilinear"),
            "depth_confidence": self._resize_crop(confidence, mode="bilinear").clamp_(0.0, 1.0),
        }

    def _load_instance_targets(
        self, suite: str, episode: int, requested: np.ndarray
    ) -> dict[str, list[torch.Tensor]]:
        assert self.bbox_root is not None
        path = self._episode_path(self.bbox_root, suite, episode, "bbox")
        bbox_keys = {
            "frame_indices",
            "camera_keys",
            "bbox_xyxy",
            "bbox_confidences",
            "bbox_offsets",
            "meta_json",
        }
        if self.load_mask:
            bbox_keys.add("bbox_masks")
        arrays = self._bbox_cache.load(path, keys=bbox_keys)
        self._validate_identity(arrays, path=path, suite=suite, episode=episode)
        cameras = self._camera_positions(arrays, path)
        frames = self._frame_positions(arrays, requested, path)

        offsets = arrays["bbox_offsets"]
        flat_boxes = arrays["bbox_xyxy"]
        flat_scores = arrays["bbox_confidences"]
        flat_masks = arrays.get("bbox_masks")
        if self.load_mask and flat_masks is None:
            raise ValueError(
                f"Mask loading is enabled but {path} has no bbox_masks. "
                "Generate caches with preprocess_libero_bbox.py --bbox-backend grounded_sam2."
            )

        # Teacher images have one common spatial size across cameras.
        if flat_masks is not None and flat_masks.ndim == 3:
            camera_h, camera_w = map(int, flat_masks.shape[-2:])
        else:
            import json

            meta = json.loads(str(arrays["meta_json"].item()))
            image_size = meta.get("image_size")
            if isinstance(image_size, Sequence) and not isinstance(image_size, str):
                camera_h, camera_w = int(image_size[0]), int(image_size[1])
            else:
                camera_h = camera_w = int(image_size)
        canvas_h, canvas_w, origins = self._layout(camera_h, camera_w)
        resized_h, resized_w, top, left, scale_y, scale_x = self._geometry((canvas_h, canvas_w))

        result: dict[str, list[torch.Tensor]] = {
            "boxes": [],
            "box_labels": [],
            "box_scores": [],
            "box_camera_indices": [],
        }
        if self.load_mask:
            result["masks"] = []

        for frame_pos in frames:
            frame_boxes: list[np.ndarray] = []
            frame_scores: list[np.ndarray] = []
            frame_camera_ids: list[np.ndarray] = []
            frame_masks: list[torch.Tensor] = []
            for output_camera, stored_camera in enumerate(cameras):
                start = int(offsets[stored_camera, frame_pos])
                end = int(offsets[stored_camera, frame_pos + 1])
                boxes = np.asarray(flat_boxes[start:end], dtype=np.float32).copy()
                y_offset, x_offset = origins[output_camera]
                if len(boxes):
                    boxes[:, (0, 2)] += x_offset
                    boxes[:, (1, 3)] += y_offset
                    frame_boxes.append(boxes)
                    frame_scores.append(np.asarray(flat_scores[start:end], dtype=np.float32))
                    frame_camera_ids.append(
                        np.full((len(boxes),), output_camera, dtype=np.int64)
                    )
                if self.load_mask and end > start:
                    assert flat_masks is not None
                    masks = torch.from_numpy(flat_masks[start:end].astype(np.float32))
                    canvas = torch.zeros((len(masks), canvas_h, canvas_w), dtype=torch.float32)
                    canvas[:, y_offset : y_offset + camera_h, x_offset : x_offset + camera_w] = masks
                    frame_masks.append(canvas)

            if frame_boxes:
                boxes_xyxy = torch.from_numpy(np.concatenate(frame_boxes, axis=0))
                scores = torch.from_numpy(np.concatenate(frame_scores, axis=0))
                camera_ids = torch.from_numpy(np.concatenate(frame_camera_ids, axis=0))
                boxes_xyxy[:, (0, 2)] = boxes_xyxy[:, (0, 2)] * scale_x - left
                boxes_xyxy[:, (1, 3)] = boxes_xyxy[:, (1, 3)] * scale_y - top
                boxes_xyxy[:, (0, 2)].clamp_(0, self.target_hw[1])
                boxes_xyxy[:, (1, 3)].clamp_(0, self.target_hw[0])
                keep = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (
                    boxes_xyxy[:, 3] > boxes_xyxy[:, 1]
                )
                boxes_xyxy = boxes_xyxy[keep]
                scores = scores[keep]
                camera_ids = camera_ids[keep]
                cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5 / self.target_hw[1]
                cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5 / self.target_hw[0]
                width = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / self.target_hw[1]
                height = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / self.target_hw[0]
                boxes_cxcywh = torch.stack((cx, cy, width, height), dim=-1).clamp_(0.0, 1.0)
            else:
                keep = torch.zeros((0,), dtype=torch.bool)
                boxes_cxcywh = torch.zeros((0, 4), dtype=torch.float32)
                scores = torch.zeros((0,), dtype=torch.float32)
                camera_ids = torch.zeros((0,), dtype=torch.int64)

            result["boxes"].append(boxes_cxcywh)
            # Current LIBERO caches contain text prompts, not a stable class taxonomy.
            # Class zero therefore means "object" and the head uses C=1 objectness.
            result["box_labels"].append(torch.zeros(len(boxes_cxcywh), dtype=torch.int64))
            result["box_scores"].append(scores)
            result["box_camera_indices"].append(camera_ids)
            if self.load_mask:
                if frame_masks:
                    masks = torch.cat(frame_masks, dim=0).unsqueeze(1)
                    masks = self._resize_crop(masks, mode="nearest").squeeze(1)
                    masks = masks[keep].to(dtype=torch.float32)
                else:
                    masks = torch.zeros((0, *self.target_hw), dtype=torch.float32)
                result["masks"].append(masks)

        if not self.load_bbox:
            for key in ("boxes", "box_labels", "box_scores", "box_camera_indices"):
                result.pop(key)
        return result

    def _load_trajectory_targets(
        self, suite: str, episode: int, requested: np.ndarray
    ) -> dict[str, torch.Tensor]:
        assert self.trajectory_root is not None
        path = self._episode_path(self.trajectory_root, suite, episode, "motion")
        arrays = self._trajectory_cache.load(
            path,
            keys={
                "frame_indices",
                "camera_keys",
                "motion_points",
                "motion_visibility",
                "motion_point_source",
                "meta_json",
            },
        )
        self._validate_identity(arrays, path=path, suite=suite, episode=episode)
        cameras = self._camera_positions(arrays, path)
        frames = self._frame_positions(arrays, requested, path)
        points_array = arrays["motion_points"]
        visibility_array = arrays["motion_visibility"]
        points = torch.from_numpy(points_array[np.ix_(cameras, frames)].astype(np.float32))
        visibility = torch.from_numpy(
            visibility_array[np.ix_(cameras, frames)].astype(np.float32)
        )
        # [C,T,N,2]. Cache coordinates are per-camera normalized xy.
        num_cameras = len(cameras)
        if num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                for camera in range(num_cameras):
                    points[camera, ..., 0] = (points[camera, ..., 0] + camera) / num_cameras
            else:
                for camera in range(num_cameras):
                    points[camera, ..., 1] = (points[camera, ..., 1] + camera) / num_cameras

        # Apply the same final aspect-preserving resize and center crop as RGB.
        import json

        meta = json.loads(str(arrays["meta_json"].item()))
        image_size = meta.get("image_size")
        if isinstance(image_size, Sequence) and not isinstance(image_size, str):
            camera_h, camera_w = int(image_size[0]), int(image_size[1])
        else:
            camera_h = camera_w = int(image_size)
        canvas_h, canvas_w, _origins = self._layout(camera_h, camera_w)
        _rh, _rw, top, left, scale_y, scale_x = self._geometry((canvas_h, canvas_w))
        points[..., 0] = (
            points[..., 0] * canvas_w * scale_x - left
        ) / self.target_hw[1]
        points[..., 1] = (
            points[..., 1] * canvas_h * scale_y - top
        ) / self.target_hw[0]
        in_frame = (
            (points[..., 0] >= 0.0)
            & (points[..., 0] <= 1.0)
            & (points[..., 1] >= 0.0)
            & (points[..., 1] <= 1.0)
        )
        visibility = visibility * in_frame.to(dtype=visibility.dtype)
        points.clamp_(0.0, 1.0)

        # [C,T,N,*] -> [C,N,T,*] -> [C*N,T,*]
        points = points.permute(0, 2, 1, 3).reshape(-1, len(frames), 2)
        visibility = visibility.permute(0, 2, 1).reshape(-1, len(frames))
        source = torch.from_numpy(arrays["motion_point_source"][cameras].astype(np.int64)).reshape(-1)
        camera_ids = torch.arange(num_cameras, dtype=torch.int64).repeat_interleave(
            points.shape[0] // num_cameras
        )
        return {
            "trajectories": points,
            "traj_visibility": visibility,
            "traj_query_points": points[:, 0].clone(),
            "traj_point_source": source,
            "traj_camera_indices": camera_ids,
        }

    def load(
        self,
        *,
        dataset_index: Any,
        episode_index: Any,
        frame_indices: Sequence[int] | np.ndarray | torch.Tensor,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        dataset = _scalar_int(dataset_index, "dataset_index")
        episode = _scalar_int(episode_index, "episode_index")
        if dataset < 0 or dataset >= len(self.suite_names):
            raise IndexError(f"dataset_index {dataset} is outside [0, {len(self.suite_names)})")
        suite = self.suite_names[dataset]
        requested = np.asarray(torch.as_tensor(frame_indices, dtype=torch.int64).tolist(), dtype=np.int64)
        if requested.ndim != 1 or len(requested) == 0:
            raise ValueError(f"frame_indices must be a non-empty vector, got {requested.shape}")

        result: dict[str, Any] = {
            "aux_frame_indices": torch.from_numpy(requested.copy()),
        }
        if self.load_depth:
            result.update(self._load_depth_targets(suite, episode, requested))
        if self.load_bbox or self.load_mask:
            result.update(self._load_instance_targets(suite, episode, requested))
        if self.load_trajectory:
            result.update(self._load_trajectory_targets(suite, episode, requested))
        return result


def auxiliary_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate Fast-WAM samples while preserving ragged per-frame instances.

    BBox/mask fields stay as ``list[batch][time]``.  Trajectory point counts
    are padded across samples and ``traj_point_is_pad`` marks padded points.
    Every baseline field continues to use PyTorch's default collation.
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    keys = set(batch[0])
    if any(set(sample) != keys for sample in batch[1:]):
        raise ValueError("All samples in an auxiliary batch must contain the same keys")

    result: dict[str, Any] = {}
    for key in keys - _INSTANCE_KEYS - _TRAJECTORY_KEYS:
        result[key] = default_collate([sample[key] for sample in batch])

    for key in keys & _INSTANCE_KEYS:
        result[key] = [sample[key] for sample in batch]

    if "trajectories" in keys:
        counts = [int(sample["trajectories"].shape[0]) for sample in batch]
        max_points = max(counts)
        horizon = int(batch[0]["trajectories"].shape[1])
        if any(int(sample["trajectories"].shape[1]) != horizon for sample in batch):
            raise ValueError("Trajectory horizons must match within a batch")
        batch_size = len(batch)
        trajectories = torch.zeros((batch_size, max_points, horizon, 2), dtype=torch.float32)
        visibility = torch.zeros((batch_size, max_points, horizon), dtype=torch.float32)
        query_points = torch.zeros((batch_size, max_points, 2), dtype=torch.float32)
        point_source = torch.full((batch_size, max_points), -1, dtype=torch.int64)
        camera_indices = torch.full((batch_size, max_points), -1, dtype=torch.int64)
        point_is_pad = torch.ones((batch_size, max_points), dtype=torch.bool)
        for index, sample in enumerate(batch):
            count = counts[index]
            trajectories[index, :count] = sample["trajectories"]
            visibility[index, :count] = sample["traj_visibility"]
            query_points[index, :count] = sample["traj_query_points"]
            point_source[index, :count] = sample["traj_point_source"]
            camera_indices[index, :count] = sample["traj_camera_indices"]
            point_is_pad[index, :count] = False
        result.update(
            {
                "trajectories": trajectories,
                "traj_visibility": visibility,
                "traj_query_points": query_points,
                "traj_point_source": point_source,
                "traj_camera_indices": camera_indices,
                "traj_point_is_pad": point_is_pad,
            }
        )
    elif keys & _TRAJECTORY_KEYS:
        missing = sorted((keys & _TRAJECTORY_KEYS) - {"trajectories"})
        raise ValueError(f"Trajectory metadata exists without trajectories: {missing}")

    return result
