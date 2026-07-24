import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fastwam.datasets.auxiliary_labels import (
    AuxiliaryLabelLoadingError,
    LiberoAuxiliaryLabelLoader,
    auxiliary_collate_fn,
)
from fastwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


class AuxiliaryLabelLoaderTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.suite = "libero_spatial_no_noops_lerobot"
        self.dataset_dir = root / "datasets" / self.suite
        self.dataset_dir.mkdir(parents=True)
        self.depth_root = root / "depth"
        self.bbox_root = root / "bbox"
        self.motion_root = root / "motion"
        for cache_root in (self.depth_root, self.bbox_root, self.motion_root):
            (cache_root / self.suite).mkdir(parents=True)

        cameras = np.asarray(
            ["observation.images.image", "observation.images.wrist_image"]
        )
        frames = np.arange(6, dtype=np.int32)
        identity = {
            "suite": self.suite,
            "episode_index": 0,
            "image_size": 2,
        }

        depth = np.zeros((2, 6, 2, 2), dtype=np.float16)
        for camera in range(2):
            for frame in range(6):
                depth[camera, frame] = camera * 0.1 + frame * 0.01
        np.savez_compressed(
            self.depth_root / self.suite / "episode_000000.depth.npz",
            frame_indices=frames,
            camera_keys=cameras,
            depth=depth,
            depth_conf=np.ones_like(depth),
            meta_json=np.asarray(json.dumps(identity)),
        )

        # One full-camera instance for every camera/frame. Offsets continue
        # globally across cameras, matching preprocess_libero_bbox.py.
        boxes = []
        scores = []
        masks = []
        offsets = np.zeros((2, 7), dtype=np.int64)
        cursor = 0
        for camera in range(2):
            offsets[camera, 0] = cursor
            for frame in range(6):
                boxes.append(np.asarray([[0, 0, 2, 2]], dtype=np.float32))
                scores.append(np.asarray([0.9 - camera * 0.1], dtype=np.float32))
                masks.append(np.ones((1, 2, 2), dtype=np.bool_))
                cursor += 1
                offsets[camera, frame + 1] = cursor
        np.savez_compressed(
            self.bbox_root / self.suite / "episode_000000.bbox.npz",
            frame_indices=frames,
            camera_keys=cameras,
            bbox_xyxy=np.concatenate(boxes),
            bbox_confidences=np.concatenate(scores),
            bbox_offsets=offsets,
            bbox_counts=np.ones((2, 6), dtype=np.int32),
            bbox_masks=np.concatenate(masks),
            meta_json=np.asarray(json.dumps(identity)),
        )

        points = np.full((2, 6, 2, 2), 0.5, dtype=np.float16)
        visibility = np.ones((2, 6, 2), dtype=np.float16)
        source = np.asarray([[0, 1], [0, 1]], dtype=np.int16)
        np.savez_compressed(
            self.motion_root / self.suite / "episode_000000.motion.npz",
            frame_indices=frames,
            camera_keys=cameras,
            motion_points=points,
            motion_visibility=visibility,
            motion_point_source=source,
            meta_json=np.asarray(json.dumps(identity)),
        )

        self.loader = LiberoAuxiliaryLabelLoader(
            config={
                "enabled": True,
                "camera_keys": {
                    "depth": cameras.tolist(),
                    "bbox": cameras.tolist(),
                    "mask": cameras.tolist(),
                    "trajectory": cameras.tolist(),
                },
                "load_depth": True,
                "load_bbox": True,
                "load_mask": True,
                "load_trajectory": True,
                "depth_cache_dir": str(self.depth_root),
                "bbox_cache_dir": str(self.bbox_root),
                "trajectory_cache_dir": str(self.motion_root),
                "cache_size": 1,
            },
            dataset_dirs=[str(self.dataset_dir)],
            camera_keys=cameras.tolist(),
            video_size=[2, 4],
            concat_multi_camera="horizontal",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_disabled_loader_does_not_touch_cache_paths(self):
        loader = LiberoAuxiliaryLabelLoader(
            config=None,
            dataset_dirs=[str(self.dataset_dir)],
            camera_keys=["observation.images.image"],
            video_size=[2, 2],
            concat_multi_camera=None,
        )
        self.assertEqual(
            loader.load(dataset_index=0, episode_index=999, frame_indices=[0]),
            {},
        )

    def test_loads_and_aligns_all_modalities(self):
        sample = self.loader.load(
            dataset_index=torch.tensor(0),
            episode_index=torch.tensor(0),
            frame_indices=torch.tensor([1, 3, 5]),
        )

        self.assertEqual(tuple(sample["depth"].shape), (3, 1, 2, 4))
        torch.testing.assert_close(
            sample["depth"][:, 0, 0, 0],
            torch.tensor([0.01, 0.03, 0.05]),
            atol=5e-5,
            rtol=0,
        )
        torch.testing.assert_close(
            sample["depth"][:, 0, 0, 3],
            torch.tensor([0.11, 0.13, 0.15]),
            atol=5e-5,
            rtol=0,
        )
        self.assertEqual(tuple(sample["depth_confidence"].shape), (3, 1, 2, 4))

        expected_boxes = torch.tensor(
            [[0.25, 0.5, 0.5, 1.0], [0.75, 0.5, 0.5, 1.0]]
        )
        self.assertEqual(len(sample["boxes"]), 3)
        for boxes, masks in zip(sample["boxes"], sample["masks"], strict=True):
            torch.testing.assert_close(boxes, expected_boxes)
            self.assertEqual(tuple(masks.shape), (2, 2, 4))
            self.assertTrue(bool((masks[0, :, :2] == 1).all()))
            self.assertTrue(bool((masks[0, :, 2:] == 0).all()))
            self.assertTrue(bool((masks[1, :, :2] == 0).all()))
            self.assertTrue(bool((masks[1, :, 2:] == 1).all()))

        self.assertEqual(tuple(sample["trajectories"].shape), (4, 3, 2))
        torch.testing.assert_close(sample["trajectories"][:2, :, 0], torch.full((2, 3), 0.25))
        torch.testing.assert_close(sample["trajectories"][2:, :, 0], torch.full((2, 3), 0.75))
        torch.testing.assert_close(sample["traj_query_points"], sample["trajectories"][:, 0])
        torch.testing.assert_close(sample["traj_query_points_local"], torch.full((4, 2), 0.5))
        torch.testing.assert_close(sample["aux_frame_indices"], torch.tensor([1, 3, 5]))

    def test_agentview_only_targets_use_single_view_canvas(self):
        loader = LiberoAuxiliaryLabelLoader(
            config={
                "enabled": True,
                "camera_keys": {
                    "depth": ["observation.images.image"],
                    "bbox": ["observation.images.image"],
                    "mask": ["observation.images.image"],
                    "trajectory": ["observation.images.image"],
                },
                "load_depth": True,
                "load_bbox": True,
                "load_mask": True,
                "load_trajectory": True,
                "depth_cache_dir": str(self.depth_root),
                "bbox_cache_dir": str(self.bbox_root),
                "trajectory_cache_dir": str(self.motion_root),
            },
            dataset_dirs=[str(self.dataset_dir)],
            camera_keys=[
                "observation.images.image",
                "observation.images.wrist_image",
            ],
            video_size=[2, 4],
            concat_multi_camera="horizontal",
        )
        sample = loader.load(dataset_index=0, episode_index=0, frame_indices=[1, 3, 5])

        self.assertEqual(tuple(sample["depth"].shape), (3, 1, 2, 2))
        self.assertTrue(bool((sample["depth_confidence"] == 1).all()))
        for boxes, masks, camera_ids in zip(
            sample["boxes"], sample["masks"], sample["box_camera_indices"], strict=True
        ):
            torch.testing.assert_close(boxes, torch.tensor([[0.5, 0.5, 1.0, 1.0]]))
            self.assertEqual(tuple(masks.shape), (1, 2, 2))
            self.assertTrue(bool((masks == 1).all()))
            torch.testing.assert_close(camera_ids, torch.zeros(1, dtype=torch.int64))
        self.assertEqual(tuple(sample["trajectories"].shape), (2, 3, 2))
        torch.testing.assert_close(sample["trajectories"][..., 0], torch.full((2, 3), 0.5))
        torch.testing.assert_close(sample["traj_query_points_local"], torch.full((2, 2), 0.5))
        torch.testing.assert_close(
            sample["traj_camera_indices"], torch.zeros(2, dtype=torch.int64)
        )

    def test_requires_exact_time_alignment(self):
        with self.assertRaisesRegex(ValueError, "no exact labels"):
            self.loader.load(dataset_index=0, episode_index=0, frame_indices=[1, 3, 6])

    def test_modalities_can_be_loaded_independently(self):
        cases = {
            "depth": ({"load_depth": True}, {"depth", "depth_confidence"}),
            "bbox": ({"load_bbox": True}, {"boxes", "box_labels", "box_scores"}),
            "mask": ({"load_mask": True}, {"masks"}),
            "trajectory": (
                {"load_trajectory": True},
                {
                    "trajectories",
                    "traj_visibility",
                    "traj_query_points",
                    "traj_query_points_local",
                },
            ),
        }
        roots = {
            "depth_cache_dir": str(self.depth_root),
            "bbox_cache_dir": str(self.bbox_root),
            "trajectory_cache_dir": str(self.motion_root),
        }
        modality_keys = {
            "depth",
            "depth_confidence",
            "boxes",
            "box_labels",
            "box_scores",
            "box_camera_indices",
            "masks",
            "trajectories",
            "traj_visibility",
            "traj_query_points",
            "traj_query_points_local",
            "traj_point_source",
            "traj_camera_indices",
        }
        for name, (enabled, expected) in cases.items():
            config = {
                "enabled": True,
                "load_depth": False,
                "load_bbox": False,
                "load_mask": False,
                "load_trajectory": False,
                **roots,
                **enabled,
            }
            loader = LiberoAuxiliaryLabelLoader(
                config=config,
                dataset_dirs=[str(self.dataset_dir)],
                camera_keys=[
                    "observation.images.image",
                    "observation.images.wrist_image",
                ],
                video_size=[2, 4],
                concat_multi_camera="horizontal",
            )
            sample = loader.load(dataset_index=0, episode_index=0, frame_indices=[1, 3, 5])
            present = set(sample) & modality_keys
            self.assertTrue(expected <= present, name)
            if name == "mask":
                self.assertNotIn("boxes", sample)
            self.assertFalse((present - expected) & {"depth", "boxes", "masks", "trajectories"}, name)

    def test_collator_keeps_instances_ragged_and_pads_trajectories(self):
        first = self.loader.load(dataset_index=0, episode_index=0, frame_indices=[1, 3, 5])
        second = dict(first)
        second["trajectories"] = first["trajectories"][:3]
        second["traj_visibility"] = first["traj_visibility"][:3]
        second["traj_query_points"] = first["traj_query_points"][:3]
        second["traj_query_points_local"] = first["traj_query_points_local"][:3]
        second["traj_point_source"] = first["traj_point_source"][:3]
        second["traj_camera_indices"] = first["traj_camera_indices"][:3]
        second["boxes"] = [value[:1] for value in first["boxes"]]
        second["box_labels"] = [value[:1] for value in first["box_labels"]]
        second["box_scores"] = [value[:1] for value in first["box_scores"]]
        second["box_camera_indices"] = [value[:1] for value in first["box_camera_indices"]]
        second["masks"] = [value[:1] for value in first["masks"]]

        batch = auxiliary_collate_fn([first, second])
        self.assertEqual(tuple(batch["trajectories"].shape), (2, 4, 3, 2))
        self.assertEqual(batch["boxes"][0][0].shape[0], 2)
        self.assertEqual(batch["boxes"][1][0].shape[0], 1)
        torch.testing.assert_close(
            batch["traj_point_is_pad"],
            torch.tensor([[False, False, False, False], [False, False, False, True]]),
        )


class _FakeLeRobotDataset:
    global_sample_stride = 1

    def __init__(self, sample):
        self.sample = sample

    def __getitem__(self, _index):
        return self.sample

    def __len__(self):
        return 1


class _RecordingAuxiliaryLoader:
    def __init__(self, enabled):
        self.enabled = enabled
        self.requested = None

    def load(self, *, dataset_index, episode_index, frame_indices):
        self.requested = torch.as_tensor(frame_indices).clone()
        return {"aux_frame_indices": self.requested.clone()}


class _FailingAuxiliaryLoader:
    enabled = True

    def load(self, **_kwargs):
        raise ValueError("missing exact label")


class RobotVideoDatasetIntegrationTest(unittest.TestCase):
    @staticmethod
    def _make_dataset(auxiliary_enabled: bool):
        image_is_pad = torch.zeros(33, dtype=torch.bool)
        image_is_pad[29:] = True
        sample = {
            "pixel_values": torch.zeros((2, 33, 3, 2, 2), dtype=torch.uint8),
            "image_is_pad": image_is_pad,
            "action": torch.zeros((32, 7), dtype=torch.float32),
            "proprio": torch.zeros((33, 8), dtype=torch.float32),
            "instruction": "move the object",
            "action_is_pad": torch.zeros(32, dtype=torch.bool),
            "proprio_is_pad": image_is_pad.clone(),
            "dataset_index": torch.tensor(0),
            "episode_index": torch.tensor(4),
            "frame_index": torch.tensor(10),
        }
        dataset = RobotVideoDataset.__new__(RobotVideoDataset)
        dataset.lerobot_dataset = _FakeLeRobotDataset(sample)
        dataset.num_frames = 33
        dataset.max_padding_retry = 0
        dataset.skip_padding_as_possible = False
        dataset.video_sample_indices = list(range(0, 33, 4))
        dataset.concat_multi_camera = "horizontal"
        dataset.video_size = [2, 4]
        dataset.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": 4, "img_h": 2}
        )
        dataset.crop_transform = CenterCrop(args={"img_w": 4, "img_h": 2})
        dataset.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})
        dataset.override_instruction = None
        dataset.auxiliary_label_loader = _RecordingAuxiliaryLoader(auxiliary_enabled)
        dataset._get_cached_text_context = lambda _prompt: (
            torch.zeros((2, 3), dtype=torch.float32),
            torch.ones(2, dtype=torch.bool),
        )
        return dataset

    def test_window_indices_match_rgb_and_episode_padding(self):
        dataset = self._make_dataset(auxiliary_enabled=True)
        sample = dataset._get(0)
        expected = torch.tensor([10, 14, 18, 22, 26, 30, 34, 38, 38])
        torch.testing.assert_close(dataset.auxiliary_label_loader.requested, expected)
        torch.testing.assert_close(sample["aux_frame_indices"], expected)
        self.assertEqual(tuple(sample["video"].shape), (3, 9, 2, 4))
        self.assertIn("episode_index", sample)

    def test_disabled_path_preserves_baseline_output_keys(self):
        dataset = self._make_dataset(auxiliary_enabled=False)
        sample = dataset._get(0)
        self.assertNotIn("aux_frame_indices", sample)
        self.assertNotIn("dataset_index", sample)
        self.assertNotIn("episode_index", sample)
        self.assertNotIn("frame_index", sample)

    def test_auxiliary_failure_is_not_hidden_by_random_retry(self):
        dataset = self._make_dataset(auxiliary_enabled=True)
        dataset.auxiliary_label_loader = _FailingAuxiliaryLoader()
        with self.assertRaises(AuxiliaryLabelLoadingError):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
