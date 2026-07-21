from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.auxiliary.factory import build_auxiliary_branches
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


class TinyVAE(nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 1

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(z_dim=3)

    def encode(self, video, **_kwargs):
        if isinstance(video, list):
            return [item[:, :1].clone() for item in video]
        return video[:, :, ::4].clone()

    def decode(self, latents, **_kwargs):
        return latents


def auxiliary_config(enabled_names=()):
    enabled_names = set(enabled_names)
    return {
        "enabled": bool(enabled_names),
        "compute_auxiliary": True,
        "common": {
            "hidden_dim": 16,
            "ffn_dim": 32,
            "max_frames": 8,
            "max_tokens": 256,
            "use_gradient_checkpointing": False,
        },
        "depth": {
            "enabled": "depth" in enabled_names,
            "output_size": [8, 8],
            "decoder_dim": 4,
            "decoder_grid": [2, 2],
        },
        "bbox": {
            "enabled": "bbox" in enabled_names,
            "num_queries": 2,
            "num_classes": 1,
            "beta_l1": 5.0,
            "beta_giou": 2.0,
        },
        "mask": {
            "enabled": "mask" in enabled_names,
            "num_queries": 2,
            "output_size": [8, 8],
            "mask_dim": 4,
            "feature_grid": [2, 2],
        },
        "trajectory": {
            "enabled": "trajectory" in enabled_names,
            "num_points": 4,
            "horizon": None,
        },
    }


def build_tiny_model(enabled_names=()) -> FastWAM:
    video = WanVideoDiT(
        hidden_dim=16,
        in_dim=3,
        ffn_dim=32,
        out_dim=3,
        text_dim=12,
        freq_dim=8,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        attn_head_dim=8,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
    )
    action = ActionDiT(
        hidden_dim=16,
        action_dim=3,
        ffn_dim=32,
        text_dim=12,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=1,
    )
    config = auxiliary_config(enabled_names)
    branches, config = build_auxiliary_branches(config, video_expert=video)
    mot = MoT({"video": video, "action": action, **branches}, mot_checkpoint_mixed_attn=False)
    return FastWAM(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=TinyVAE(),
        text_dim=12,
        device="cpu",
        torch_dtype=torch.float32,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
        auxiliary_config=config,
        auxiliary_loss_weights={name: 0.1 for name in enabled_names},
    )


def make_sample(batch_size=1):
    horizon = 5
    boxes = []
    masks = []
    labels = []
    for _ in range(batch_size):
        boxes.append([torch.tensor([[0.5, 0.5, 0.25, 0.25]]) for _ in range(horizon)])
        labels.append([torch.zeros(1, dtype=torch.long) for _ in range(horizon)])
        frame_masks = []
        for _ in range(horizon):
            mask = torch.zeros(1, 16, 16)
            mask[:, 6:10, 6:10] = 1
            frame_masks.append(mask)
        masks.append(frame_masks)
    trajectories = torch.rand(batch_size, 6, horizon, 2)
    visibility = torch.ones(batch_size, 6, horizon)
    return {
        "video": torch.randn(batch_size, 3, horizon, 16, 16),
        "context": torch.randn(batch_size, 3, 12),
        "context_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "action": torch.randn(batch_size, 4, 3),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
        "image_is_pad": torch.zeros(batch_size, horizon, dtype=torch.bool),
        "depth": torch.rand(batch_size, horizon, 1, 16, 16),
        "depth_confidence": torch.ones(batch_size, horizon, 1, 16, 16),
        "boxes": boxes,
        "box_labels": labels,
        "masks": masks,
        "trajectories": trajectories,
        "traj_visibility": visibility,
        "traj_query_points": trajectories[:, :, 0].clone(),
        "traj_camera_indices": torch.tensor([[0, 0, 0, 1, 1, 1]]).expand(batch_size, -1),
        "traj_point_is_pad": torch.zeros(batch_size, 6, dtype=torch.bool),
    }


@pytest.mark.parametrize("name", ["depth", "bbox", "mask", "trajectory"])
def test_each_branch_forward_backward_and_shared_video_gradient(name):
    model = build_tiny_model([name]).train()
    # Isolate the auxiliary objective: any Video gradient below must therefore
    # come through the clean-frame K/V path, not the base video/action losses.
    model.loss_lambda_video = 0.0
    model.loss_lambda_action = 0.0
    loss, metrics = model.training_loss(make_sample())
    assert torch.isfinite(loss)
    assert set(metrics) == {
        "loss_video", "loss_action", "loss_depth", "loss_bbox", "loss_mask",
        "loss_trajectory", "loss_total",
    }
    loss.backward()
    branch = model.get_auxiliary_branch(name)
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in branch.parameters())
    # Auxiliary query rows read clean-frame Video K/V, so their loss contributes
    # gradients to the same Video expert used by Action inference.
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.video_expert.parameters())


def test_full_loss_baseline_switch_amp_and_inference_skip(monkeypatch):
    names = ["depth", "bbox", "mask", "trajectory"]
    model = build_tiny_model(names).train()
    sample = make_sample()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, metrics = model.training_loss(sample)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())

    model.zero_grad(set_to_none=True)
    baseline_loss, baseline_metrics = model.training_loss(sample, compute_auxiliary=False)
    assert torch.isfinite(baseline_loss)
    assert all(baseline_metrics[f"loss_{name}"] == 0.0 for name in names)

    model.eval()
    for name in names:
        branch = model.get_auxiliary_branch(name)
        monkeypatch.setattr(
            branch,
            "pre_dit",
            lambda *args, _name=name, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"inference called {_name}")
            ),
        )
    output = model.infer_action(
        prompt=None,
        input_image=torch.randn(3, 16, 16),
        action_horizon=4,
        context=torch.randn(1, 3, 12),
        context_mask=torch.ones(1, 3, dtype=torch.bool),
        num_inference_steps=1,
        seed=0,
    )
    assert output["action"].shape == (4, 3)


def test_full_checkpoint_loads_into_aux_disabled_deployment(tmp_path: Path):
    full = build_tiny_model(["depth", "bbox", "mask", "trajectory"])
    checkpoint = tmp_path / "full.pt"
    full.save_checkpoint(checkpoint, step=3)
    deployment = build_tiny_model([])
    payload = deployment.load_checkpoint(checkpoint)
    assert payload["step"] == 3
    assert deployment.auxiliary_branch_names == ()


def test_mot_registered_parameters_cover_all_branch_heads():
    model = build_tiny_model(["depth", "bbox", "mask", "trajectory"])
    optimizer_ids = {id(parameter) for parameter in model.dit.parameters()}
    for name in model.auxiliary_branch_names:
        assert all(id(parameter) in optimizer_ids for parameter in model.get_auxiliary_branch(name).parameters())


def test_auxiliary_disabled_matches_baseline_exactly():
    baseline = build_tiny_model([]).train()
    full = build_tiny_model(["depth", "bbox", "mask", "trajectory"]).train()
    full.video_expert.load_state_dict(baseline.video_expert.state_dict())
    full.action_expert.load_state_dict(baseline.action_expert.state_dict())
    sample = make_sample()
    torch.manual_seed(123)
    baseline_loss, baseline_metrics = baseline.training_loss(sample)
    torch.manual_seed(123)
    switched_loss, switched_metrics = full.training_loss(sample, compute_auxiliary=False)
    torch.testing.assert_close(switched_loss, baseline_loss)
    assert switched_metrics["loss_video"] == pytest.approx(baseline_metrics["loss_video"])
    assert switched_metrics["loss_action"] == pytest.approx(baseline_metrics["loss_action"])


def test_single_process_ddp_full_backward(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    rendezvous = tmp_path / "ddp_init"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        model = DistributedDataParallel(
            build_tiny_model(["depth", "bbox", "mask", "trajectory"]).train()
        )
        loss, _metrics = model(make_sample())
        loss.backward()
        assert torch.isfinite(loss)
    finally:
        dist.destroy_process_group()
