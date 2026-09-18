from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from fastwam.losses.spatial_auxiliary import mask_hungarian_loss, prepare_trajectory_targets
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.auxiliary.factory import build_auxiliary_branches
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.trainer import Wan22Trainer


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


def test_trainer_dit_only_mode_computes_auxiliary_losses():
    model = build_tiny_model(["depth", "bbox", "mask", "trajectory"])
    Wan22Trainer._apply_dit_only_train_mode(model)
    assert not model.training
    assert model.dit.training

    model.loss_lambda_video = 0.0
    model.loss_lambda_action = 0.0
    loss, metrics = model.training_loss(make_sample())
    assert loss > 0
    assert all(metrics[f"loss_{name}"] > 0 for name in model.auxiliary_branch_names)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.video_expert.parameters()
    )


def test_auxiliary_attention_reads_agentview_video_tokens_only():
    model = build_tiny_model(["depth"])
    model.auxiliary_config["common"]["conditioning_camera_indices"] = [0]
    model.auxiliary_config["common"]["conditioning_num_cameras"] = 2
    mask = model._build_training_attention_mask(
        seq_lens={"video": 8, "action": 2, "depth": 2},
        video_tokens_per_frame=8,
        video_grid_size=(1, 2, 4),
        device=torch.device("cpu"),
    )
    action_to_video = mask[8:10, :8]
    depth_to_video = mask[10:12, :8]
    assert bool(action_to_video.all())
    torch.testing.assert_close(
        depth_to_video,
        torch.tensor(
            [
                [True, True, False, False, True, True, False, False],
                [True, True, False, False, True, True, False, False],
            ]
        ),
    )


def test_depth_spatial_decoder_keeps_one_token_per_frame_and_reads_video_grid():
    model = build_tiny_model(["depth"])
    branch = model.get_auxiliary_branch("depth")
    batch_size = 2
    num_frames = 5
    state = branch.pre_dit(
        batch_size=batch_size,
        num_frames=num_frames,
        context=torch.randn(batch_size, 3, 12),
        context_mask=torch.ones(batch_size, 3, dtype=torch.bool),
    )
    assert state["tokens"].shape == (batch_size, num_frames, branch.hidden_dim)

    video_spatial_tokens = torch.randn(
        batch_size, 2, 3, model.video_expert.hidden_dim, requires_grad=True
    )
    prediction = branch.post_dit(
        state["tokens"],
        state,
        video_spatial_tokens=video_spatial_tokens,
    )
    assert prediction.shape == (batch_size, num_frames, 1, 8, 8)
    prediction.square().mean().backward()
    assert video_spatial_tokens.grad is not None
    assert torch.count_nonzero(video_spatial_tokens.grad) > 0


def test_depth_spatial_decoder_selects_configured_first_frame_camera_grid():
    model = build_tiny_model(["depth"])
    model.auxiliary_config["common"]["conditioning_num_cameras"] = 2
    model.auxiliary_config["depth"]["conditioning_camera_indices"] = [0]
    # Two frames of a 2x4 token grid. Values 8..15 belong to the second frame
    # and must never enter the clean first-frame depth decoder condition.
    video_tokens = torch.arange(16, dtype=torch.float32).view(1, 16, 1)
    selected = model._select_auxiliary_first_frame_video_tokens(
        "depth", video_tokens, (2, 2, 4)
    )
    assert selected.shape == (1, 2, 2, 1)
    torch.testing.assert_close(
        selected[0, ..., 0],
        torch.tensor([[0.0, 1.0], [4.0, 5.0]]),
    )


def test_auxiliary_only_step_changes_action_through_video_expert():
    model = build_tiny_model(["depth"])
    image = torch.randn(3, 16, 16)
    context = torch.randn(1, 3, 12)
    context_mask = torch.ones(1, 3, dtype=torch.bool)
    inference_kwargs = {
        "prompt": None,
        "input_image": image,
        "action_horizon": 4,
        "context": context,
        "context_mask": context_mask,
        "num_inference_steps": 1,
        "seed": 7,
    }
    before_action = model.infer_action(**inference_kwargs)["action"]
    before_action_parameters = {
        name: parameter.detach().clone() for name, parameter in model.action_expert.named_parameters()
    }
    before_video_parameters = {
        name: parameter.detach().clone() for name, parameter in model.video_expert.named_parameters()
    }

    Wan22Trainer._apply_dit_only_train_mode(model)
    model.loss_lambda_video = 0.0
    model.loss_lambda_action = 0.0
    optimizer = torch.optim.SGD(model.dit.parameters(), lr=0.05)
    loss, _metrics = model.training_loss(make_sample())
    loss.backward()
    optimizer.step()

    assert all(
        torch.equal(parameter, before_action_parameters[name])
        for name, parameter in model.action_expert.named_parameters()
    )
    assert any(
        not torch.equal(parameter, before_video_parameters[name])
        for name, parameter in model.video_expert.named_parameters()
    )
    after_action = model.infer_action(**inference_kwargs)["action"]
    assert not torch.equal(before_action, after_action)


def test_auxiliary_configuration_validation():
    model = build_tiny_model(["depth"])
    enabled_loader = SimpleNamespace(
        enabled=True,
        load_depth=True,
        load_bbox=False,
        load_mask=False,
        load_trajectory=False,
    )
    Wan22Trainer._validate_auxiliary_configuration(
        model, SimpleNamespace(auxiliary_label_loader=enabled_loader)
    )
    with pytest.raises(ValueError, match="labels are disabled"):
        Wan22Trainer._validate_auxiliary_configuration(
            model, SimpleNamespace(auxiliary_label_loader=SimpleNamespace(enabled=False))
        )
    with pytest.raises(ValueError, match="no auxiliary branches"):
        Wan22Trainer._validate_auxiliary_configuration(
            build_tiny_model([]), SimpleNamespace(auxiliary_label_loader=enabled_loader)
        )
    master_only = build_tiny_model([])
    master_only.auxiliary_config["enabled"] = True
    with pytest.raises(ValueError, match="no auxiliary branch is enabled"):
        Wan22Trainer._validate_auxiliary_configuration(
            master_only,
            SimpleNamespace(auxiliary_label_loader=SimpleNamespace(enabled=False)),
        )


def test_unmatched_mask_queries_receive_background_gradient():
    prediction = torch.full((1, 1, 2, 4, 4), -5.0)
    prediction[0, 0, 0, 1:3, 1:3] = 5.0
    prediction[0, 0, 1] = 5.0
    prediction.requires_grad_()
    target = torch.zeros(1, 4, 4)
    target[0, 1:3, 1:3] = 1.0

    mask_hungarian_loss(prediction, [[target]]).backward()
    assert torch.count_nonzero(prediction.grad[0, 0, 1]) > 0


def test_trajectory_selection_uses_camera_local_queries():
    global_queries = torch.tensor([[[0.25, 0.5], [0.49, 0.5], [0.75, 0.5], [0.51, 0.5]]])
    local_queries = torch.tensor([[[0.5, 0.5], [0.98, 0.5], [0.5, 0.5], [0.02, 0.5]]])
    trajectories = global_queries.unsqueeze(2).expand(-1, -1, 2, -1).clone()
    targets = prepare_trajectory_targets(
        {
            "trajectories": trajectories,
            "traj_visibility": torch.ones(1, 4, 2),
            "traj_query_points": global_queries,
            "traj_query_points_local": local_queries,
            "traj_camera_indices": torch.tensor([[0, 0, 1, 1]]),
        },
        num_points=2,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(
        targets["query_points"][0], torch.tensor([[0.25, 0.5], [0.75, 0.5]])
    )


def test_trajectory_atm_random_sampling_and_temporal_patch_tokens():
    model = build_tiny_model(["trajectory"])
    branch = model.get_auxiliary_branch("trajectory")
    sample = make_sample()
    targets = prepare_trajectory_targets(
        sample,
        num_points=branch.num_points,
        device=torch.device("cpu"),
        random_sample=True,
    )
    assert targets["query_points"].shape == (1, branch.num_points, 2)
    assert bool(targets["point_valid"].all())

    state = branch.pre_dit(
        query_points=targets["query_points"],
        num_frames=sample["trajectories"].shape[2],
        context=torch.randn(1, 3, 12),
        context_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    expected_time_patches = math.ceil(
        sample["trajectories"].shape[2] / branch.track_patch_size
    )
    assert state["tokens"].shape[1] == branch.num_points * expected_time_patches


def test_eval_can_explicitly_return_all_auxiliary_decoder_outputs():
    model = build_tiny_model(["depth", "bbox", "mask", "trajectory"]).eval()
    loss, metrics, outputs = model.training_loss(
        make_sample(),
        compute_auxiliary=True,
        return_auxiliary_outputs=True,
    )
    assert torch.isfinite(loss)
    assert set(outputs) == {"depth", "bbox", "mask", "trajectory"}
    assert all(metrics[f"loss_{name}"] > 0 for name in outputs)


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
        base_model = build_tiny_model(["depth", "bbox", "mask", "trajectory"])
        Wan22Trainer._apply_dit_only_train_mode(base_model)
        model = DistributedDataParallel(base_model)
        for _ in range(2):
            loss, metrics = model(make_sample())
            assert all(metrics[f"loss_{name}"] > 0 for name in base_model.auxiliary_branch_names)
            loss.backward()
            assert torch.isfinite(loss)
            model.zero_grad(set_to_none=True)
    finally:
        dist.destroy_process_group()
