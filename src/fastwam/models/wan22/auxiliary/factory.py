"""Construction of independently configurable auxiliary DiT experts."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .bbox_branch import BBoxDiTBranch
from .depth_branch import DepthDiTBranch
from .mask_branch import MaskDiTBranch
from .trajectory_branch import TrajectoryDiTBranch


BRANCH_CLASSES = {
    "depth": DepthDiTBranch,
    "bbox": BBoxDiTBranch,
    "mask": MaskDiTBranch,
    "trajectory": TrajectoryDiTBranch,
}


def build_auxiliary_branches(
    config: dict[str, Any] | None,
    *,
    video_expert: nn.Module,
) -> tuple[dict[str, nn.Module], dict[str, Any]]:
    config = dict(config or {})
    if not bool(config.get("enabled", False)):
        return {}, config
    common_cfg = dict(config.get("common", {}))
    # These keys configure MoT visibility, not branch construction.
    common_cfg.pop("conditioning_camera_indices", None)
    common_cfg.pop("conditioning_num_cameras", None)
    common_defaults = {
        # A bottleneck hidden width keeps four independent 30-layer experts
        # tractable; q/k/v still use Video-DiT's exact head layout for MoT.
        "hidden_dim": int(common_cfg.pop("hidden_dim", 256) or video_expert.hidden_dim),
        "ffn_dim": int(common_cfg.pop("ffn_dim", 1024)),
        "text_dim": int(common_cfg.pop("text_dim", getattr(video_expert, "text_dim", 4096))),
        "freq_dim": int(common_cfg.pop("freq_dim", video_expert.freq_dim)),
        "eps": float(common_cfg.pop("eps", 1e-6)),
        "num_heads": int(video_expert.num_heads),
        "attn_head_dim": int(video_expert.attn_head_dim),
        "num_layers": int(len(video_expert.blocks)),
        "max_frames": int(common_cfg.pop("max_frames", 64)),
        "max_tokens": int(common_cfg.pop("max_tokens", 4096)),
        "use_gradient_checkpointing": bool(
            common_cfg.pop("use_gradient_checkpointing", getattr(video_expert, "use_gradient_checkpointing", False))
        ),
    }
    if common_cfg:
        raise ValueError(f"Unknown auxiliary.common keys: {sorted(common_cfg)}")
    branches: dict[str, nn.Module] = {}
    for name, branch_cls in BRANCH_CLASSES.items():
        branch_cfg = dict(config.get(name, {}))
        enabled = bool(branch_cfg.pop("enabled", False))
        if not enabled:
            continue
        for loss_key in {
            "alpha_grad",
            "beta_l1",
            "beta_giou",
            "beta_dice",
            "beta_vis",
            "use_atm_track_transformer",
            "conditioning_camera_indices",
            "conditioning_num_cameras",
        }:
            branch_cfg.pop(loss_key, None)
        init_from_video = bool(branch_cfg.pop("init_from_video_dit", False))
        kwargs = dict(common_defaults)
        for key in ("hidden_dim", "ffn_dim", "text_dim", "freq_dim", "eps", "max_frames", "max_tokens", "use_gradient_checkpointing"):
            if key in branch_cfg:
                value = branch_cfg.pop(key)
                if value is not None:
                    kwargs[key] = value
        if name == "depth":
            # The depth head consumes final clean-frame Video tokens directly
            # while keeping its MoT sequence at one query per output frame.
            branch_cfg.pop("video_hidden_dim", None)
            kwargs["video_hidden_dim"] = int(video_expert.hidden_dim)
        branch = branch_cls(**kwargs, **branch_cfg)
        if init_from_video:
            branch.initialize_backbone_from_video(video_expert)
        branches[name] = branch
    return branches, config
