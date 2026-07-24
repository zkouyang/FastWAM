from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..wan_video_dit import sinusoidal_embedding_1d
from .base_video_dit_branch import AuxiliaryVideoDiTBranch


class TrajectoryDiTBranch(AuxiliaryVideoDiTBranch):
    """ATM Track-Transformer tokenization on top of the existing DiT expert.

    ``num_points`` follows ATM's per-view convention.  The data target builder
    samples this many first-frame-visible points independently for each enabled
    camera.  Point identities intentionally have no learned embeddings: the
    query coordinate supplies spatial identity and temporal patch embeddings
    are shared across the unordered point set.
    """

    def __init__(
        self,
        *,
        num_points: int = 32,
        horizon: Optional[int] = None,
        track_patch_size: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # This branch uses ATM's temporal patch embedding below instead of the
        # generic per-frame embedding registered by the auxiliary base class.
        # Remove it so DDP does not see a permanently unused trainable tensor.
        del self.frame_embedding
        self.num_points = int(num_points)
        self.horizon = None if horizon is None else int(horizon)
        self.track_patch_size = int(track_patch_size)
        if self.num_points <= 0:
            raise ValueError("num_points must be positive")
        if self.track_patch_size <= 0:
            raise ValueError("track_patch_size must be positive")

        # Direct adaptation of ATM TrackPatchEmbed: future coordinates are
        # replaced with the first-frame query and temporally patchified by a
        # strided Conv1d.
        self.track_patch_encoder = nn.Conv1d(
            2,
            self.hidden_dim,
            kernel_size=self.track_patch_size,
            stride=self.track_patch_size,
        )
        self.track_time_embedding = nn.Parameter(
            torch.zeros(1, self.max_frames, 1, self.hidden_dim)
        )
        with torch.no_grad():
            positions = torch.arange(self.max_frames, dtype=torch.float32)
            temporal = sinusoidal_embedding_1d(self.hidden_dim, positions)
            self.track_time_embedding.copy_(temporal.view(1, self.max_frames, 1, self.hidden_dim))
        self.coordinate_head = nn.Linear(
            self.hidden_dim, 2 * self.track_patch_size
        )
        self.visibility_head = nn.Linear(self.hidden_dim, self.track_patch_size)

    def pre_dit(
        self,
        *,
        query_points: torch.Tensor,
        num_frames: int,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ):
        if query_points.ndim != 3 or query_points.shape[-1] != 2:
            raise ValueError("query_points must be [B,N,2]")
        if self.horizon is not None and num_frames != self.horizon:
            raise ValueError(f"trajectory horizon {num_frames} != configured {self.horizon}")
        if num_frames <= 0:
            raise ValueError("trajectory num_frames must be positive")

        batch_size, num_tracks, _ = query_points.shape
        padded_frames = int(math.ceil(num_frames / self.track_patch_size)) * self.track_patch_size
        masked_tracks = query_points[:, None].expand(-1, padded_frames, -1, -1)
        masked_tracks = masked_tracks.permute(0, 2, 3, 1).reshape(
            batch_size * num_tracks, 2, padded_frames
        )
        tokens = self.track_patch_encoder(masked_tracks)
        token_frames = int(tokens.shape[-1])
        if token_frames > self.max_frames:
            raise ValueError(
                f"trajectory token frames {token_frames} exceed max_frames={self.max_frames}"
            )
        tokens = tokens.view(
            batch_size, num_tracks, self.hidden_dim, token_frames
        ).permute(0, 3, 1, 2)
        tokens = tokens + self.track_time_embedding[:, :token_frames].to(
            device=tokens.device, dtype=tokens.dtype
        )
        tokens = tokens.flatten(1, 2)

        # All points in one temporal patch share the same RoPE position.  This
        # retains ATM's permutation equivariance over the point dimension.
        rope_position_ids = torch.arange(token_frames, device=tokens.device).repeat_interleave(
            num_tracks
        )
        return self._prepare_tokens(
            tokens,
            num_frames=token_frames,
            tokens_per_frame=num_tracks,
            context=context,
            context_mask=context_mask,
            add_frame_embedding=False,
            rope_position_ids=rope_position_ids,
            meta={
                "trajectory_horizon": int(num_frames),
                "padded_horizon": padded_frames,
                "num_tracks": num_tracks,
                "track_patch_size": self.track_patch_size,
            },
        )

    def post_dit(self, tokens: torch.Tensor, pre_state):
        b = tokens.shape[0]
        token_frames = pre_state["meta"]["num_frames"]
        num_tracks = pre_state["meta"]["num_tracks"]
        horizon = pre_state["meta"]["trajectory_horizon"]
        patch_size = pre_state["meta"]["track_patch_size"]
        x = tokens.view(b, token_frames, num_tracks, self.hidden_dim)
        coords = self.coordinate_head(x).view(
            b, token_frames, num_tracks, patch_size, 2
        )
        coords = coords.permute(0, 2, 1, 3, 4).reshape(
            b, num_tracks, token_frames * patch_size, 2
        )[:, :, :horizon]
        visibility = self.visibility_head(x).permute(0, 2, 1, 3).reshape(
            b, num_tracks, token_frames * patch_size
        )[:, :, :horizon]
        return {"pred_coords": coords, "pred_visibility": visibility}
