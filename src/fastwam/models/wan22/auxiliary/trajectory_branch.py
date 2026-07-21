from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base_video_dit_branch import AuxiliaryVideoDiTBranch, make_mlp


class TrajectoryDiTBranch(AuxiliaryVideoDiTBranch):
    """ATM-style query-point/time tokens with coordinate and visibility heads."""

    def __init__(self, *, num_points: int = 64, horizon: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.num_points = int(num_points)
        self.horizon = None if horizon is None else int(horizon)
        self.query_encoder = nn.Sequential(
            nn.Linear(2, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.point_embedding = nn.Parameter(
            torch.randn(1, self.num_points, self.hidden_dim) / self.hidden_dim**0.5
        )
        self.coordinate_head = make_mlp(self.hidden_dim, 2, layers=3)
        self.visibility_head = nn.Linear(self.hidden_dim, 1)

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
        if query_points.shape[1] != self.num_points:
            raise ValueError(f"expected {self.num_points} trajectory queries, got {query_points.shape[1]}")
        if self.horizon is not None and num_frames != self.horizon:
            raise ValueError(f"trajectory horizon {num_frames} != configured {self.horizon}")
        query = self.query_encoder(query_points) + self.point_embedding
        # ATM masks future track coordinates to the first-frame query.  Repeating
        # that query and adding temporal embeddings in the base realizes the
        # same input convention without exposing future targets.
        tokens = query.unsqueeze(1).expand(-1, num_frames, -1, -1).flatten(1, 2)
        return self._prepare_tokens(
            tokens,
            num_frames=num_frames,
            tokens_per_frame=self.num_points,
            context=context,
            context_mask=context_mask,
        )

    def post_dit(self, tokens: torch.Tensor, pre_state):
        b = tokens.shape[0]
        t = pre_state["meta"]["num_frames"]
        x = tokens.view(b, t, self.num_points, self.hidden_dim)
        coords = self.coordinate_head(x).sigmoid().permute(0, 2, 1, 3).contiguous()
        visibility = self.visibility_head(x).squeeze(-1).permute(0, 2, 1).contiguous()
        return {"pred_coords": coords, "pred_visibility": visibility}
