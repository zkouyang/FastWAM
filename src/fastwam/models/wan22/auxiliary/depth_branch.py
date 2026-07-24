from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_video_dit_branch import AuxiliaryVideoDiTBranch


class DepthDiTBranch(AuxiliaryVideoDiTBranch):
    """Per-frame task tokens plus a lightweight spatial depth decoder."""

    def __init__(
        self,
        *,
        output_size: Sequence[int] = (128, 128),
        decoder_dim: int = 32,
        decoder_grid: Sequence[int] = (8, 8),
        horizon: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.horizon = None if horizon is None else int(horizon)
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.decoder_dim = int(decoder_dim)
        self.decoder_grid = (int(decoder_grid[0]), int(decoder_grid[1]))
        self.depth_queries = nn.Parameter(torch.randn(1, 1, self.hidden_dim) / self.hidden_dim**0.5)
        self.to_grid = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.decoder_dim * self.decoder_grid[0] * self.decoder_grid[1]),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.decoder_dim, self.decoder_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.decoder_dim, 1, 3, padding=1),
        )

    def pre_dit(
        self,
        *,
        batch_size: int,
        num_frames: int,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ):
        if self.horizon is not None and num_frames != self.horizon:
            raise ValueError(f"depth horizon {num_frames} != configured {self.horizon}")
        tokens = self.depth_queries.expand(batch_size, num_frames, -1)
        return self._prepare_tokens(
            tokens,
            num_frames=num_frames,
            tokens_per_frame=1,
            context=context,
            context_mask=context_mask,
        )

    def post_dit(self, tokens: torch.Tensor, pre_state):
        batch_size = tokens.shape[0]
        num_frames = pre_state["meta"]["num_frames"]
        x = self.to_grid(tokens).view(
            batch_size * num_frames, self.decoder_dim, *self.decoder_grid
        )
        x = self.decoder(x)
        x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        return x.view(batch_size, num_frames, 1, *self.output_size)
