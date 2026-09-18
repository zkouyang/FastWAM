from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_video_dit_branch import AuxiliaryVideoDiTBranch


class DepthDiTBranch(AuxiliaryVideoDiTBranch):
    """Per-frame task tokens plus a Video-spatial depth decoder."""

    def __init__(
        self,
        *,
        output_size: Sequence[int] = (224, 224),
        decoder_dim: int = 64,
        decoder_grid: Sequence[int] = (14, 14),
        video_hidden_dim: Optional[int] = None,
        horizon: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.horizon = None if horizon is None else int(horizon)
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.decoder_dim = int(decoder_dim)
        self.decoder_grid = (int(decoder_grid[0]), int(decoder_grid[1]))
        self.video_hidden_dim = int(video_hidden_dim or self.hidden_dim)
        if min(*self.output_size, *self.decoder_grid, self.decoder_dim) <= 0:
            raise ValueError("Depth decoder dimensions must all be positive")

        self.depth_queries = nn.Parameter(torch.randn(1, 1, self.hidden_dim) / self.hidden_dim**0.5)
        self.to_grid = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.decoder_dim * self.decoder_grid[0] * self.decoder_grid[1]),
        )

        # A Video DiT token represents a 2x2 latent patch. Learned unpacking
        # preserves that spatial correspondence instead of compressing all
        # first-frame Video K/V into one global depth vector.
        self.video_to_grid = nn.Sequential(
            nn.LayerNorm(self.video_hidden_dim),
            nn.Linear(self.video_hidden_dim, self.decoder_dim * 4),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(self.decoder_dim * 2, self.decoder_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.decoder_dim, self.decoder_dim, 3, padding=1),
            nn.GELU(),
        )

        self.upsample_blocks = nn.ModuleList()
        current_dim = self.decoder_dim
        current_h, current_w = self.decoder_grid
        minimum_dim = min(8, self.decoder_dim)
        while current_h * 2 <= self.output_size[0] and current_w * 2 <= self.output_size[1]:
            next_dim = max(current_dim // 2, minimum_dim)
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_dim, next_dim, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(next_dim, next_dim, 3, padding=1),
                    nn.GELU(),
                )
            )
            current_dim = next_dim
            current_h *= 2
            current_w *= 2
        self.output_head = nn.Conv2d(current_dim, 1, 3, padding=1)

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

    def post_dit(
        self,
        tokens: torch.Tensor,
        pre_state,
        *,
        video_spatial_tokens: torch.Tensor,
    ):
        batch_size = tokens.shape[0]
        num_frames = pre_state["meta"]["num_frames"]
        if tokens.shape[1] != num_frames:
            raise ValueError(
                f"depth token length {tokens.shape[1]} != configured frame count {num_frames}"
            )
        if video_spatial_tokens.ndim != 4 or video_spatial_tokens.shape[0] != batch_size:
            raise ValueError(
                "video_spatial_tokens must be [B,H,W,D] and match the depth token batch, "
                f"got {tuple(video_spatial_tokens.shape)}"
            )
        if video_spatial_tokens.shape[-1] != self.video_hidden_dim:
            raise ValueError(
                f"Video token dim {video_spatial_tokens.shape[-1]} != {self.video_hidden_dim}"
            )

        depth_grid = self.to_grid(tokens).view(
            batch_size * num_frames, self.decoder_dim, *self.decoder_grid
        )

        grid_h, grid_w = video_spatial_tokens.shape[1:3]
        video_grid = self.video_to_grid(video_spatial_tokens)
        video_grid = video_grid.view(
            batch_size, grid_h, grid_w, 2, 2, self.decoder_dim
        ).permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size, self.decoder_dim, grid_h * 2, grid_w * 2
        )
        if video_grid.shape[-2:] != self.decoder_grid:
            video_grid = F.interpolate(
                video_grid, size=self.decoder_grid, mode="bilinear", align_corners=False
            )
        video_grid = video_grid[:, None].expand(-1, num_frames, -1, -1, -1).reshape(
            batch_size * num_frames, self.decoder_dim, *self.decoder_grid
        )

        x = self.fusion(torch.cat((depth_grid, video_grid), dim=1))
        for block in self.upsample_blocks:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = block(x)
        x = self.output_head(x)
        if x.shape[-2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        return x.view(batch_size, num_frames, 1, *self.output_size)
