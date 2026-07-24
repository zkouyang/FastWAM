from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_video_dit_branch import AuxiliaryVideoDiTBranch, make_mlp


class MaskDiTBranch(AuxiliaryVideoDiTBranch):
    """Independent query-to-mask expert; it does not consume BBox outputs."""

    def __init__(
        self,
        *,
        num_queries: int = 16,
        output_size: Sequence[int] = (128, 128),
        mask_dim: int = 32,
        feature_grid: Sequence[int] = (8, 8),
        horizon: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_queries = int(num_queries)
        self.horizon = None if horizon is None else int(horizon)
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.mask_dim = int(mask_dim)
        self.feature_grid = (int(feature_grid[0]), int(feature_grid[1]))
        self.image_query = nn.Parameter(torch.randn(1, 1, self.hidden_dim) / self.hidden_dim**0.5)
        self.mask_queries = nn.Parameter(
            torch.randn(1, self.num_queries, self.hidden_dim) / self.hidden_dim**0.5
        )
        self.image_head = nn.Linear(
            self.hidden_dim, self.mask_dim * self.feature_grid[0] * self.feature_grid[1]
        )
        self.query_head = make_mlp(self.hidden_dim, self.mask_dim, layers=3)

    def pre_dit(self, *, batch_size: int, num_frames: int, context, context_mask: Optional[torch.Tensor]):
        if self.horizon is not None and num_frames != self.horizon:
            raise ValueError(f"mask horizon {num_frames} != configured {self.horizon}")
        one_frame = torch.cat((self.image_query, self.mask_queries), dim=1)
        tokens = one_frame.unsqueeze(1).expand(batch_size, num_frames, -1, -1)
        return self._prepare_tokens(
            tokens.flatten(1, 2),
            num_frames=num_frames,
            tokens_per_frame=self.num_queries + 1,
            context=context,
            context_mask=context_mask,
        )

    def post_dit(self, tokens: torch.Tensor, pre_state):
        b = tokens.shape[0]
        t = pre_state["meta"]["num_frames"]
        x = tokens.view(b, t, self.num_queries + 1, self.hidden_dim)
        feature = self.image_head(x[:, :, 0]).view(b * t, self.mask_dim, *self.feature_grid)
        queries = self.query_head(x[:, :, 1:]).view(b * t, self.num_queries, self.mask_dim)
        logits = torch.einsum("bqd,bdhw->bqhw", queries, feature) / self.mask_dim**0.5
        logits = F.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)
        return logits.view(b, t, self.num_queries, *self.output_size)
