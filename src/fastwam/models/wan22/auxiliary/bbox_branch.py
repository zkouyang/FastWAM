from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base_video_dit_branch import AuxiliaryVideoDiTBranch, make_mlp


class BBoxDiTBranch(AuxiliaryVideoDiTBranch):
    """Independent DETR-like object-query expert."""

    def __init__(self, *, num_queries: int = 16, num_classes: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.object_queries = nn.Parameter(
            torch.randn(1, self.num_queries, self.hidden_dim) / self.hidden_dim**0.5
        )
        self.class_head = nn.Linear(self.hidden_dim, self.num_classes)
        self.box_head = make_mlp(self.hidden_dim, 4, layers=3)

    def pre_dit(self, *, batch_size: int, num_frames: int, context, context_mask: Optional[torch.Tensor]):
        tokens = self.object_queries.unsqueeze(1).expand(batch_size, num_frames, -1, -1)
        return self._prepare_tokens(
            tokens.flatten(1, 2),
            num_frames=num_frames,
            tokens_per_frame=self.num_queries,
            context=context,
            context_mask=context_mask,
        )

    def post_dit(self, tokens: torch.Tensor, pre_state):
        b = tokens.shape[0]
        t = pre_state["meta"]["num_frames"]
        x = tokens.view(b, t, self.num_queries, self.hidden_dim)
        return {
            "pred_logits": self.class_head(x),
            "pred_boxes": self.box_head(x).sigmoid(),
        }
