"""Video-DiT-compatible building blocks shared by auxiliary experts.

This is a construction helper, not a unified SSI encoder.  Every concrete
branch owns its own instance of the text/time projections and DiT blocks.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ..wan_video_dit import DiTBlock, precompute_freqs_cis, sinusoidal_embedding_1d


class AuxiliaryVideoDiTBranch(nn.Module):
    """Independent DiT expert exposing the same pre/post split used by MoT."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        max_frames: int = 64,
        max_tokens: int = 4096,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.max_frames = int(max_frames)
        self.max_tokens = int(max_tokens)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        if self.attn_head_dim % 2:
            raise ValueError("attn_head_dim must be even for RoPE")
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 6)
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    attn_head_dim=self.attn_head_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    eps=float(eps),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.frame_embedding = nn.Parameter(
            torch.randn(1, self.max_frames, 1, self.hidden_dim) / self.hidden_dim**0.5
        )
        # Keep the complex RoPE cache as a plain tensor, matching VideoDiT and
        # ActionDiT. DDP otherwise tries to broadcast a complex buffer on Gloo.
        self.freqs = precompute_freqs_cis(self.attn_head_dim, end=self.max_tokens)

    def _prepare_tokens(
        self,
        tokens: torch.Tensor,
        *,
        num_frames: int,
        tokens_per_frame: int,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        timestep: Optional[torch.Tensor] = None,
        add_frame_embedding: bool = True,
        rope_position_ids: Optional[torch.Tensor] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,S,D], got {tuple(tokens.shape)}")
        batch_size, seq_len, hidden_dim = tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"token hidden dim {hidden_dim} != {self.hidden_dim}")
        if num_frames <= 0 or num_frames > self.max_frames:
            raise ValueError(f"num_frames must be in [1,{self.max_frames}], got {num_frames}")
        if seq_len != num_frames * tokens_per_frame:
            raise ValueError(
                f"token layout mismatch: {seq_len} != {num_frames}*{tokens_per_frame}"
            )
        if seq_len > self.max_tokens:
            raise ValueError(f"sequence length {seq_len} exceeds RoPE cache {self.max_tokens}")
        if context.ndim != 3 or context.shape[0] != batch_size:
            raise ValueError("context must be [B,L,text_dim] and match token batch")
        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        if context_mask.shape != context.shape[:2]:
            raise ValueError("context_mask must have shape [B,L]")

        tokens = tokens.view(batch_size, num_frames, tokens_per_frame, hidden_dim)
        if add_frame_embedding:
            frame_pos = self.frame_embedding[:, :num_frames].expand(
                batch_size, -1, tokens_per_frame, -1
            )
            tokens = tokens + frame_pos.to(dtype=tokens.dtype, device=tokens.device)
        tokens = tokens.flatten(1, 2)
        if rope_position_ids is None:
            freqs = self.freqs[:seq_len]
        else:
            if rope_position_ids.ndim != 1 or rope_position_ids.shape[0] != seq_len:
                raise ValueError(
                    "rope_position_ids must be [S] and match token sequence length, "
                    f"got {tuple(rope_position_ids.shape)} vs {seq_len}"
                )
            rope_position_ids = rope_position_ids.to(device=self.freqs.device, dtype=torch.long)
            if int(rope_position_ids.min()) < 0 or int(rope_position_ids.max()) >= self.max_tokens:
                raise ValueError("rope_position_ids are outside the configured RoPE cache")
            freqs = self.freqs[rope_position_ids]

        if timestep is None:
            timestep = torch.zeros((batch_size,), dtype=tokens.dtype, device=tokens.device)
        if timestep.shape != (batch_size,):
            raise ValueError(f"timestep must be [B], got {tuple(timestep.shape)}")
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.to(torch.bool).unsqueeze(1).expand(-1, seq_len, -1)
        payload_meta = {
            "batch_size": batch_size,
            "num_frames": int(num_frames),
            "tokens_per_frame": int(tokens_per_frame),
            "seq_len": seq_len,
        }
        if meta:
            payload_meta.update(meta)
        return {
            "tokens": tokens,
            "freqs": freqs.view(seq_len, 1, -1).to(tokens.device),
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": payload_meta,
        }

    def initialize_backbone_from_video(self, video_expert: nn.Module) -> None:
        """Best-effort initialization for shape-compatible Video-DiT modules."""

        for name in ("text_embedding", "time_embedding", "time_projection", "blocks"):
            source = getattr(video_expert, name, None)
            target = getattr(self, name)
            if source is None:
                continue
            source_state = source.state_dict()
            target_state = target.state_dict()
            compatible = {
                key: value
                for key, value in source_state.items()
                if key in target_state and value.shape == target_state[key].shape
            }
            target.load_state_dict(compatible, strict=False)


def make_mlp(hidden_dim: int, out_dim: int, *, layers: int = 3) -> nn.Sequential:
    modules: list[nn.Module] = []
    for _ in range(max(1, layers - 1)):
        modules.extend((nn.Linear(hidden_dim, hidden_dim), nn.GELU()))
    modules.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*modules)
