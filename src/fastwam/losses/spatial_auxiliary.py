"""Self-contained losses for the four independent spatial branches."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F


def depth_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    confidence: Optional[torch.Tensor] = None,
    alpha_grad: float = 0.0,
) -> torch.Tensor:
    if target.ndim != 5:
        raise ValueError(f"depth target must be [B,T,1,H,W], got {tuple(target.shape)}")
    target = target.to(device=pred.device, dtype=torch.float32)
    if target.shape[-2:] != pred.shape[-2:]:
        b, t = target.shape[:2]
        target = F.interpolate(
            target.flatten(0, 1), size=pred.shape[-2:], mode="bilinear", align_corners=False
        ).view(b, t, 1, *pred.shape[-2:])
    raw = F.smooth_l1_loss(pred.float(), target, reduction="none")
    if confidence is not None:
        confidence = confidence.to(device=pred.device, dtype=torch.float32)
        if confidence.shape[-2:] != pred.shape[-2:]:
            b, t = confidence.shape[:2]
            confidence = F.interpolate(
                confidence.flatten(0, 1), size=pred.shape[-2:], mode="bilinear", align_corners=False
            ).view(b, t, 1, *pred.shape[-2:])
        loss = (raw * confidence).sum() / confidence.sum().clamp(min=1.0)
    else:
        loss = raw.mean()
    if alpha_grad:
        dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
        dx_target = target[..., :, 1:] - target[..., :, :-1]
        dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
        dy_target = target[..., 1:, :] - target[..., :-1, :]
        loss = loss + float(alpha_grad) * (
            F.smooth_l1_loss(dx_pred.float(), dx_target) +
            F.smooth_l1_loss(dy_pred.float(), dy_target)
        )
    return loss


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (
        boxes[..., 3] - boxes[..., 1]
    ).clamp(min=0)


def _pairwise_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / union.clamp(min=1e-7)
    enc_lt = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enc_rb = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[..., 0] * enc_wh[..., 1]
    return iou - (enc_area - union) / enc_area.clamp(min=1e-7)


def _linear_sum_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cost.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=cost.device)
        return empty, empty
    try:
        from scipy.optimize import linear_sum_assignment

        row, col = linear_sum_assignment(cost.detach().float().cpu().numpy())
        return (
            torch.as_tensor(row, dtype=torch.long, device=cost.device),
            torch.as_tensor(col, dtype=torch.long, device=cost.device),
        )
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise RuntimeError("scipy is required for Hungarian matching") from exc


def bbox_hungarian_loss(
    outputs: dict[str, torch.Tensor],
    boxes: list[list[torch.Tensor]],
    labels: Optional[list[list[torch.Tensor]]] = None,
    *,
    beta_l1: float = 5.0,
    beta_giou: float = 2.0,
) -> torch.Tensor:
    logits = outputs["pred_logits"].float()
    pred_boxes = outputs["pred_boxes"].float()
    bsz, horizon, queries, classes = logits.shape
    if len(boxes) != bsz:
        raise ValueError("bbox target batch does not match predictions")
    cls_losses: list[torch.Tensor] = []
    box_losses: list[torch.Tensor] = []
    giou_losses: list[torch.Tensor] = []
    for b in range(bsz):
        if len(boxes[b]) != horizon:
            raise ValueError("bbox target horizon does not match predictions")
        for t in range(horizon):
            target_boxes = boxes[b][t].to(device=logits.device, dtype=torch.float32)
            target_labels = (
                labels[b][t].to(device=logits.device, dtype=torch.long)
                if labels is not None else torch.zeros(len(target_boxes), dtype=torch.long, device=logits.device)
            )
            cls_target = torch.zeros_like(logits[b, t])
            if len(target_boxes):
                if classes == 1:
                    cls_cost = -logits[b, t].sigmoid()[:, 0, None].expand(-1, len(target_boxes))
                else:
                    cls_cost = -logits[b, t].softmax(-1)[:, target_labels]
                l1_cost = torch.cdist(pred_boxes[b, t], target_boxes, p=1)
                giou = _pairwise_giou(
                    _cxcywh_to_xyxy(pred_boxes[b, t]), _cxcywh_to_xyxy(target_boxes)
                )
                pred_idx, target_idx = _linear_sum_assignment(
                    cls_cost + float(beta_l1) * l1_cost - float(beta_giou) * giou
                )
                if classes == 1:
                    cls_target[pred_idx, 0] = 1.0
                else:
                    cls_target[pred_idx, target_labels[target_idx]] = 1.0
                box_losses.append(F.l1_loss(pred_boxes[b, t, pred_idx], target_boxes[target_idx]))
                matched_giou = _pairwise_giou(
                    _cxcywh_to_xyxy(pred_boxes[b, t, pred_idx]),
                    _cxcywh_to_xyxy(target_boxes[target_idx]),
                ).diagonal()
                giou_losses.append((1.0 - matched_giou).mean())
            cls_losses.append(F.binary_cross_entropy_with_logits(logits[b, t], cls_target))
    zero = logits.sum() * 0.0
    return (
        torch.stack(cls_losses).mean()
        + float(beta_l1) * (torch.stack(box_losses).mean() if box_losses else zero)
        + float(beta_giou) * (torch.stack(giou_losses).mean() if giou_losses else zero)
    )


def _resize_masks(masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if masks.shape[-2:] == size:
        return masks.float()
    return F.interpolate(masks[:, None].float(), size=size, mode="nearest")[:, 0]


def mask_hungarian_loss(
    pred_masks: torch.Tensor,
    targets: list[list[torch.Tensor]],
    *,
    beta_dice: float = 1.0,
) -> torch.Tensor:
    pred_masks = pred_masks.float()
    bsz, horizon, _queries, height, width = pred_masks.shape
    losses: list[torch.Tensor] = []
    for b in range(bsz):
        if len(targets[b]) != horizon:
            raise ValueError("mask target horizon does not match predictions")
        for t in range(horizon):
            target = _resize_masks(targets[b][t].to(pred_masks.device), (height, width))
            if len(target) == 0:
                losses.append(F.binary_cross_entropy_with_logits(pred_masks[b, t], torch.zeros_like(pred_masks[b, t])))
                continue
            pred_prob = pred_masks[b, t].sigmoid().flatten(1)
            target_flat = target.flatten(1)
            intersection = torch.einsum("qd,nd->qn", pred_prob, target_flat)
            dice_cost = 1.0 - (2.0 * intersection + 1.0) / (
                pred_prob.sum(1)[:, None] + target_flat.sum(1)[None, :] + 1.0
            )
            # Pairwise stable BCE cost, independent of BBox predictions/matching.
            p = pred_prob.clamp(1e-6, 1 - 1e-6)
            bce_cost = -(
                torch.einsum("qd,nd->qn", p.log(), target_flat)
                + torch.einsum("qd,nd->qn", (1 - p).log(), 1 - target_flat)
            ) / (height * width)
            pred_idx, target_idx = _linear_sum_assignment(bce_cost + float(beta_dice) * dice_cost)
            matched_logits = pred_masks[b, t, pred_idx]
            matched_targets = target[target_idx]
            # Matched queries learn their instance masks; unmatched queries are
            # explicit background instead of receiving no gradient.
            all_targets = torch.zeros_like(pred_masks[b, t])
            all_targets[pred_idx] = matched_targets
            bce = F.binary_cross_entropy_with_logits(pred_masks[b, t], all_targets)
            matched_prob = matched_logits.sigmoid().flatten(1)
            matched_flat = matched_targets.flatten(1)
            dice = 1.0 - (2 * (matched_prob * matched_flat).sum(1) + 1) / (
                matched_prob.sum(1) + matched_flat.sum(1) + 1
            )
            losses.append(bce + float(beta_dice) * dice.mean())
    return torch.stack(losses).mean() if losses else pred_masks.sum() * 0.0


def _double_grid(count: int, *, device: torch.device) -> torch.Tensor:
    """ATM-style regular grid plus a half-cell shifted grid."""

    if count <= 0:
        return torch.empty((0, 2), device=device)
    side = max(1, int(math.ceil(math.sqrt(count / 2))))
    axis = (torch.arange(side, device=device, dtype=torch.float32) + 0.5) / side
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    first = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
    shifted = (first + 0.5 / side).remainder(1.0)
    return torch.cat((first, shifted), dim=0)[:count]


def prepare_trajectory_targets(
    sample: dict[str, Any],
    *,
    num_points: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    trajectories = sample["trajectories"].to(device=device, dtype=torch.float32)
    visibility = sample["traj_visibility"].to(device=device, dtype=torch.float32)
    queries = sample["traj_query_points"].to(device=device, dtype=torch.float32)
    selection_queries = sample.get("traj_query_points_local", sample["traj_query_points"]).to(
        device=device, dtype=torch.float32
    )
    pad = sample.get("traj_point_is_pad")
    cameras = sample.get("traj_camera_indices")
    if pad is None:
        pad = torch.zeros(queries.shape[:2], dtype=torch.bool, device=device)
    else:
        pad = pad.to(device=device, dtype=torch.bool)
    if cameras is None:
        cameras = torch.zeros(queries.shape[:2], dtype=torch.long, device=device)
    else:
        cameras = cameras.to(device=device, dtype=torch.long)
    bsz, _available, horizon, _ = trajectories.shape
    out_coords = torch.zeros((bsz, num_points, horizon, 2), device=device)
    out_vis = torch.zeros((bsz, num_points, horizon), device=device)
    out_query = torch.zeros((bsz, num_points, 2), device=device)
    out_valid = torch.zeros((bsz, num_points), dtype=torch.bool, device=device)
    for b in range(bsz):
        valid = (~pad[b]) & (visibility[b, :, 0] > 0)
        camera_values = torch.unique(cameras[b, valid])
        selected: list[torch.Tensor] = []
        remaining = num_points
        for camera_pos, camera in enumerate(camera_values):
            candidate = torch.where(valid & (cameras[b] == camera))[0]
            allocation = remaining // max(1, len(camera_values) - camera_pos)
            allocation = min(allocation, len(candidate))
            if allocation == 0:
                continue
            candidate_points = selection_queries[b, candidate]
            grid = _double_grid(allocation, device=device)
            distances = torch.cdist(grid, candidate_points)
            chosen: list[int] = []
            for row in range(len(grid)):
                order = distances[row].argsort()
                local = next((int(i) for i in order.tolist() if int(i) not in chosen), None)
                if local is not None:
                    chosen.append(local)
            indices = candidate[torch.tensor(chosen, device=device)]
            selected.append(indices)
            remaining -= len(indices)
        if remaining > 0:
            all_valid = torch.where(valid)[0]
            already = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long, device=device)
            keep = all_valid[~torch.isin(all_valid, already)][:remaining]
            if len(keep):
                selected.append(keep)
        if not selected:
            continue
        indices = torch.cat(selected)[:num_points]
        count = len(indices)
        out_coords[b, :count] = trajectories[b, indices]
        out_vis[b, :count] = visibility[b, indices]
        out_query[b, :count] = queries[b, indices]
        out_valid[b, :count] = True
    return {
        "coords": out_coords,
        "visibility": out_vis,
        "query_points": out_query,
        "point_valid": out_valid,
    }


def trajectory_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    beta_vis: float = 1.0,
) -> torch.Tensor:
    pred_coords = outputs["pred_coords"].float()
    pred_vis = outputs["pred_visibility"].float()
    coords = targets["coords"].to(pred_coords)
    visibility = targets["visibility"].to(pred_coords)
    point_valid = targets["point_valid"].to(pred_coords.device)
    coord_valid = visibility * point_valid.unsqueeze(-1).to(visibility.dtype)
    coord_raw = F.smooth_l1_loss(pred_coords, coords, reduction="none").mean(-1)
    coord = (coord_raw * coord_valid).sum() / coord_valid.sum().clamp(min=1.0)
    vis_raw = F.binary_cross_entropy_with_logits(pred_vis, visibility, reduction="none")
    valid = point_valid.unsqueeze(-1).expand_as(vis_raw).to(vis_raw.dtype)
    vis = (vis_raw * valid).sum() / valid.sum().clamp(min=1.0)
    return coord + float(beta_vis) * vis
