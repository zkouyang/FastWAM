"""Losses for FastWAM spatial auxiliary experts."""

from .spatial_auxiliary import (
    bbox_hungarian_loss,
    depth_regression_loss,
    mask_hungarian_loss,
    prepare_trajectory_targets,
    trajectory_loss,
)

__all__ = [
    "depth_regression_loss",
    "bbox_hungarian_loss",
    "mask_hungarian_loss",
    "prepare_trajectory_targets",
    "trajectory_loss",
]
