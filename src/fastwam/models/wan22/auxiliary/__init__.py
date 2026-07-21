"""Independent training-only spatial experts for FastWAM."""

from .bbox_branch import BBoxDiTBranch
from .depth_branch import DepthDiTBranch
from .mask_branch import MaskDiTBranch
from .trajectory_branch import TrajectoryDiTBranch

__all__ = [
    "DepthDiTBranch",
    "BBoxDiTBranch",
    "MaskDiTBranch",
    "TrajectoryDiTBranch",
]
