"""G1 robot model instantiation helpers."""

from .g1 import instantiate_g1_robot_model
from .g1_inspire import instantiate_g1_rh56dfx_robot_model

__all__ = ["instantiate_g1_robot_model", "instantiate_g1_rh56dfx_robot_model"]
