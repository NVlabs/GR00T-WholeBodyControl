"""
Full list of loco-manipulation tasks.

GroundOnly - ground only environments

locomanip_pnp - factory environments, pick and place tasks:
LMBottlePnP
LMBoxPnP
"""

from .base import REGISTERED_LOCOMANIPULATION_ENVS

ALL_LOCOMANIPULATION_ENVIRONMENTS = REGISTERED_LOCOMANIPULATION_ENVS.keys()
