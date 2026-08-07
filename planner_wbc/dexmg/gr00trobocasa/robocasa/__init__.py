from robocasa.environments.locomanipulation.base import (
    PnPBottle,
    PickBottleShelf,
    PnPBottleHigh,
    NavPickBottle,
    PnPBottleRandRobotPose,
    VisualReach,
    PnPBottleFixtureToFixture,
    PnPBottleFixtureToFixtureSourceDemo,
    PnPBottleShelfToTable,
    PnPBottleTableToTable,
    PickBottleGround,
    PickBottles,
    NavPickBottles,
    PnPBottlesTableToTable,
)
from robocasa.environments.locomanipulation.locomanip_basic import (
    LMPickBottle,
    LMPickBottleHigh,
    LMNavPickBottle,
    LMPickBottleGround,
    LMPnPBottle,
    LMPickMultipleBottles,
    LMPnPMultipleBottles,
    LMPickBottleShelf,
    LMNavPickBottleShelf,
    LMPickBottleShelfLow,
    LMNavPickBottleShelfLow,
    LMPnPBottleToPlate,
    LMPnPAppleToPlate,
)
from robocasa.environments.locomanipulation.locomanip_pnp import (
    LMBottlePnP,
    LMBoxPnP,
)

from robocasa.environments.locomanipulation.locomanip_dc import (
    LMNavPickBottleDC,
    LMPnPAppleToPlateDC,
)

# from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
from robosuite.controllers import ALL_PART_CONTROLLERS, load_composite_controller_config
from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.models.grippers import ALL_GRIPPERS
from robosuite.robots import ALL_ROBOTS


import mujoco

assert (
    mujoco.__version__ == "3.2.6" or mujoco.__version__ == "3.3.2"
), "MuJoCo version must be 3.2.6 or 3.3.2. Please install the correct version."

import numpy

assert numpy.__version__ in [
    "1.23.2",
    "1.23.3",
    "1.23.5",
    "1.26.4",
    "2.2.5",
    "2.2.6",
], "numpy version must be either 1.23.{2,3,5}, 1.26.4 or 2.2.{5,6}. Please install one of these versions."

import robosuite

assert robosuite.__version__ in [
    "1.5.0",
    "1.5.1",
], "robosuite version must be 1.5.{0,1}. Please install the correct version"

__version__ = "0.2.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
