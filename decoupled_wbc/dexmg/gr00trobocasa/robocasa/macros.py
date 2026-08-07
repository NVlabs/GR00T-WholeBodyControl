"""
Macro settings that can be imported and toggled. Internally, specific parts of the codebase rely on these settings
for determining core functionality.

To make sure global reference is maintained, should import these settings as:

`import robocasa.macros as macros`
"""

SHOW_SITES = False

# whether to print debugging information
VERBOSE = False

# Spacemouse settings. Used by SpaceMouse class in robosuite/devices/spacemouse.py
SPACEMOUSE_VENDOR_ID = 9583
SPACEMOUSE_PRODUCT_ID = 50741

DATASET_BASE_PATH = None

try:
    from robocasa.macros_private import *
except ImportError:
    from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

    import robocasa

    ROBOSUITE_DEFAULT_LOGGER.warning("No private macro file found!")
    ROBOSUITE_DEFAULT_LOGGER.warning("It is recommended to use a private macro file")
    ROBOSUITE_DEFAULT_LOGGER.warning(
        "To setup, run: python {}/scripts/setup_macros.py".format(robocasa.__path__[0])
    )
