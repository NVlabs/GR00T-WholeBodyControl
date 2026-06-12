"""CLI wrapper for GEAR-SONIC LeRobot dataset cleansing."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.data.cleanse_lerobot_dataset import main


if __name__ == "__main__":
    main()
