"""Resolve model paths without depending on the working directory."""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from gear_sonic.utils.hand_control.inspire.config import DEFAULT_MAPPING_PATH

# config.yaml paths start with gear_sonic/, relative to this package's parent.
PACKAGE_PARENT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class InspireAssetPaths:
    model: Path
    scene: Path
    urdf: Path


def load_asset_paths(config_path: str | Path = DEFAULT_MAPPING_PATH) -> InspireAssetPaths:
    """Resolve configured assets and check URDF meshes before loading Pinocchio.

    Configured relative paths are relative to the package parent, even when a
    custom configuration is used. Absolute paths are accepted for local assets.
    """
    with Path(config_path).expanduser().open(encoding="utf-8") as stream:
        simulation = yaml.safe_load(stream)["simulation"]

    def resolve(key: str) -> Path:
        path = (PACKAGE_PARENT / Path(simulation[key]).expanduser()).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing Inspire asset: {path}")
        return path

    assets = InspireAssetPaths(resolve("model_path"), resolve("scene_path"), resolve("urdf_path"))
    for mesh in ET.parse(assets.urdf).findall(".//mesh"):
        path = (assets.urdf.parent / mesh.attrib["filename"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing Inspire mesh: {path}")
        with path.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(
                    f"Mesh is a Git LFS pointer: {path}. Run git lfs pull "
                    '--include="gear_sonic/data/robots/g1/meshes/**" --exclude=""'
                )
    return assets
