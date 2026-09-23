"""
Much of this code is directly copied from obj2mjcf:
https://github.com/kevinzakka/obj2mjcf/blob/main/obj2mjcf/_cli.py

Credit: Kevin Zakka
"""

from dataclasses import dataclass
from typing import Optional, Sequence
from PIL import Image
from pathlib import Path

import os
import shutil

# MTL fields relevant to MuJoCo.
_MTL_FIELDS = (
    # Ambient, diffuse and specular colors.
    "Ka",
    "Kd",
    "Ks",
    # d or Tr are used for the rgba transparency.
    "d",
    "Tr",
    # Shininess.
    "Ns",
    # References a texture file.
    "map_Kd",
)

# Character used to denote a comment in an MTL file.
_MTL_COMMENT_CHAR = "#"


@dataclass
class Material:
    name: str
    Ka: Optional[str] = None
    Kd: Optional[str] = None
    Ks: Optional[str] = None
    d: Optional[str] = None
    Tr: Optional[str] = None
    Ns: Optional[str] = None
    map_Kd: Optional[str] = None

    @staticmethod
    def from_string(lines: Sequence[str]) -> "Material":
        """Construct a Material object from a string."""
        attrs = {"name": lines[0].split(" ")[1].strip()}
        for line in lines[1:]:
            for attr in _MTL_FIELDS:
                if line.startswith(attr):
                    elems = line.split(" ")[1:]
                    elems = [elem for elem in elems if elem != ""]
                    attrs[attr] = " ".join(elems)
                    break
        return Material(**attrs)

    def mjcf_rgba(self) -> str:
        Kd = self.Kd or "1.0 1.0 1.0"
        if self.d is not None:  # alpha
            alpha = self.d
        elif self.Tr is not None:  # 1 - alpha
            alpha = str(1.0 - float(self.Tr))
        else:
            alpha = "1.0"
        # alpha = "1.0"
        # TODO(kevin): Figure out how to use Ka for computing rgba.
        return f"{Kd} {alpha}"

    def mjcf_shininess(self) -> str:
        if self.Ns is not None:
            # Normalize Ns value to [0, 1]. Ns values normally range from 0 to 1000.
            Ns = float(self.Ns) / 1_000
        else:
            Ns = 0.5
        return f"{Ns}"

    def mjcf_specular(self) -> str:
        if self.Ks is not None:
            # Take the average of the specular RGB values.
            Ks = sum(list(map(float, self.Ks.split(" ")))) / 3
        else:
            Ks = 0.5
        return f"{Ks}"


def get_mtls(filename, work_dir):
    filename = Path(filename)
    work_dir = Path(work_dir)

    process_mtl = False
    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("mtllib"):  # Deals with commented out lines.
                process_mtl = True
                name = line.split()[1]
                break

    sub_mtls: List[List[str]] = []
    mtls: List[Material] = []
    if process_mtl:
        # Make sure the MTL file exists. The MTL filepath is relative to the OBJ's.
        mtl_filename = filename.parent / name
        if not mtl_filename.exists():
            raise RuntimeError(
                f"The MTL file {mtl_filename.resolve()} referenced in the OBJ file "
                f"{filename} does not exist"
            )
        # logging.info(f"Found MTL file: {mtl_filename}")

        # Parse the MTL file into separate materials.
        with open(mtl_filename, "r") as f:
            lines = f.readlines()
        # Remove comments.
        lines = [line for line in lines if not line.startswith(_MTL_COMMENT_CHAR)]
        # Remove empty lines.
        lines = [line for line in lines if line.strip()]
        # Remove trailing whitespace.
        lines = [line.strip() for line in lines]
        # Split at each new material definition.
        for line in lines:
            if line.startswith("newmtl"):
                sub_mtls.append([])
            sub_mtls[-1].append(line)
        for sub_mtl in sub_mtls:
            mtls.append(Material.from_string(sub_mtl))

        # Process each material.
        for mtl in mtls:
            # logging.info(f"Found material: {mtl.name}")
            if mtl.map_Kd is not None:
                texture_path = Path(mtl.map_Kd)
                texture_name = texture_path.name
                src_filename = filename.parent / texture_path
                if not src_filename.exists():
                    raise RuntimeError(
                        f"The texture file {src_filename} referenced in the MTL file "
                        f"{mtl.name} does not exist"
                    )
                # We want a flat directory structure in work_dir.
                dst_filename = work_dir / texture_name
                shutil.copy(src_filename, dst_filename)
                # MuJoCo only supports PNG textures ¯\_(ツ)_/¯.
                if texture_path.suffix.lower() in [".jpg", ".jpeg"]:
                    image = Image.open(dst_filename)
                    os.remove(dst_filename)
                    dst_filename = (work_dir / texture_path.stem).with_suffix(".png")
                    image.save(dst_filename)
                    texture_name = dst_filename.name
                    mtl.map_Kd = texture_name
                resize_texture(dst_filename, 1.0)
        # logging.info("Done processing MTL file")

    return mtls


def resize_texture(filename: Path, resize_percent) -> None:
    """Resize a texture to a percentage of its original size."""
    if resize_percent == 1.0:
        return
    image = Image.open(filename)
    new_width = int(image.size[0] * resize_percent)
    new_height = int(image.size[1] * resize_percent)
    logging.info(f"Resizing {filename} to {new_width}x{new_height}")
    image = image.resize((new_width, new_height), Image.LANCZOS)
    image.save(filename)


def get_image_paths(mtl_path):
    image_paths = []

    # Parse the MTL file into separate materials.
    with open(mtl_path, "r") as f:
        lines = f.readlines()
    # Remove comments.
    lines = [line for line in lines if not line.startswith(_MTL_COMMENT_CHAR)]
    # Remove empty lines.
    lines = [line for line in lines if line.strip()]
    # Remove trailing whitespace.
    lines = [line.strip() for line in lines]
    # Split at each new material definition.

    sub_mtls = []

    for line in lines:
        if line.startswith("newmtl"):
            sub_mtls.append([])
        sub_mtls[-1].append(line)
    for sub_mtl in sub_mtls:
        # print(sub_mtl)

        path = Material.from_string(sub_mtl).map_Kd

        if path is not None:
            image_paths.append(path)

    return image_paths
