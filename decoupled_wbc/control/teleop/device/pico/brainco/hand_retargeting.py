"""Brainco-only port of xr_teleoperate/teleop/robot_control/hand_retargeting.py.

The brainco.yml lives next to this file under assets/brainco_hand/. URDF paths
inside the YAML are relative to that asset directory, so we point
dex_retargeting at the same directory as the default URDF root.

The YAML follows xr_teleoperate's "union" convention: it carries BOTH the
DexPilot-only and vector-only field variants in the same dict, distinguished
by suffixes (e.g. `target_link_human_indices_dexpilot` vs
`target_link_human_indices_vector`). Recent dex_retargeting releases
(>= 0.4) reject those suffixed keys, so we filter the dict by the active
`type:` before handing it off.
"""

import logging
from pathlib import Path

import yaml
from dex_retargeting.retargeting_config import RetargetingConfig

logger = logging.getLogger(__name__)

ASSET_ROOT = Path(__file__).parent / "assets"
BRAINCO_YAML = ASSET_ROOT / "brainco_hand" / "brainco.yml"

# Fields whose names depend on the retargeting type. We rename the active one
# back to its canonical name and drop the inactive one.
_TYPE_SUFFIXED_KEYS = ("target_link_human_indices",)

# Fields only valid for one type — drop them when building the other.
_DEXPILOT_ONLY_KEYS = ("wrist_link_name", "finger_tip_link_names")
_VECTOR_ONLY_KEYS = ("target_origin_link_names", "target_task_link_names")


def _normalize_section(section: dict) -> dict:
    """Rewrite a left/right brainco.yml section into the dict layout
    `dex_retargeting.RetargetingConfig.from_dict` accepts.

    - Selects the right `<key>_<type>` variant of any suffixed field.
    - Strips fields that don't apply to the active retargeting type.
    """
    rt_type = (section.get("type") or "").lower()
    if rt_type not in {"dexpilot", "vector"}:
        raise ValueError(
            f"brainco.yml: unsupported retargeting type {section.get('type')!r}; "
            "expected 'DexPilot' or 'vector'."
        )

    out = dict(section)
    for base in _TYPE_SUFFIXED_KEYS:
        active = f"{base}_{rt_type}"
        inactive = f"{base}_{'vector' if rt_type == 'dexpilot' else 'dexpilot'}"
        if active in out:
            out[base] = out.pop(active)
        out.pop(inactive, None)

    drop = _VECTOR_ONLY_KEYS if rt_type == "dexpilot" else _DEXPILOT_ONLY_KEYS
    for k in drop:
        out.pop(k, None)
    return out


class BraincoHandRetargeting:
    def __init__(self, config_path: Path = BRAINCO_YAML, asset_root: Path = ASSET_ROOT):
        RetargetingConfig.set_default_urdf_dir(str(asset_root))

        with config_path.open("r") as f:
            cfg = yaml.safe_load(f)
        if "left" not in cfg or "right" not in cfg:
            raise ValueError("brainco.yml must contain 'left' and 'right' keys.")

        left_cfg = _normalize_section(cfg["left"])
        right_cfg = _normalize_section(cfg["right"])

        self.left_retargeting = RetargetingConfig.from_dict(left_cfg).build()
        self.right_retargeting = RetargetingConfig.from_dict(right_cfg).build()

        self.left_joint_names = self.left_retargeting.joint_names
        self.right_joint_names = self.right_retargeting.joint_names

        self.left_indices = self.left_retargeting.optimizer.target_link_human_indices
        self.right_indices = self.right_retargeting.optimizer.target_link_human_indices

        # Driver Motor ID order from
        # https://www.brainco-hz.com/docs/revolimb-hand/product/parameters.html
        self.left_brainco_api_joint_names = [
            "left_thumb_metacarpal_joint",
            "left_thumb_proximal_joint",
            "left_index_proximal_joint",
            "left_middle_proximal_joint",
            "left_ring_proximal_joint",
            "left_pinky_proximal_joint",
        ]
        self.right_brainco_api_joint_names = [
            "right_thumb_metacarpal_joint",
            "right_thumb_proximal_joint",
            "right_index_proximal_joint",
            "right_middle_proximal_joint",
            "right_ring_proximal_joint",
            "right_pinky_proximal_joint",
        ]
        self.left_dex_to_hardware = [
            self.left_joint_names.index(n) for n in self.left_brainco_api_joint_names
        ]
        self.right_dex_to_hardware = [
            self.right_joint_names.index(n) for n in self.right_brainco_api_joint_names
        ]
