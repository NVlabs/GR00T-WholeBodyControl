"""Train clip-contained teleop-v15 from official multi-motion SONIC init."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import (
    causal_multimotion_v14_contract,
    make_causal_multimotion_v14_env_cfg,
)
from gear_sonic.scripts import (
    train_g1_23dof_mjlab_causal_history as base,
    train_g1_23dof_mjlab_teleop_v13 as v13,
)
from gear_sonic.trl.mjlab import causal_history_runner as runner_module
from gear_sonic.trl.mjlab.causal_teleop_runner_v15 import CausalTeleopRunnerV15

_THIS_FILE = Path(__file__).resolve()
_SOURCE_FILES = (
    _THIS_FILE,
    _THIS_FILE.parents[1] / "envs" / "mjlab" / "sonic_true23_causal_multimotion_v14.py",
    _THIS_FILE.parents[1] / "trl" / "mjlab" / "causal_teleop_runner_v15.py",
    _THIS_FILE.parents[1] / "scripts" / "train_g1_23dof_mjlab_teleop_v13.py",
)


def _install_isolated_v15_hooks(span_sidecar: Path) -> None:
    from gear_sonic.envs.mjlab import sonic_true23_causal_history as causal_task

    v13._install_corpus_command(span_sidecar)  # noqa: SLF001
    runner_module.CausalHistoryMjlabOnPolicyRunner = CausalTeleopRunnerV15
    base.CausalHistoryMjlabOnPolicyRunner = CausalTeleopRunnerV15
    causal_task.make_causal_history_recovery_env_cfg = make_causal_multimotion_v14_env_cfg
    base.CAUSAL_SOURCE_FILES = tuple(dict.fromkeys((*base.CAUSAL_SOURCE_FILES, *_SOURCE_FILES)))
    original = base._resolved_training_config

    def resolved_with_v15(*args: Any, **kwargs: Any) -> dict[str, Any]:
        resolved = original(*args, **kwargs)
        resolved["causal_multimotion_v14"] = causal_multimotion_v14_contract()
        resolved["causal_teleop_v15"] = {
            "source": "official_multi_motion_init",
            "trainable": "final_affine_only",
            "deployment_ready": False,
            "hardware_authorized": False,
        }
        return resolved

    base._resolved_training_config = resolved_with_v15


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    values = list(sys.argv[1:] if argv is None else argv)
    filtered: list[str] = []
    sidecar: Path | None = None
    index = 0
    while index < len(values):
        if values[index] == "--spans":
            sidecar = Path(values[index + 1]).expanduser().resolve()
            index += 2
            continue
        filtered.append(values[index])
        index += 1
    if sidecar is None:
        for index, token in enumerate(filtered):
            if token == "--motion-file":
                sidecar = Path(filtered[index + 1]).with_suffix(".spans.json")
                break
    if sidecar is None or not sidecar.is_file():
        raise SystemExit("v15 requires exact corpus --spans sidecar")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    spans = payload.get("spans")
    if (
        not isinstance(spans, list)
        or not spans
        or payload.get("clip_count") != len(spans)
        or sum(int(span["length"]) for span in spans) != payload.get("total_frames")
    ):
        raise SystemExit("v15 corpus span identity mismatch")
    _install_isolated_v15_hooks(sidecar)
    return base.main(filtered)


if __name__ == "__main__":
    raise SystemExit(main())
