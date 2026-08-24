"""Train corrected clip-contained causal multi-motion teleop-v14 policy."""

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
from gear_sonic.trl.mjlab.causal_teleop_runner_v14 import CausalTeleopRunnerV14

_THIS_FILE = Path(__file__).resolve()
_SOURCE_FILES = (
    _THIS_FILE,
    _THIS_FILE.parents[1] / "envs" / "mjlab" / "sonic_true23_causal_multimotion_v14.py",
    _THIS_FILE.parents[1] / "trl" / "mjlab" / "causal_teleop_runner_v14.py",
    _THIS_FILE.parents[1] / "scripts" / "train_g1_23dof_mjlab_teleop_v13.py",
)


def _install_isolated_v14_hooks(span_sidecar: Path) -> None:
    from gear_sonic.envs.mjlab import sonic_true23_causal_history as causal_task

    v13._install_corpus_command(span_sidecar)  # noqa: SLF001
    runner_module.CausalHistoryMjlabOnPolicyRunner = CausalTeleopRunnerV14
    base.CausalHistoryMjlabOnPolicyRunner = CausalTeleopRunnerV14
    causal_task.make_causal_history_recovery_env_cfg = make_causal_multimotion_v14_env_cfg

    source_files = list(base.CAUSAL_SOURCE_FILES)
    for path in _SOURCE_FILES:
        if path not in source_files:
            source_files.append(path)
    base.CAUSAL_SOURCE_FILES = tuple(source_files)
    original = base._resolved_training_config

    def resolved_with_v14(*args: Any, **kwargs: Any) -> dict[str, Any]:
        resolved = original(*args, **kwargs)
        resolved["causal_multimotion_v14"] = causal_multimotion_v14_contract()
        return resolved

    base._resolved_training_config = resolved_with_v14


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
        raise SystemExit("v14 requires exact corpus --spans sidecar")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    clip_count = payload.get("clip_count")
    total_frames = payload.get("total_frames")
    spans = payload.get("spans")
    if (
        isinstance(clip_count, bool)
        or not isinstance(clip_count, int)
        or clip_count <= 0
        or isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames <= 12
        or not isinstance(spans, list)
        or len(spans) != clip_count
        or sum(int(span["length"]) for span in spans) != total_frames
    ):
        raise SystemExit("v14 corpus span identity mismatch")
    _install_isolated_v14_hooks(sidecar)
    return base.main(filtered)


if __name__ == "__main__":
    raise SystemExit(main())
