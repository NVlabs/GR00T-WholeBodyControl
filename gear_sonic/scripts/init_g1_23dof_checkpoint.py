#!/usr/bin/env python3
"""Build offline true-23 SONIC initialization checkpoint; never talks to robot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
from types import MappingProxyType
from typing import Any, Mapping

import torch

from gear_sonic.trl.utils.g1_23dof_checkpoint import initialize_checkpoint
from gear_sonic.utils.g1_23dof_contract import (
    APPROVED_WARM_START_RELEASES,
    LOW_LATENCY_RELEASE_HF_REVISION,
    LOW_LATENCY_RELEASE_SHA256,
    NORMAL_RELEASE_SHA256,
    OBS_LAYOUT_CHECKPOINT_INIT_29,
    OBS_LAYOUT_NATIVE_23,
)

LEGACY_SONIC_RELEASE_SHA256 = (
    NORMAL_RELEASE_SHA256
)
LOW_LATENCY_SONIC_RELEASE_SHA256 = (
    LOW_LATENCY_RELEASE_SHA256
)
LOW_LATENCY_SONIC_RELEASE_HF_REVISION = (
    LOW_LATENCY_RELEASE_HF_REVISION
)
PINNED_LEGACY_RELEASES = MappingProxyType(
    {
        digest: MappingProxyType(dict(release))
        for digest, release in APPROVED_WARM_START_RELEASES.items()
    }
)
PINNED_LEGACY_SONIC_RELEASE_SHA256 = frozenset(PINNED_LEGACY_RELEASES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_legacy_trl_pickle_shims() -> None:
    """Provide discarded trainer-state class names when TRL is unavailable."""

    class PPOConfig:
        pass

    class OnlineTrainerState:
        pass

    if "trl" not in sys.modules and importlib.util.find_spec("trl") is None:
        modules = {
            "trl": types.ModuleType("trl"),
            "trl.trainer": types.ModuleType("trl.trainer"),
            "trl.trainer.ppo_config": types.ModuleType("trl.trainer.ppo_config"),
            "trl.trainer.utils": types.ModuleType("trl.trainer.utils"),
            "trl.experimental": types.ModuleType("trl.experimental"),
            "trl.experimental.ppo": types.ModuleType("trl.experimental.ppo"),
            "trl.experimental.ppo.ppo_trainer": types.ModuleType(
                "trl.experimental.ppo.ppo_trainer"
            ),
        }
        PPOConfig.__module__ = "trl.trainer.ppo_config"
        modules["trl.trainer.ppo_config"].PPOConfig = PPOConfig
        OnlineTrainerState.__module__ = "trl.trainer.utils"
        modules["trl.trainer.utils"].OnlineTrainerState = OnlineTrainerState

        class ExperimentalOnlineTrainerState:
            pass

        ExperimentalOnlineTrainerState.__module__ = (
            "trl.experimental.ppo.ppo_trainer"
        )
        modules[
            "trl.experimental.ppo.ppo_trainer"
        ].OnlineTrainerState = ExperimentalOnlineTrainerState
        modules["trl"].trainer = modules["trl.trainer"]
        modules["trl"].experimental = modules["trl.experimental"]
        modules["trl.trainer"].ppo_config = modules["trl.trainer.ppo_config"]
        modules["trl.trainer"].utils = modules["trl.trainer.utils"]
        modules["trl.experimental"].ppo = modules["trl.experimental.ppo"]
        modules["trl.experimental.ppo"].ppo_trainer = modules[
            "trl.experimental.ppo.ppo_trainer"
        ]
        sys.modules.update(modules)

    class DiscardedLegacyValue:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def __setstate__(self, state: Any) -> None:
            if isinstance(state, Mapping):
                self.__dict__.update(state)
            else:
                self.state = state

    if (
        "transformers" not in sys.modules
        and importlib.util.find_spec("transformers") is None
    ):
        transformers_modules = {
            "transformers": types.ModuleType("transformers"),
            "transformers.training_args": types.ModuleType(
                "transformers.training_args"
            ),
            "transformers.trainer_utils": types.ModuleType(
                "transformers.trainer_utils"
            ),
            "transformers.trainer_pt_utils": types.ModuleType(
                "transformers.trainer_pt_utils"
            ),
        }
        transformers_modules[
            "transformers.training_args"
        ].OptimizerNames = DiscardedLegacyValue
        for name in (
            "SaveStrategy",
            "HubStrategy",
            "IntervalStrategy",
            "SchedulerType",
        ):
            setattr(
                transformers_modules["transformers.trainer_utils"],
                name,
                DiscardedLegacyValue,
            )
        transformers_modules[
            "transformers.trainer_pt_utils"
        ].AcceleratorConfig = DiscardedLegacyValue
        transformers_modules["transformers"].training_args = (
            transformers_modules["transformers.training_args"]
        )
        transformers_modules["transformers"].trainer_utils = (
            transformers_modules["transformers.trainer_utils"]
        )
        transformers_modules["transformers"].trainer_pt_utils = (
            transformers_modules["transformers.trainer_pt_utils"]
        )
        sys.modules.update(transformers_modules)

    if (
        "accelerate" not in sys.modules
        and importlib.util.find_spec("accelerate") is None
    ):
        accelerate_modules = {
            "accelerate": types.ModuleType("accelerate"),
            "accelerate.utils": types.ModuleType("accelerate.utils"),
            "accelerate.utils.dataclasses": types.ModuleType(
                "accelerate.utils.dataclasses"
            ),
            "accelerate.state": types.ModuleType("accelerate.state"),
        }
        accelerate_modules[
            "accelerate.utils.dataclasses"
        ].DistributedType = DiscardedLegacyValue
        accelerate_modules["accelerate.state"].PartialState = (
            DiscardedLegacyValue
        )
        accelerate_modules["accelerate"].utils = accelerate_modules[
            "accelerate.utils"
        ]
        accelerate_modules["accelerate"].state = accelerate_modules[
            "accelerate.state"
        ]
        accelerate_modules["accelerate.utils"].dataclasses = (
            accelerate_modules["accelerate.utils.dataclasses"]
        )
        sys.modules.update(accelerate_modules)


def _load_pinned_legacy_release(
    path: Path,
) -> tuple[Mapping[str, Any], str, Mapping[str, str | None]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"legacy release checkpoint missing: {resolved}")
    actual_sha256 = _sha256(resolved)
    release = PINNED_LEGACY_RELEASES.get(actual_sha256)
    if release is None:
        raise ValueError(
            "refusing unsafe legacy checkpoint load: SHA-256 mismatch "
            f"(got {actual_sha256}; no exact pinned release match)"
        )
    _install_legacy_trl_pickle_shims()
    source = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(source, Mapping):
        raise ValueError("pinned legacy release checkpoint root is not a mapping")
    return source, actual_sha256, release


def load_pinned_legacy_release(path: Path) -> Mapping[str, Any]:
    """Unsafe-load only one exact audited legacy SONIC release checkpoint."""

    source, _, _ = _load_pinned_legacy_release(path)
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument(
        "--observation-layout",
        choices=(OBS_LAYOUT_CHECKPOINT_INIT_29, OBS_LAYOUT_NATIVE_23),
        default=OBS_LAYOUT_CHECKPOINT_INIT_29,
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    source, source_sha256, release = _load_pinned_legacy_release(args.source)
    initialized, report = initialize_checkpoint(
        source,
        history_length=args.history_length,
        target_observation_layout=args.observation_layout,
        reference_profile=str(release["reference_profile"]),
        source_checkpoint_sha256=source_sha256,
        source_revision=release["source_revision"],
        source_family=str(release["source_family"]),
    )
    from gear_sonic.utils.g1_23dof_artifact import (
        inspect_true23_policy_state,
    )

    initial_policy_state_sha256 = inspect_true23_policy_state(initialized)
    if (
        initial_policy_state_sha256
        != release["initial_policy_state_sha256"]
    ):
        raise ValueError(
            "converted true23 initialization policy hash differs from "
            "exact approved release conversion"
        )
    report["initial_policy_state_sha256"] = initial_policy_state_sha256
    initialized["g1_23dof_initialization_report"][
        "initial_policy_state_sha256"
    ] = initial_policy_state_sha256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(initialized, args.output)
    # Re-open through the constrained unpickler before announcing success.
    from gear_sonic.utils.g1_23dof_checkpoint_io import (
        load_safe_true23_checkpoint,
    )

    load_safe_true23_checkpoint(args.output)
    print(f"wrote initialization-only checkpoint: {args.output}")  # noqa: T201
    print(report)  # noqa: T201


if __name__ == "__main__":
    main()
