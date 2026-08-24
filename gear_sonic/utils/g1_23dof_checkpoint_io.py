"""Safe, weights-only checkpoint I/O for native G1 true23 artifacts.

Promotion, readiness, export, evaluation, and weights-only initialization must
never deserialize the arbitrary Python objects present in a full trainer resume
checkpoint.  Those trusted local resume files remain a separate format.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

SAFE_CHECKPOINT_SCHEMA_VERSION = 1
SAFE_CHECKPOINT_KIND = "g1_23dof_safe_weights_checkpoint"
SAFE_CHECKPOINT_HEADER = "g1_23dof_safe_checkpoint"
INITIALIZATION_STAGE = "checkpoint_initialization"
TRAINED_STAGE = "trained"

_HEADER_KEYS = {
    "schema_version",
    "kind",
    "checkpoint_stage",
    "resume_state_included",
}
_INITIALIZATION_KEYS = {
    SAFE_CHECKPOINT_HEADER,
    "policy_state_dict",
    "g1_23dof_metadata",
    "g1_23dof_initialization_report",
}
_TRAINED_KEYS = {
    SAFE_CHECKPOINT_HEADER,
    "policy_state_dict",
    "state",
    "g1_23dof_metadata",
    "g1_23dof_training_evidence",
}
_LEGACY_INITIALIZATION_KEYS = {
    "policy_state_dict",
    "g1_23dof_metadata",
    "g1_23dof_initialization_report",
}


def promotion_checkpoint_path(resume_checkpoint_path: str | Path) -> Path:
    """Return the separate safe promotion filename for a trainer checkpoint."""

    path = Path(resume_checkpoint_path)
    suffix = path.suffix or ".pt"
    return path.with_name(f"{path.stem}.promotion{suffix}")


def _primitive_copy(value: Any, context: str) -> Any:
    value_type = type(value)
    if value is None or value_type in {bool, int, float, str}:
        return value
    if value_type in {list, tuple}:
        return [
            _primitive_copy(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError(f"{context} keys must be non-empty strings")
            result[key] = _primitive_copy(item, f"{context}.{key}")
        return result
    raise ValueError(
        f"{context} contains unsafe value type {value_type.__module__}.{value_type.__qualname__}"
    )


def _policy_tensor_copy(policy_state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(policy_state, Mapping) or not policy_state:
        raise ValueError("policy_state_dict must be a non-empty mapping")
    result: dict[str, torch.Tensor] = {}
    for key, value in policy_state.items():
        if type(key) is not str or not key:
            raise ValueError("policy_state_dict keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"policy_state_dict[{key!r}] must be a tensor")
        result[key] = value.detach().cpu().clone()
    return result


def _header(stage: str) -> dict[str, Any]:
    if stage not in {INITIALIZATION_STAGE, TRAINED_STAGE}:
        raise ValueError(f"unsupported safe checkpoint stage: {stage!r}")
    return {
        "schema_version": SAFE_CHECKPOINT_SCHEMA_VERSION,
        "kind": SAFE_CHECKPOINT_KIND,
        "checkpoint_stage": stage,
        "resume_state_included": False,
    }


def build_safe_initialization_checkpoint(
    *,
    policy_state_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    initialization_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a portable initialization checkpoint with no trainer objects."""

    result = {
        SAFE_CHECKPOINT_HEADER: _header(INITIALIZATION_STAGE),
        "policy_state_dict": _policy_tensor_copy(policy_state_dict),
        "g1_23dof_metadata": _primitive_copy(metadata, "g1_23dof_metadata"),
        "g1_23dof_initialization_report": _primitive_copy(
            initialization_report,
            "g1_23dof_initialization_report",
        ),
    }
    validate_safe_true23_checkpoint(result)
    return result


def build_safe_promotion_checkpoint(
    *,
    policy_state_dict: Mapping[str, Any],
    global_step: int,
    metadata: Mapping[str, Any],
    training_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the only true23 checkpoint format accepted by promotion paths."""

    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step <= 0:
        raise ValueError("trained safe checkpoint requires global_step > 0")
    result = {
        SAFE_CHECKPOINT_HEADER: _header(TRAINED_STAGE),
        "policy_state_dict": _policy_tensor_copy(policy_state_dict),
        "state": {"global_step": global_step},
        "g1_23dof_metadata": _primitive_copy(metadata, "g1_23dof_metadata"),
        "g1_23dof_training_evidence": _primitive_copy(
            training_evidence,
            "g1_23dof_training_evidence",
        ),
    }
    validate_safe_true23_checkpoint(result)
    return result


def checkpoint_stage(checkpoint: Mapping[str, Any]) -> str:
    """Return the validated metadata stage from a safe checkpoint."""

    metadata = checkpoint.get("g1_23dof_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("safe checkpoint lacks g1_23dof_metadata")
    stage = metadata.get("checkpoint_stage")
    if stage not in {INITIALIZATION_STAGE, TRAINED_STAGE}:
        raise ValueError(f"unsupported checkpoint_stage: {stage!r}")
    return str(stage)


def extract_global_step(checkpoint: Mapping[str, Any]) -> int:
    """Extract the positive step from the safe primitive ``state`` mapping."""

    state = checkpoint.get("state")
    if not isinstance(state, Mapping) or set(state) != {"global_step"}:
        raise ValueError("trained safe checkpoint state must contain only global_step")
    global_step = state.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step <= 0:
        raise ValueError("trained safe checkpoint requires state.global_step > 0")
    return global_step


def _validate_header(header: Any, stage: str) -> None:
    if not isinstance(header, Mapping) or set(header) != _HEADER_KEYS:
        raise ValueError("safe checkpoint header has unexpected keys")
    expected = _header(stage)
    if dict(header) != expected:
        raise ValueError("safe checkpoint header does not match stage/schema")


def _validate_policy_tensors(policy_state: Any) -> None:
    if not isinstance(policy_state, Mapping) or not policy_state:
        raise ValueError("policy_state_dict must be a non-empty mapping")
    for key, value in policy_state.items():
        if type(key) is not str or not key or not isinstance(value, torch.Tensor):
            raise ValueError("policy_state_dict must contain only named tensors")


def validate_safe_true23_checkpoint(
    checkpoint: Any,
    *,
    allow_legacy_initialization: bool = False,
) -> Mapping[str, Any]:
    """Reject resume objects, unknown globals, and schema drift."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError("safe checkpoint root must be a mapping")

    if SAFE_CHECKPOINT_HEADER not in checkpoint:
        if not allow_legacy_initialization or set(checkpoint) != _LEGACY_INITIALIZATION_KEYS:
            raise ValueError("checkpoint lacks the safe true23 schema header")
        stage = checkpoint_stage(checkpoint)
        if stage != INITIALIZATION_STAGE:
            raise ValueError("only legacy weights-only initialization is accepted")
        _validate_policy_tensors(checkpoint["policy_state_dict"])
        _primitive_copy(checkpoint["g1_23dof_metadata"], "g1_23dof_metadata")
        _primitive_copy(
            checkpoint["g1_23dof_initialization_report"],
            "g1_23dof_initialization_report",
        )
        return checkpoint

    stage = checkpoint_stage(checkpoint)
    expected_keys = _TRAINED_KEYS if stage == TRAINED_STAGE else _INITIALIZATION_KEYS
    if set(checkpoint) != expected_keys:
        raise ValueError(
            f"safe checkpoint_stage={stage!r} checkpoint has unexpected root keys: "
            f"{sorted(set(checkpoint) ^ expected_keys)}"
        )
    _validate_header(checkpoint[SAFE_CHECKPOINT_HEADER], stage)
    _validate_policy_tensors(checkpoint["policy_state_dict"])
    _primitive_copy(checkpoint["g1_23dof_metadata"], "g1_23dof_metadata")
    if stage == TRAINED_STAGE:
        extract_global_step(checkpoint)
        _primitive_copy(
            checkpoint["g1_23dof_training_evidence"],
            "g1_23dof_training_evidence",
        )
    else:
        _primitive_copy(
            checkpoint["g1_23dof_initialization_report"],
            "g1_23dof_initialization_report",
        )
    return checkpoint


def load_safe_true23_checkpoint(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    allow_legacy_initialization: bool = False,
) -> Mapping[str, Any]:
    """Load true23 weights without permitting arbitrary pickle globals."""

    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError(
            "checkpoint is not a safe weights-only true23 artifact; "
            "use the separate *.promotion.pt checkpoint, not a trainer resume file"
        ) from exc
    return validate_safe_true23_checkpoint(
        checkpoint,
        allow_legacy_initialization=allow_legacy_initialization,
    )


def load_true23_trainer_checkpoint(
    path: str | Path,
    *,
    resume: bool,
    map_location: Any,
) -> Mapping[str, Any]:
    """Load true23 policy initialization safely or an explicit trusted resume.

    ``resume=False`` accepts only the weights-only initialization/promotion
    schemas. ``resume=True`` is the sole deliberate arbitrary-pickle boundary
    and is reserved for a trusted local full trainer checkpoint.
    """

    if type(resume) is not bool:
        raise TypeError("resume must be an explicit bool")
    if not resume:
        return load_safe_true23_checkpoint(
            path,
            map_location=map_location,
            allow_legacy_initialization=True,
        )

    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("trusted full trainer checkpoint root must be a mapping")
    if SAFE_CHECKPOINT_HEADER in checkpoint:
        raise ValueError(
            "safe true23 initialization/promotion checkpoints contain no "
            "trainer state and cannot be used with resume=True"
        )
    return checkpoint
