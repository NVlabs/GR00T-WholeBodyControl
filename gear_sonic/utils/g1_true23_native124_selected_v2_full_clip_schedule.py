"""Pinned causal coverage schedule for full DadDance source qualification.

This module contains no simulator runner.  It defines the only admitted
continuous window and the overlapping fixed-phase windows needed to localize
reset-seam behavior without confusing phase restartability with continuous
whole-clip reachability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes, sha256_bytes

CONTROL_HZ = 50
MOTION_FRAME_COUNT = 2090
CAUSAL_MIN_Q9 = 9
CAUSAL_MAX_Q9 = MOTION_FRAME_COUNT - 2
FINAL_Q10_PROOF = MOTION_FRAME_COUNT - 1
FULL_CLIP_TRANSITIONS = CAUSAL_MAX_Q9 - CAUSAL_MIN_Q9 + 1
PHASE_OVERLAP_TRANSITIONS = 100

CONTINUOUS_SCHEMA = "g1_true23_daddance_full_clip_continuous_v1"
PHASE_SCHEDULE_SCHEMA = "g1_true23_daddance_full_phase_schedule_v1"
QUALIFICATION_SCHEDULE_SCHEMA = "g1_true23_daddance_full_clip_qualification_schedule_v1"

CONTINUOUS_CONTRACT_SHA256 = "2aacae50b0f70ffed9b287706ea76067eac00d0ae2363c72a405b233bb096ae9"
PHASE_SCHEDULE_SHA256 = "02ced13dadbfa7a183268d3493f81a46e997a6fc431f2624a34fa0c7bd0bae52"
QUALIFICATION_SCHEDULE_SHA256 = "fcc1d58307ad749b719d1610e31efd151d3f45cfdfc04ce081e4bad90c9d38e2"


def _require_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


@dataclass(frozen=True)
class CausalQualificationWindow:
    """One deterministic q9 reset window and its post-seam scored range."""

    window_id: str
    anchor_q9: int
    transitions: int
    burn_in_transitions: int

    def __post_init__(self) -> None:
        if not isinstance(self.window_id, str) or not self.window_id:
            raise ValueError("window_id must be a non-empty string")
        anchor = _require_int(self.anchor_q9, "anchor_q9")
        transitions = _require_int(self.transitions, "transitions")
        burn_in = _require_int(self.burn_in_transitions, "burn_in_transitions")
        if not CAUSAL_MIN_Q9 <= anchor <= CAUSAL_MAX_Q9:
            raise ValueError("anchor_q9 is outside the causal q9 domain")
        if transitions <= 0:
            raise ValueError("transitions must be positive")
        if not 0 <= burn_in < transitions:
            raise ValueError("burn_in_transitions must be within the window")
        if self.last_q9 > CAUSAL_MAX_Q9:
            raise ValueError("window loses final q10 proof")

    @property
    def last_q9(self) -> int:
        return self.anchor_q9 + self.transitions - 1

    @property
    def first_scored_q9(self) -> int:
        return self.anchor_q9 + self.burn_in_transitions

    @property
    def score_q9(self) -> tuple[int, int]:
        return self.first_scored_q9, self.last_q9

    def to_phase_record(self) -> dict[str, Any]:
        return {
            "anchor_q9": self.anchor_q9,
            "burn_in_transitions": self.burn_in_transitions,
            "id": self.window_id,
            "score_q9": list(self.score_q9),
            "transitions": self.transitions,
        }


FULL_CLIP_CONTINUOUS_WINDOW = CausalQualificationWindow(
    window_id="continuous",
    anchor_q9=CAUSAL_MIN_Q9,
    transitions=FULL_CLIP_TRANSITIONS,
    burn_in_transitions=0,
)

PHASE_WINDOWS = (
    CausalQualificationWindow("w0", 9, 500, 0),
    CausalQualificationWindow("w1", 409, 500, 100),
    CausalQualificationWindow("w2", 809, 500, 100),
    CausalQualificationWindow("w3", 1209, 500, 100),
    CausalQualificationWindow("w4", 1609, 480, 100),
)


def _validate_pinned_windows() -> None:
    continuous = FULL_CLIP_CONTINUOUS_WINDOW
    if (
        continuous.anchor_q9 != CAUSAL_MIN_Q9
        or continuous.last_q9 != CAUSAL_MAX_Q9
        or continuous.transitions != FULL_CLIP_TRANSITIONS
        or continuous.last_q9 + 1 != FINAL_Q10_PROOF
    ):
        raise RuntimeError("continuous DadDance causal coverage drift")

    cursor = CAUSAL_MIN_Q9
    previous: CausalQualificationWindow | None = None
    for window in PHASE_WINDOWS:
        if window.first_scored_q9 != cursor:
            raise RuntimeError("phase primary score coverage has a gap or duplicate")
        if previous is not None:
            overlap = previous.last_q9 - window.anchor_q9 + 1
            if overlap != PHASE_OVERLAP_TRANSITIONS:
                raise RuntimeError("phase action overlap drift")
        cursor = window.last_q9 + 1
        previous = window
    if cursor != CAUSAL_MAX_Q9 + 1:
        raise RuntimeError("phase primary score coverage does not reach final q9")


def continuous_contract() -> dict[str, Any]:
    """Return fresh canonical full-domain continuous-rollout contract."""

    _validate_pinned_windows()
    window = FULL_CLIP_CONTINUOUS_WINDOW
    return {
        "anchor_q9": window.anchor_q9,
        "causal_max_q9": CAUSAL_MAX_Q9,
        "causal_min_q9": CAUSAL_MIN_Q9,
        "control_hz": CONTROL_HZ,
        "motion_frame_count": MOTION_FRAME_COUNT,
        "q10_proof_last": FINAL_Q10_PROOF,
        "schema": CONTINUOUS_SCHEMA,
        "score_q9": list(window.score_q9),
        "transitions": window.transitions,
    }


def phase_schedule_contract() -> dict[str, Any]:
    """Return fresh canonical overlapping fixed-phase schedule."""

    _validate_pinned_windows()
    return {
        "causal_max_q9": CAUSAL_MAX_Q9,
        "causal_min_q9": CAUSAL_MIN_Q9,
        "control_hz": CONTROL_HZ,
        "motion_frame_count": MOTION_FRAME_COUNT,
        "overlap_transitions": PHASE_OVERLAP_TRANSITIONS,
        "schema": PHASE_SCHEDULE_SCHEMA,
        "windows": [window.to_phase_record() for window in PHASE_WINDOWS],
    }


def qualification_schedule_contract() -> dict[str, Any]:
    """Bind continuous proof and phase-local diagnostics without conflating them."""

    return {
        "continuous": continuous_contract(),
        "phase_schedule": phase_schedule_contract(),
        "schema": QUALIFICATION_SCHEDULE_SCHEMA,
    }


def _require_pinned_contract(
    value: Mapping[str, Any],
    *,
    expected: dict[str, Any],
    expected_sha256: str,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    supplied = dict(value)
    if supplied != expected:
        raise ValueError(f"{context} differs from the pinned schedule")
    actual_hash = sha256_bytes(canonical_json_bytes(supplied))
    if actual_hash != expected_sha256:
        raise RuntimeError(f"{context} canonical SHA256 drift")
    return expected


def validate_continuous_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    return _require_pinned_contract(
        value,
        expected=continuous_contract(),
        expected_sha256=CONTINUOUS_CONTRACT_SHA256,
        context="continuous contract",
    )


def validate_phase_schedule_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    return _require_pinned_contract(
        value,
        expected=phase_schedule_contract(),
        expected_sha256=PHASE_SCHEDULE_SHA256,
        context="phase schedule contract",
    )


def validate_qualification_schedule_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    return _require_pinned_contract(
        value,
        expected=qualification_schedule_contract(),
        expected_sha256=QUALIFICATION_SCHEDULE_SHA256,
        context="qualification schedule contract",
    )


__all__ = [
    "CAUSAL_MAX_Q9",
    "CAUSAL_MIN_Q9",
    "CONTINUOUS_CONTRACT_SHA256",
    "CausalQualificationWindow",
    "FINAL_Q10_PROOF",
    "FULL_CLIP_CONTINUOUS_WINDOW",
    "FULL_CLIP_TRANSITIONS",
    "MOTION_FRAME_COUNT",
    "PHASE_OVERLAP_TRANSITIONS",
    "PHASE_SCHEDULE_SHA256",
    "PHASE_WINDOWS",
    "QUALIFICATION_SCHEDULE_SHA256",
    "continuous_contract",
    "phase_schedule_contract",
    "qualification_schedule_contract",
    "validate_continuous_contract",
    "validate_phase_schedule_contract",
    "validate_qualification_schedule_contract",
]
