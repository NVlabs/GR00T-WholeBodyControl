"""One fail-closed q200 disturbance screen using push-trained model21248."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import numpy as np
import onnxruntime as ort
import torch

from gear_sonic.utils import (
    g1_true23_sonic_rank256_q200_recovery_qualification_campaign as q200_campaign,
    g1_true23_sonic_student_teacher_recovery as recovery,
)
from gear_sonic.utils.g1_23dof_native124_actor_export import load_native124_actor

CHECKPOINT_RELATIVE_PATH = Path(
    "artifacts/g1_native124_multimotion/scaling_all61/"
    "balanced_recovery/run2_2048/model_21248.pt"
)
CHECKPOINT_SHA256 = "e4b6862143f4d5bd9868a4dac1c74908550d5416df91c37ab5d00e581d3f8e97"
ACTOR_STATE_SHA256 = "e75f3171734e7685f74c7359ffa657963e43801950063faabe9eb496ddb3672c"
ONNX_RELATIVE_PATH = Path(
    "artifacts/g1_native124_multimotion/scaling_all61/"
    "balanced_recovery/run2_2048/policy.onnx"
)
ONNX_SHA256 = "1ab3be110ee9246fabc58da79b92255bf388f58c9ebbe6bfc9a384aa5e61ec16"
FAILED_CAMPAIGN_RELATIVE_PATH = Path(
    "artifacts/g1_true23/sonic_rank256_q200_recovery_qualification_campaign_v1.json"
)
FAILED_CAMPAIGN_SHA256 = "33981a7d602725846dd181161fb5f986fe5540d2dee23ff3d264cf314469a315"
SCHEMA_VERSION = 1
KIND = "g1_true23_sonic_rank256_q200_balanced_teacher_disturbance_screen_v1"


def _bound(root: Path, relative: Path, expected: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if path.is_symlink() or not path.is_file() or recovery.sha256_file(path) != expected:
        raise ValueError(f"balanced screen bound input mismatch: {relative}")
    return path


class _BalancedTeacherPair:
    def __init__(self, binding: Any, device: str) -> None:
        self.device = torch.device(device)
        actor, lineage = load_native124_actor(binding.checkpoint_path)
        if lineage["actor_state"]["sha256"] != ACTOR_STATE_SHA256:
            raise ValueError("balanced teacher actor-state mismatch")
        self.actor = actor.to(self.device)
        self.actor.eval()
        self.actor.requires_grad_(False)
        self.session = ort.InferenceSession(
            str(binding.onnx_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].name != "obs"
            or inputs[0].shape != [1, 124]
            or len(outputs) != 1
            or outputs[0].name != "actions"
            or outputs[0].shape != [1, 23]
        ):
            raise ValueError("balanced teacher ONNX ABI mismatch")
        self.state_sha256_before = recovery.tensor_state_sha256(self.actor.state_dict())
        self.query_count = 0
        self.violation_count = 0
        self.maximum_absolute_error = 0.0
        self.total_absolute_error = 0.0
        self.maximum_pt_duration_ms = 0.0
        self.maximum_onnx_duration_ms = 0.0

    def infer(self, selected124: torch.Tensor) -> tuple[np.ndarray, float]:
        if selected124.shape != (1, 124) or selected124.dtype != torch.float32:
            raise ValueError("balanced teacher observation ABI mismatch")
        observation = selected124.detach().cpu().contiguous().numpy().copy()
        started = time.perf_counter_ns()
        with torch.inference_mode():
            pt_tensor = self.actor(selected124.to(self.device, dtype=torch.float32))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        pt_ms = (time.perf_counter_ns() - started) / 1.0e6
        pt = pt_tensor.detach().cpu().contiguous().numpy()[0].copy()
        started = time.perf_counter_ns()
        onnx = np.asarray(self.session.run(None, {"obs": observation})[0][0], dtype=np.float32)
        onnx_ms = (time.perf_counter_ns() - started) / 1.0e6
        if pt.shape != (23,) or onnx.shape != (23,) or not np.isfinite(pt).all() or not np.isfinite(onnx).all():
            raise ValueError("balanced teacher produced invalid action")
        error = float(np.max(np.abs(pt.astype(np.float64) - onnx.astype(np.float64))))
        self.query_count += 1
        self.violation_count += int(error > recovery.TEACHER_PARITY_ATOL)
        self.maximum_absolute_error = max(self.maximum_absolute_error, error)
        self.total_absolute_error += error
        self.maximum_pt_duration_ms = max(self.maximum_pt_duration_ms, pt_ms)
        self.maximum_onnx_duration_ms = max(self.maximum_onnx_duration_ms, onnx_ms)
        return onnx, error

    def report(self) -> dict[str, Any]:
        state_after = recovery.tensor_state_sha256(self.actor.state_dict())
        return {
            "query_count": self.query_count,
            "pt_device": str(self.device),
            "onnx_provider": "CPUExecutionProvider",
            "maximum_absolute_error": self.maximum_absolute_error,
            "mean_transition_maximum_absolute_error": (
                None if self.query_count == 0 else self.total_absolute_error / self.query_count
            ),
            "violation_count": self.violation_count,
            "threshold": recovery.TEACHER_PARITY_ATOL,
            "maximum_pt_inference_duration_ms_evidence_only": self.maximum_pt_duration_ms,
            "maximum_onnx_inference_duration_ms_evidence_only": self.maximum_onnx_duration_ms,
            "actor_state_sha256_before": self.state_sha256_before,
            "actor_state_sha256_after": state_after,
            "actor_state_unchanged": state_after == self.state_sha256_before,
            "passed": bool(
                self.query_count == recovery.TOTAL_TRANSITIONS
                and self.violation_count == 0
                and self.maximum_absolute_error <= recovery.TEACHER_PARITY_ATOL
                and state_after == self.state_sha256_before
            ),
        }


@contextmanager
def _scope(root: Path) -> Iterator[None]:
    checkpoint = _bound(root, CHECKPOINT_RELATIVE_PATH, CHECKPOINT_SHA256)
    onnx = _bound(root, ONNX_RELATIVE_PATH, ONNX_SHA256)
    _bound(root, FAILED_CAMPAIGN_RELATIVE_PATH, FAILED_CAMPAIGN_SHA256)
    binding = SimpleNamespace(
        checkpoint_path=checkpoint,
        onnx_path=onnx,
        checkpoint_sha256=CHECKPOINT_SHA256,
        actor_state_sha256=ACTOR_STATE_SHA256,
        onnx_sha256=ONNX_SHA256,
    )
    names = (
        "CHECKPOINT_SHA256",
        "ACTOR_STATE_SHA256",
        "ONNX_SHA256",
        "TEACHER_ITERATION",
        "load_checkpoint21204_binding",
        "_ExactTeacherPair",
        "load_recovery_contract",
    )
    saved = {name: getattr(recovery, name) for name in names}
    campaign_saved_loader = q200_campaign.campaign.load_campaign_contract
    recovery_contract = copy.deepcopy(saved["load_recovery_contract"](root))
    recovery_contract["teacher"] = {
        "iteration": 21248,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "actor_state_sha256": ACTOR_STATE_SHA256,
        "onnx_sha256": ONNX_SHA256,
        "pytorch_device": recovery.student.DEVICE,
        "onnx_provider": "CPUExecutionProvider",
        "maximum_pt_onnx_absolute_error": recovery.TEACHER_PARITY_ATOL,
        "query_every_transition": True,
    }
    campaign_contract = copy.deepcopy(campaign_saved_loader(root))
    campaign_contract["prerequisites"]["teacher_checkpoint_sha256"] = CHECKPOINT_SHA256
    campaign_contract["prerequisites"]["teacher_actor_state_sha256"] = ACTOR_STATE_SHA256
    campaign_contract["prerequisites"]["teacher_onnx_sha256"] = ONNX_SHA256

    def load_binding(_root: Path) -> Any:
        return binding

    def load_recovery(_root: Path) -> Mapping[str, Any]:
        recovery._validate_recovery_contract(recovery_contract)  # noqa: SLF001
        return copy.deepcopy(recovery_contract)

    def load_campaign(_root: str | Path | None = None) -> Mapping[str, Any]:
        q200_campaign.campaign._validate_contract(campaign_contract)  # noqa: SLF001
        return copy.deepcopy(campaign_contract)

    try:
        recovery.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
        recovery.ACTOR_STATE_SHA256 = ACTOR_STATE_SHA256
        recovery.ONNX_SHA256 = ONNX_SHA256
        recovery.TEACHER_ITERATION = 21248
        recovery.load_checkpoint21204_binding = load_binding
        recovery._ExactTeacherPair = _BalancedTeacherPair  # type: ignore[assignment]  # noqa: SLF001
        recovery.load_recovery_contract = load_recovery
        q200_campaign.campaign.load_campaign_contract = load_campaign
        yield
    finally:
        q200_campaign.campaign.load_campaign_contract = campaign_saved_loader
        for name, value in saved.items():
            setattr(recovery, name, value)


def run(*, repository_root: Path, output: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    target = output if output.is_absolute() else root / output
    if os.path.lexists(target):
        raise FileExistsError("balanced teacher screen output exists")
    with q200_campaign._scope(root):  # noqa: SLF001
        with _scope(root):
            request = q200_campaign.campaign.CampaignRequest(root, target)
            preflight = q200_campaign.campaign._preflight_internal(request)  # noqa: SLF001
            if preflight.get("ready") is not True:
                raise RuntimeError("balanced teacher screen preflight not ready")
            spec = q200_campaign.campaign.campaign_run_specs(root)[1]
            record = q200_campaign.campaign._execute_one(  # noqa: SLF001
                spec=spec,
                request=preflight["base_request"],
                cached_preflight=preflight["recovery_preflight"],
                contract=preflight["contract"],
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "passed": record.get("passed") is True,
        "verdict": (
            "balanced_teacher_screen_passed"
            if record.get("passed") is True
            else "balanced_teacher_screen_failed"
        ),
        "failed_campaign_sha256": FAILED_CAMPAIGN_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "actor_state_sha256": ACTOR_STATE_SHA256,
        "onnx_sha256": ONNX_SHA256,
        "candidate_decoder_sha256": q200_campaign.q200.adapter.CANDIDATE_DECODER_SHA256,
        "seed": record["seed"],
        "impulse_apply_transition": record["impulse_apply_transition"],
        "impulse_apply_q9": record["impulse_apply_q9"],
        "attempted_transitions": record["attempted_transitions"],
        "terminal_q9": record["terminal_q9"],
        "first_issue": record["first_issue"],
        "teacher_parity_max_absolute_error": record["teacher_parity_max_absolute_error"],
        "minimum_base_height_m": record["minimum_base_height_m"],
        "maximum_base_tilt_rad": record["maximum_base_tilt_rad"],
        "maximum_tracking_rmse_rad": record["maximum_tracking_rmse_rad"],
        "hard_safety_violation_count": record["hard_safety_violation_count"],
        "soft_safety_warning_count": record["soft_safety_warning_count"],
        "published_teacher_label_count": 0,
        "training_performed": False,
        "support_qualified": False,
        "hardware_authorized": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return report


__all__ = ["ACTOR_STATE_SHA256", "CHECKPOINT_SHA256", "ONNX_SHA256", "run"]
