from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_causal_history_safe_target_v11 import (
    evaluator_aligned_recovery_metric,
)
from gear_sonic.utils import (
    g1_true23_native124_selected_v2_ankle_evaluation as evaluation,
    g1_true23_sonic_recovery_qualification_campaign as campaign,
)

ROOT = Path(__file__).resolve().parents[2]
PASS_REPORT_PATH = (
    ROOT / "artifacts/g1_true23/g1_true23_sonic_student_teacher_recovery_cutoff50_seed20260805_v3.json"
)


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pass_report() -> dict[str, Any]:
    return json.loads(PASS_REPORT_PATH.read_text(encoding="utf-8"))


def _probe_report(
    spec: campaign.CampaignRunSpec,
    full_report: dict[str, Any],
) -> dict[str, Any]:
    stats = full_report["rollout"]["reward_rate_by_term"]["evaluator_aligned_recovery"]
    disturbed = spec.scenario == "disturbance"
    result = {
        "effective_seed": spec.seed,
        "effective_seed_sha256": _digest({"requested_seed": spec.seed, "effective_seed": spec.seed}),
        "metric_sample_count": 510,
        "metric_trajectory_sha256": "a" * 64,
        "planned_impulse_sha256": (None if spec.impulse is None else _digest(list(spec.impulse))),
        "target_root_qvel_sha256": None if not disturbed else "b" * 64,
        "realized_impulse_sha256": None if not disturbed else "b" * 64,
        "impulse_qvel_write_exact": True if disturbed else None,
        "impulse_count": 1 if disturbed else 0,
        "impulse_apply_transition": 100 if disturbed else None,
        "impulse_apply_q9": 109 if disturbed else None,
        "impulse_readback_max_absolute_error": 0.0 if disturbed else None,
        "reward_metric_bound": True,
        "step_calls_started": 510,
        "transitions_completed": 510,
        "metric_sum": stats["sum"] / -25.0,
        "metric_minimum": stats["maximum"] / -25.0,
        "metric_maximum": stats["minimum"] / -25.0,
        "disturbance_performed": disturbed,
        "baseline_mean": None,
        "recovery_threshold": None,
        "first_threshold_crossing_transition": None,
        "stable_streak_start_transition": None,
        "stable_streak_end_transition": None,
        "recovery_time_s": None,
        "recovered": None,
    }
    if disturbed:
        result.update(
            {
                "baseline_mean": 0.2,
                "recovery_threshold": 0.3,
                "first_threshold_crossing_transition": 100,
                "stable_streak_start_transition": 100,
                "stable_streak_end_transition": 104,
                "recovery_time_s": 0.1,
                "recovered": True,
            }
        )
    return result


def _passing_record(spec: campaign.CampaignRunSpec) -> dict[str, Any]:
    full_report = _pass_report()
    return campaign._assess_run(  # noqa: SLF001
        spec=spec,
        full_report=full_report,
        probe=_probe_report(spec, full_report),
        contract=campaign.load_campaign_contract(ROOT),
    )


def _preflight_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    source = {
        "file_count": 3,
        "total_bytes": 123,
        "binding_sha256": "c" * 64,
        "files": [],
    }
    return (
        {
            "ready": True,
            "issues": [],
            "contract": campaign.load_campaign_contract(ROOT),
            "sources": source,
            "base_request": object(),
            "recovery_preflight": {},
        },
        source,
    )


def _step_evidence(metric: float) -> evaluation.StepEvidence:
    return evaluation.StepEvidence(
        reward=0.0,
        scalars={key: 0.0 for key in evaluation._SAFETY_SCALAR_KEYS},  # noqa: SLF001
        counts={key: 0 for key in evaluation._SAFETY_COUNT_KEYS},  # noqa: SLF001
        reward_rates={"evaluator_aligned_recovery": metric * -25.0},
    )


class _FakeRobot:
    def __init__(self) -> None:
        velocity = torch.tensor(
            [[0.25, -0.5, 0.75, -1.0, 1.25, -1.5]],
            dtype=torch.float32,
        )
        self.data = SimpleNamespace(
            root_link_vel_w=velocity,
            indexing=SimpleNamespace(
                free_joint_q_adr=torch.arange(7),
                free_joint_v_adr=torch.arange(6),
            ),
            data=SimpleNamespace(
                qpos=torch.tensor(
                    [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                ),
                qvel=velocity.clone(),
            ),
        )

    def write_root_link_velocity_to_sim(
        self,
        target: torch.Tensor,
        *,
        env_ids: torch.Tensor,
    ) -> None:
        assert torch.equal(env_ids, torch.tensor([0], dtype=torch.long))
        self.data.data.qvel[env_ids[:, None], self.data.indexing.free_joint_v_adr] = target


class _FakeWrapper:
    def __init__(self, seed: int) -> None:
        self.robot = _FakeRobot()
        self.command = SimpleNamespace(time_steps=torch.tensor([109], dtype=torch.long))
        self.cfg = SimpleNamespace(seed=seed)
        self.reward_manager = SimpleNamespace(
            get_term_cfg=lambda _name: SimpleNamespace(
                func=evaluator_aligned_recovery_metric,
                weight=-25.0,
            )
        )
        self.command_manager = SimpleNamespace(get_term=lambda _name: self.command)
        self.scene = {"robot": self.robot}
        self.unwrapped = self


def test_contract_and_exact_support_v1_cohort_are_frozen() -> None:
    contract = campaign.load_campaign_contract(ROOT)
    specs = campaign.campaign_run_specs(ROOT)

    assert len(specs) == 20
    assert [spec.scenario for spec in specs] == ["nominal", "disturbance"] * 10
    assert [spec.seed for spec in specs[0::2]] == contract["cohort"]["nominal_seeds"]
    assert [spec.seed for spec in specs[1::2]] == contract["cohort"]["disturbance_seeds"]
    assert len({spec.seed for spec in specs}) == 20
    assert all(spec.impulse is None for spec in specs[0::2])
    assert all(spec.impulse is not None for spec in specs[1::2])


def test_contract_rejects_handoff_conflated_disturbance_timing() -> None:
    contract = copy.deepcopy(campaign.load_campaign_contract(ROOT))
    contract["disturbance"]["teacher_local_apply_transition"] = 0
    contract["disturbance"]["global_apply_transition"] = 50
    contract["disturbance"]["apply_q9"] = 59

    with pytest.raises(ValueError, match="disturbance schedule"):
        campaign._validate_contract(contract)  # noqa: SLF001


def test_disturbance_vectors_are_exact_deterministic_and_bounded() -> None:
    contract = campaign.load_campaign_contract(ROOT)
    maxima = (0.5, 0.5, 0.2, 0.52, 0.52, 0.78)

    for spec in campaign.campaign_run_specs(ROOT)[1::2]:
        assert spec.impulse == tuple(contract["disturbance"]["exact_vectors_by_seed"][str(spec.seed)])
        assert all(abs(value) <= limit for value, limit in zip(spec.impulse, maxima))
        assert spec.descriptor()["planned_impulse_sha256"] == _digest(list(spec.impulse))


def test_prerequisites_bind_cutoff_pass_and_support_v1_seed_source() -> None:
    verified = campaign._verify_prerequisites(  # noqa: SLF001
        ROOT,
        campaign.load_campaign_contract(ROOT),
    )

    assert verified["cutoff"]["verdict"] == "mixed_controller_recovery_passed"
    assert set(verified["files"]) == {
        "recovery_contract",
        "cutoff50_pass_report",
        "support_contract",
        "candidate_manifest",
        "motion",
    }


def test_probe_adds_impulse_to_current_velocity_once_at_global100_q109() -> None:
    contract = campaign.load_campaign_contract(ROOT)
    spec = campaign.campaign_run_specs(ROOT)[1]
    probe = campaign._StepProbe(spec, contract)  # noqa: SLF001
    wrapper = _FakeWrapper(spec.seed)
    before = wrapper.robot.data.root_link_vel_w.clone()

    probe.on_construct(wrapper)
    probe.transitions_completed = 99
    probe.before_step(wrapper)
    assert probe.impulse_count == 0
    probe.transitions_completed = 100
    probe.before_step(wrapper)
    expected = before + torch.tensor(spec.impulse, dtype=torch.float32).reshape(1, 6)

    assert torch.equal(wrapper.robot.data.root_link_vel_w, before)
    assert torch.equal(wrapper.robot.data.data.qvel, expected)
    assert probe.impulse_count == 1
    assert probe.impulse_apply_transition == 100
    assert probe.impulse_apply_q9 == 109
    assert probe.readback_error <= 1.0e-6
    assert probe.qvel_write_exact is True
    assert probe.report()["target_root_qvel_sha256"] == probe.report()["realized_impulse_sha256"]
    probe.transitions_completed = 101
    probe.before_step(wrapper)
    assert probe.impulse_count == 1


def test_probe_counts_started_completed_and_metric_samples_separately() -> None:
    probe = campaign._StepProbe(  # noqa: SLF001
        campaign.campaign_run_specs(ROOT)[0],
        campaign.load_campaign_contract(ROOT),
    )

    probe.before_super_step()
    assert probe.report()["step_calls_started"] == 1
    assert probe.report()["transitions_completed"] == 0
    assert probe.report()["metric_sample_count"] == 0
    probe.after_step()
    assert probe.report()["transitions_completed"] == 1
    assert probe.report()["metric_sample_count"] == 0
    probe.capture_evidence(_step_evidence(0.25))
    assert probe.report()["metric_sample_count"] == 1


def test_probe_recovery_uses_stable_streak_completion_time() -> None:
    probe = campaign._StepProbe(  # noqa: SLF001
        campaign.campaign_run_specs(ROOT)[1],
        campaign.load_campaign_contract(ROOT),
    )
    metrics = [2.0] * 510
    metrics[90:100] = [1.0] * 10
    metrics[105:110] = [1.0] * 5
    for metric in metrics:
        probe.capture_evidence(_step_evidence(metric))

    report = probe.report()
    assert report["baseline_mean"] == 1.0
    assert report["recovery_threshold"] == 1.1
    assert report["first_threshold_crossing_transition"] == 105
    assert report["stable_streak_start_transition"] == 105
    assert report["stable_streak_end_transition"] == 109
    assert report["recovery_time_s"] == pytest.approx(0.2)
    assert report["recovered"] is True


def test_probe_rejects_stable_completion_after_two_seconds() -> None:
    probe = campaign._StepProbe(  # noqa: SLF001
        campaign.campaign_run_specs(ROOT)[1],
        campaign.load_campaign_contract(ROOT),
    )
    metrics = [2.0] * 510
    metrics[90:100] = [1.0] * 10
    metrics[197:202] = [1.0] * 5
    for metric in metrics:
        probe.capture_evidence(_step_evidence(metric))

    report = probe.report()
    assert report["stable_streak_end_transition"] == 201
    assert report["recovery_time_s"] == pytest.approx(2.04)
    assert report["recovered"] is False


@pytest.mark.parametrize(
    ("scenario_offset", "mutator", "expected_issue"),
    [
        (
            0,
            lambda report, _probe: report["rollout"].__setitem__("hard_safety_violation_count", 1),
            "hard safety count",
        ),
        (
            0,
            lambda report, _probe: report["rollout"].__setitem__("soft_safety_warning_count", 1),
            "soft safety count",
        ),
        (
            0,
            lambda report, _probe: report["support_summary"].__setitem__("teacher_parity_violation_count", 1),
            "support zero count:teacher_parity_violation_count",
        ),
        (
            0,
            lambda report, _probe: report["action_semantics"].__setitem__("mismatch_count", 1),
            "actual action chain",
        ),
        (
            1,
            lambda _report, probe: probe.__setitem__("impulse_apply_q9", 59),
            "disturbance impulse q9",
        ),
        (
            1,
            lambda _report, probe: probe.__setitem__("recovery_time_s", 0.08),
            "disturbance recovery time",
        ),
    ],
)
def test_assessor_recomputes_raw_gates_fail_closed(
    scenario_offset: int,
    mutator: Any,
    expected_issue: str,
) -> None:
    spec = campaign.campaign_run_specs(ROOT)[scenario_offset]
    report = _pass_report()
    probe = _probe_report(spec, report)
    mutator(report, probe)

    record = campaign._assess_run(  # noqa: SLF001
        spec=spec,
        full_report=report,
        probe=probe,
        contract=campaign.load_campaign_contract(ROOT),
    )

    assert record["passed"] is False
    assert expected_issue in record["first_issue"] or record["issue_count"] > 1


def test_assessor_accepts_exact_nominal_and_disturbance_evidence() -> None:
    nominal, disturbed = campaign.campaign_run_specs(ROOT)[:2]

    nominal_record = _passing_record(nominal)
    disturbance_record = _passing_record(disturbed)

    assert nominal_record["passed"] is True
    assert nominal_record["disturbance_performed"] is False
    assert nominal_record["recovered"] is None
    assert disturbance_record["passed"] is True
    assert disturbance_record["impulse_apply_transition"] == 100
    assert disturbance_record["impulse_apply_q9"] == 109
    assert disturbance_record["recovered"] is True


def test_run_campaign_preflights_once_and_qualifies_twenty_mocked_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preflight, source = _preflight_fixture()
    calls = {"preflight": 0, "execute": 0}
    progress: list[dict[str, Any]] = []

    def fake_preflight(_request: campaign.CampaignRequest) -> dict[str, Any]:
        calls["preflight"] += 1
        return preflight

    def fake_execute(**kwargs: Any) -> dict[str, Any]:
        calls["execute"] += 1
        return _passing_record(kwargs["spec"])

    monkeypatch.setattr(campaign, "_preflight_internal", fake_preflight)
    monkeypatch.setattr(campaign, "_execute_one", fake_execute)
    monkeypatch.setattr(
        campaign,
        "executed_campaign_source_binding",
        lambda _root: source,
    )
    request = campaign.CampaignRequest(ROOT, ROOT / "artifacts/unused.json")

    report = campaign.run_campaign(request, progress=lambda value: progress.append(dict(value)))

    assert calls == {"preflight": 1, "execute": 20}
    assert len(progress) == 20
    assert report["campaign_qualified"] is True
    assert report["completed_run_count"] == 20
    assert report["nominal_passed_count"] == 10
    assert report["disturbance_recovered_count"] == 10
    assert report["published_teacher_label_count"] == 0
    assert report["published_training_row_count"] == 0
    assert report["fresh_disjoint_suffix_collection_authorized"] is True


def test_run_campaign_fail_fast_keeps_truthful_partial_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight, source = _preflight_fixture()
    specs = campaign.campaign_run_specs(ROOT)
    calls = 0

    def fake_execute(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _passing_record(kwargs["spec"])
        probe = {
            "step_calls_started": 17,
            "transitions_completed": 16,
            "metric_sample_count": 15,
            "effective_seed": kwargs["spec"].seed,
            "effective_seed_sha256": _digest(
                {
                    "requested_seed": kwargs["spec"].seed,
                    "effective_seed": kwargs["spec"].seed,
                }
            ),
            "planned_impulse_sha256": specs[1].descriptor()["planned_impulse_sha256"],
            "target_root_qvel_sha256": "d" * 64,
            "realized_impulse_sha256": "d" * 64,
            "impulse_qvel_write_exact": True,
            "impulse_count": 1,
            "impulse_apply_transition": 100,
            "impulse_apply_q9": 109,
            "impulse_readback_max_absolute_error": 0.0,
            "metric_trajectory_sha256": "e" * 64,
        }
        return campaign._failed_run_record(  # noqa: SLF001
            kwargs["spec"], RuntimeError("synthetic"), probe=probe
        )

    monkeypatch.setattr(campaign, "_preflight_internal", lambda _request: preflight)
    monkeypatch.setattr(campaign, "_execute_one", fake_execute)
    monkeypatch.setattr(
        campaign,
        "executed_campaign_source_binding",
        lambda _root: source,
    )

    report = campaign.run_campaign(campaign.CampaignRequest(ROOT, ROOT / "artifacts/unused.json"))

    assert calls == 2
    assert report["campaign_qualified"] is False
    assert report["completed_run_count"] == 2
    assert report["passed_run_count"] == 1
    assert report["remaining_run_count"] == 18
    assert report["first_failed_run_id"] == specs[1].run_id
    failed = report["runs"][1]
    assert failed["step_calls_started"] == 17
    assert failed["transitions_completed"] == 16
    assert failed["attempted_transitions"] == 16
    assert failed["metric_sample_count"] == 15


def test_failed_run_publishes_bounded_error_type_and_message_digest() -> None:
    spec = campaign.campaign_run_specs(ROOT)[0]
    secret_array_text = "tensor([[1.0, 2.0, 3.0]])"

    record = campaign._failed_run_record(  # noqa: SLF001
        spec,
        RuntimeError(secret_array_text),
    )

    assert record["first_issue"] == "runtime:RuntimeError"
    assert secret_array_text not in json.dumps(record)
    assert len(record["failure_detail_sha256"]) == 64


def test_post_run_source_rehash_failure_keeps_completed_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight, _source = _preflight_fixture()
    monkeypatch.setattr(campaign, "_preflight_internal", lambda _request: preflight)
    monkeypatch.setattr(
        campaign,
        "_execute_one",
        lambda **kwargs: _passing_record(kwargs["spec"]),
    )
    monkeypatch.setattr(
        campaign,
        "executed_campaign_source_binding",
        lambda _root: (_ for _ in ()).throw(OSError("rehash failed")),
    )

    report = campaign.run_campaign(campaign.CampaignRequest(ROOT, ROOT / "artifacts/unused.json"))

    assert report["completed_run_count"] == 20
    assert report["passed_run_count"] == 20
    assert report["campaign_qualified"] is False
    assert report["source_rehash_succeeded"] is False
    assert report["sources_unchanged"] is False
    assert report["source_rehash_error_sha256"] is not None
    assert report["first_failure"].startswith("post_run_source_rehash:")


def test_post_run_source_digest_change_keeps_completed_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight, _source = _preflight_fixture()
    changed_source = {
        "file_count": 3,
        "total_bytes": 124,
        "binding_sha256": "d" * 64,
        "files": [],
    }
    monkeypatch.setattr(campaign, "_preflight_internal", lambda _request: preflight)
    monkeypatch.setattr(
        campaign,
        "_execute_one",
        lambda **kwargs: _passing_record(kwargs["spec"]),
    )
    monkeypatch.setattr(
        campaign,
        "executed_campaign_source_binding",
        lambda _root: changed_source,
    )

    report = campaign.run_campaign(campaign.CampaignRequest(ROOT, ROOT / "artifacts/unused.json"))

    assert report["completed_run_count"] == 20
    assert report["passed_run_count"] == 20
    assert report["campaign_qualified"] is False
    assert report["source_rehash_succeeded"] is True
    assert report["sources_unchanged"] is False
    assert report["first_failure"] == "post_run_sources_changed"
    assert report["failure_detail_sha256"] is not None


def test_writer_rejects_forged_qualified_empty_campaign(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specs = campaign.campaign_run_specs(ROOT)
    preflight, source = _preflight_fixture()
    failed = campaign._campaign_report(  # noqa: SLF001
        preflight=preflight,
        specs=specs,
        records=[],
        sources_after=source,
        runtime_failure="synthetic pre-rollout failure",
    )
    forged = dict(failed)
    forged.update(
        {
            "campaign_qualified": True,
            "verdict": "qualification_campaign_passed",
            "completed_run_count": 20,
            "passed_run_count": 20,
            "nominal_passed_count": 10,
            "disturbance_passed_count": 10,
            "disturbance_recovered_count": 10,
            "remaining_run_count": 0,
            "first_failure": None,
            "fresh_disjoint_suffix_collection_authorized": True,
        }
    )
    request = campaign.CampaignRequest(tmp_path, tmp_path / "forged.json")
    monkeypatch.setattr(campaign, "campaign_run_specs", lambda _root: specs)

    with pytest.raises(ValueError, match="counter mismatch|not derived"):
        campaign.write_campaign_report_new(request, forged)
    assert not request.output_path.exists()


def test_writer_is_exclusive_and_report_contains_no_training_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specs = campaign.campaign_run_specs(ROOT)
    preflight, source = _preflight_fixture()
    report = campaign._campaign_report(  # noqa: SLF001
        preflight=preflight,
        specs=specs,
        records=[],
        sources_after=source,
        runtime_failure="synthetic pre-rollout failure",
    )
    request = campaign.CampaignRequest(tmp_path, tmp_path / "campaign.json")
    monkeypatch.setattr(campaign, "campaign_run_specs", lambda _root: specs)

    output = campaign.write_campaign_report_new(request, report)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded["runs"] == []
    assert loaded["published_teacher_label_count"] == 0
    assert loaded["published_training_row_count"] == 0
    assert "actions" not in loaded
    assert "observations" not in loaded
    with pytest.raises(FileExistsError):
        campaign.write_campaign_report_new(request, report)
