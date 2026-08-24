from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from gear_sonic.scripts import train_g1_true23_native124_supported_idle as launcher

_REPO_ROOT = launcher.REPO_ROOT
_QUALIFICATION_CHECKPOINT = launcher.DEFAULT_SOURCE_RUN / "model_14000.pt"
_QUALIFICATION_ACTOR = (
    _REPO_ROOT / "artifacts/g1_true23_step1c_checkpoint_screen/actors/model_14000.stable_v2.native124.onnx"
)
_QUALIFICATION_EXPORT = _QUALIFICATION_ACTOR.with_suffix(".export.json")
_DIAGNOSTIC_TEMPLATE = (
    _REPO_ROOT
    / "artifacts/g1_true23_step1c_native_policy_model_11500"
    / "exact_model_zero_velocity_diagnostic_v1.json"
)


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return launcher.sha256_file(path)


def _authorization(plan: dict, job_id: str) -> dict:
    return {
        "schema_version": 1,
        "kind": launcher.AUTHORIZATION_KIND,
        "authorized": True,
        "plan_payload_sha256": plan["payload_sha256"],
        "job_id": job_id,
        "corpus_sha256": launcher.EXPECTED_CORPUS_SHA256,
        "simulator_only": True,
        "hardware_commands_authorized": False,
        "corpus_diagnostic_flag_acknowledged": True,
    }


def _gate(
    checkpoint_sha256: str,
    actor_onnx: dict,
    export_report: dict,
    diagnostic_report: dict,
    phase: str = "static",
) -> dict:
    return {
        "schema_version": 1,
        "kind": launcher.GATE_KIND,
        "qualified_phase": phase,
        "checkpoint_sha256": checkpoint_sha256,
        "contact_evidence_scope": launcher.CONTACT_EVIDENCE_SCOPE,
        "corpus_sha256": launcher.EXPECTED_CORPUS_SHA256,
        "evidence_class": launcher.DIAGNOSTIC_EVIDENCE_CLASS,
        "horizon_steps": 500,
        "clip_ids": list(launcher.QUALIFICATION_CLIPS),
        "passed": True,
        "no_terminations": True,
        "no_hard_limit_violations": True,
        "actor_onnx": actor_onnx,
        "export_report": export_report,
        "diagnostic_report": diagnostic_report,
    }


def _write_qualification_evidence(
    root: Path,
    *,
    termination: object | None = None,
) -> tuple[Path, str, dict, dict, dict]:
    checkpoint_sha256 = launcher.sha256_file(_QUALIFICATION_CHECKPOINT)
    actor_path = root / "actor.onnx"
    shutil.copyfile(_QUALIFICATION_ACTOR, actor_path)
    policy_sha256 = launcher.sha256_file(actor_path)
    export_path = root / "export.json"
    export = json.loads(_QUALIFICATION_EXPORT.read_text(encoding="utf-8"))
    export["checkpoint_lineage"]["checkpoint"]["sha256"] = checkpoint_sha256
    _write_json(export_path, export)
    diagnostic_path = root / "diagnostic.json"
    diagnostic = json.loads(_DIAGNOSTIC_TEMPLATE.read_text(encoding="utf-8"))
    diagnostic["evidence_class"] = launcher.DIAGNOSTIC_EVIDENCE_CLASS
    diagnostic["policy"]["sha256"] = policy_sha256
    diagnostic["policy"]["path"] = str(actor_path)
    for run in diagnostic["runs"]:
        run["contact_steps"] = {"left_foot": 500, "right_foot": 500}
        run["first_shipped_termination"] = termination if run["mode"] == "static_frame0_zero_velocity" else None
        run["joint_hard_limit_violation_steps"] = 0
        run["horizon_steps"] = 500
        run["steps_executed"] = 500
    _write_json(diagnostic_path, diagnostic)
    return (
        _QUALIFICATION_CHECKPOINT,
        checkpoint_sha256,
        launcher.file_binding(actor_path),
        launcher.file_binding(export_path),
        launcher.file_binding(diagnostic_path),
    )


def test_default_plan_is_read_only_static_ab_and_never_imports_mjlab(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mjlab" or name.startswith("mjlab."):
            raise AssertionError("planning imported MJLab")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert launcher.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert launcher.validate_plan(plan) == plan
    assert [job["job_id"] for job in plan["payload"]["jobs"]] == [
        "static_model3500",
        "static_model11500",
    ]
    assert [job["planned_updates"] for job in plan["payload"]["jobs"]] == [2000, 2000]
    assert plan["payload"]["corpus_flags"] == {
        "diagnostic_only": True,
        "training_authorized": False,
    }
    assert not plan["payload"]["safety"]["domain_randomization_events"]
    assert not plan["payload"]["safety"]["episode_length_randomization"]
    assert plan["payload"]["execution"]["available"] is False
    assert plan["payload"]["execution"]["blocker"] == launcher.FUTURE_EXECUTION_BLOCKER
    assert plan["payload"]["qualification"]["evidence_class"] == ("self_attested_curriculum_diagnostic_only")
    assert plan["payload"]["qualification"]["contact_evidence_scope"] == (
        "aggregate_curriculum_only_not_final_step1b_stance_qualification"
    )
    assert plan["payload"]["qualification"]["unlocks_later_phases"] is False
    assert plan["payload"]["pinned_package_versions"] == launcher.PINNED_PACKAGE_VERSIONS
    assert plan["payload"]["pinned_package_versions"]["onnx"] == "1.22.0"
    assert plan["payload"]["pinned_package_versions"]["onnxruntime"] == "1.23.2"
    runtime_paths = {binding["path"] for binding in plan["payload"]["runtime_files"]}
    assert {
        "gear_sonic/trl/mjlab/supported_idle_checkpoint.py",
        "gear_sonic/trl/mjlab/supported_idle_runner.py",
        "gear_sonic/utils/g1_23dof_native124_actor_export.py",
        "gear_sonic/utils/g1_true23_step1c_onnx_diagnostic.py",
        "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        "external_dependencies/unitree_rl_mjlab/scripts/train.py",
    } <= runtime_paths


def test_static_plan_pins_exact_seed_hashes_and_iterations() -> None:
    plan = launcher.build_plan()
    jobs = {job["job_id"]: job for job in plan["payload"]["jobs"]}
    assert jobs["static_model3500"]["source_checkpoint"]["sha256"] == (
        "41488ab06ee876fb6ddf77ac18d22231e67f06f9c09c45531b21ed2abd709596"
    )
    assert jobs["static_model3500"]["source_checkpoint_expected_iter"] == 3500
    assert jobs["static_model11500"]["source_checkpoint"]["sha256"] == (
        "1302ed2d7128c5f129611c29a34181d1ac7e27d2c15f551e49453e41ee81ec4a"
    )
    assert jobs["static_model11500"]["source_checkpoint_expected_iter"] == 11500


def test_plan_hash_and_safety_tampering_fail_closed() -> None:
    plan = launcher.build_plan()
    plan["payload"]["safety"]["domain_randomization_events"] = True
    with pytest.raises(ValueError, match="payload hash"):
        launcher.validate_plan(plan)
    plan["payload_sha256"] = launcher.payload_sha256(plan["payload"])
    with pytest.raises(ValueError, match="safety contract"):
        launcher.validate_plan(plan)


def test_plan_schema_rejects_boolean_version() -> None:
    plan = launcher.build_plan()
    plan["schema_version"] = True
    with pytest.raises(ValueError, match="not bool"):
        launcher.validate_plan(plan)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload["optimizer"].update(learning_rate=0.001), "optimizer"),
        (lambda payload: payload.update(num_envs=256), "num_envs"),
        (lambda payload: payload["jobs"][0].update(planned_updates=9999), "job 'planned_updates'"),
        (lambda payload: payload["jobs"][0].update(job_id="renamed"), "job IDs/order"),
        (lambda payload: payload.update(unexpected=True), "payload keys"),
        (lambda payload: payload["runtime_files"].pop(), "runtime file"),
    ],
)
def test_exact_plan_contract_rejects_rehashed_semantic_mutation(
    mutation: object,
    match: str,
) -> None:
    plan = launcher.build_plan()
    mutation(plan["payload"])
    plan["payload_sha256"] = launcher.payload_sha256(plan["payload"])
    with pytest.raises(ValueError, match=match):
        launcher.validate_plan(plan)


def test_num_envs_is_fixed() -> None:
    with pytest.raises(ValueError, match="fixed at 512"):
        launcher.build_plan(num_envs=256)


def test_external_authorization_requires_exact_file_hash_plan_and_job(tmp_path: Path) -> None:
    plan = launcher.build_plan()
    path = tmp_path / "authorization.json"
    authorization = _authorization(plan, "static_model3500")
    expected_hash = _write_json(path, authorization)
    assert (
        launcher.validate_authorization(
            path,
            expected_file_sha256=expected_hash,
            plan=plan,
            job_id="static_model3500",
        )
        == authorization
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        launcher.validate_authorization(
            path,
            expected_file_sha256="0" * 64,
            plan=plan,
            job_id="static_model3500",
        )
    authorization["authorized"] = False
    changed_hash = _write_json(path, authorization)
    with pytest.raises(ValueError, match="authorized"):
        launcher.validate_authorization(
            path,
            expected_file_sha256=changed_hash,
            plan=plan,
            job_id="static_model3500",
        )


def test_qualification_gate_requires_bound_reports_and_two_clean_clips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    checkpoint_path, checkpoint_sha256, actor_binding, export_binding, diagnostic_binding = (
        _write_qualification_evidence(tmp_path)
    )
    path = tmp_path / "gate.json"
    value = _gate(checkpoint_sha256, actor_binding, export_binding, diagnostic_binding)
    _write_json(path, value)
    validated, binding = launcher._validate_gate(
        path,
        parent_checkpoint_path=checkpoint_path,
        parent_sha256=checkpoint_sha256,
        required_phase="static",
    )
    assert validated == value
    assert binding == launcher.file_binding(path)
    value["clip_ids"] = [launcher.CHANGE_IDLE]
    _write_json(path, value)
    with pytest.raises(ValueError, match="clip_ids"):
        launcher._validate_gate(
            path,
            parent_checkpoint_path=checkpoint_path,
            parent_sha256=checkpoint_sha256,
            required_phase="static",
        )
    value = _gate(checkpoint_sha256, actor_binding, export_binding, diagnostic_binding)
    value["no_terminations"] = False
    _write_json(path, value)
    with pytest.raises(ValueError, match="no_terminations"):
        launcher._validate_gate(
            path,
            parent_checkpoint_path=checkpoint_path,
            parent_sha256=checkpoint_sha256,
            required_phase="static",
        )
    value = _gate(checkpoint_sha256, actor_binding, export_binding, diagnostic_binding)
    _write_json(path, value)
    with (tmp_path / "diagnostic.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="SHA256"):
        launcher._validate_gate(
            path,
            parent_checkpoint_path=checkpoint_path,
            parent_sha256=checkpoint_sha256,
            required_phase="static",
        )


def test_qualification_gate_rejects_terminated_bound_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    checkpoint_path, checkpoint_sha256, actor_binding, export_binding, diagnostic_binding = (
        _write_qualification_evidence(
            tmp_path,
            termination={"step": 499, "terms": ["anchor_pos"]},
        )
    )
    path = tmp_path / "gate.json"
    _write_json(
        path,
        _gate(checkpoint_sha256, actor_binding, export_binding, diagnostic_binding),
    )
    with pytest.raises(ValueError, match="shipped termination"):
        launcher._validate_gate(
            path,
            parent_checkpoint_path=checkpoint_path,
            parent_sha256=checkpoint_sha256,
            required_phase="static",
        )


def test_qualification_gate_rejects_one_missing_stance_contact_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    checkpoint_path, checkpoint_sha256, actor_binding, export_binding, _ = _write_qualification_evidence(tmp_path)
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    selected = [run for run in diagnostic["runs"] if run["mode"] == "static_frame0_zero_velocity"]
    selected[0]["contact_steps"]["left_foot"] = 499
    _write_json(diagnostic_path, diagnostic)
    gate_path = tmp_path / "gate.json"
    _write_json(
        gate_path,
        _gate(
            checkpoint_sha256,
            actor_binding,
            export_binding,
            launcher.file_binding(diagnostic_path),
        ),
    )
    with pytest.raises(ValueError, match="stance contact mismatch"):
        launcher._validate_gate(
            gate_path,
            parent_checkpoint_path=checkpoint_path,
            parent_sha256=checkpoint_sha256,
            required_phase="static",
        )


def test_qualification_gate_rejects_missing_actor_onnx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    checkpoint_path, checkpoint_sha256, actor_binding, export_binding, diagnostic_binding = (
        _write_qualification_evidence(tmp_path)
    )
    (tmp_path / actor_binding["path"]).unlink()
    gate_path = tmp_path / "gate.json"
    _write_json(
        gate_path,
        _gate(checkpoint_sha256, actor_binding, export_binding, diagnostic_binding),
    )
    with pytest.raises(ValueError, match="bound file is missing"):
        launcher._validate_gate(
            gate_path,
            parent_checkpoint_path=checkpoint_path,
            parent_sha256=checkpoint_sha256,
            required_phase="static",
        )


def test_qualification_gate_rejects_self_reported_mutated_actor_onnx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onnx

    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    checkpoint_path, checkpoint_sha256, actor_binding, _, _ = _write_qualification_evidence(tmp_path)
    actor_path = tmp_path / actor_binding["path"]
    model = onnx.load(actor_path)
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 22
    onnx.save(model, actor_path)
    mutated_sha256 = launcher.sha256_file(actor_path)
    actor_binding = launcher.file_binding(actor_path)

    export_path = tmp_path / "export.json"
    export = json.loads(export_path.read_text(encoding="utf-8"))
    export["export"]["output_sha256"] = mutated_sha256
    export["parity"]["exported_onnx_vs_checkpoint"]["onnx"]["sha256"] = mutated_sha256
    _write_json(export_path, export)
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    diagnostic["policy"]["sha256"] = mutated_sha256
    _write_json(diagnostic_path, diagnostic)

    gate_path = tmp_path / "gate.json"
    _write_json(
        gate_path,
        _gate(
            checkpoint_sha256,
            actor_binding,
            launcher.file_binding(export_path),
            launcher.file_binding(diagnostic_path),
        ),
    )
    with pytest.raises(ValueError, match="gate actor ONNX"):
        launcher._validate_gate(
            gate_path,
            parent_checkpoint_path=checkpoint_path,
            parent_sha256=checkpoint_sha256,
            required_phase="static",
        )


def test_later_phase_plan_is_blocked_without_verified_replay() -> None:
    with pytest.raises(ValueError, match=launcher.FUTURE_EXECUTION_BLOCKER):
        launcher.build_plan(phase_name="trajectory_start")
    with pytest.raises(ValueError, match="does not accept parent"):
        launcher.build_plan(
            phase_name="static",
            parent_checkpoint=launcher.SEED_CHECKPOINTS["model3500"][0],
        )


def test_self_attested_rewritten_report_cannot_unlock_later_build_or_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static_plan = launcher.build_plan()
    crafted_later_plan = json.loads(json.dumps(static_plan))
    crafted_later_plan["payload"]["phase_name"] = "trajectory_start"
    crafted_later_plan["payload_sha256"] = launcher.payload_sha256(crafted_later_plan["payload"])

    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    checkpoint_path, checkpoint_sha256, actor_binding, export_binding, diagnostic_binding = (
        _write_qualification_evidence(tmp_path)
    )
    gate_path = tmp_path / "rewritten-self-attested-gate.json"
    gate = _gate(checkpoint_sha256, actor_binding, export_binding, diagnostic_binding)
    _write_json(gate_path, gate)
    validated_gate, _ = launcher._validate_gate(
        gate_path,
        parent_checkpoint_path=checkpoint_path,
        parent_sha256=checkpoint_sha256,
        required_phase="static",
    )
    assert validated_gate["evidence_class"] == launcher.DIAGNOSTIC_EVIDENCE_CLASS

    with pytest.raises(ValueError, match=launcher.FUTURE_EXECUTION_BLOCKER):
        launcher.build_plan(
            phase_name="trajectory_start",
            parent_checkpoint=checkpoint_path,
            qualification_gate=gate_path,
        )
    with pytest.raises(ValueError, match=launcher.FUTURE_EXECUTION_BLOCKER):
        launcher.validate_plan(crafted_later_plan)

    plan_path = tmp_path / "crafted-later-plan.json"
    plan_hash = _write_json(plan_path, crafted_later_plan)
    auth_path = tmp_path / "crafted-authorization.json"
    auth_hash = _write_json(auth_path, _authorization(crafted_later_plan, "static_model3500"))
    with pytest.raises(ValueError, match=launcher.FUTURE_EXECUTION_BLOCKER):
        launcher.preflight_run(
            plan_path=plan_path,
            plan_file_sha256=plan_hash,
            authorization_path=auth_path,
            authorization_file_sha256=auth_hash,
            job_id="static_model3500",
        )


def test_safe_seed_load_uses_exact_schema_hash_and_iter() -> None:
    path, expected_hash, expected_iter = launcher.SEED_CHECKPOINTS["model3500"]
    checkpoint = launcher.safe_load_checkpoint(
        path,
        expected_sha256=expected_hash,
        expected_iter=expected_iter,
    )
    assert checkpoint["iter"] == 3500
    assert len(checkpoint["actor_state_dict"]) == 13
    assert len(checkpoint["critic_state_dict"]) == 12
    with pytest.raises(ValueError, match="bytes differ"):
        launcher.safe_load_checkpoint(
            path,
            expected_sha256="0" * 64,
            expected_iter=expected_iter,
        )
    with pytest.raises(ValueError, match="iter differs"):
        launcher.safe_load_checkpoint(
            path,
            expected_sha256=expected_hash,
            expected_iter=expected_iter + 1,
        )


def test_safe_seed_load_rejects_exact_key_shape_and_dtype_drift(tmp_path: Path) -> None:
    import torch

    source, _, _ = launcher.SEED_CHECKPOINTS["model3500"]
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    checkpoint["actor_state_dict"]["mlp.6.bias"] = torch.zeros(22, dtype=torch.float32)
    changed = tmp_path / "wrong_shape.pt"
    torch.save(checkpoint, changed)
    with pytest.raises(ValueError, match="shape/dtype"):
        launcher.safe_load_checkpoint(
            changed,
            expected_sha256=launcher.sha256_file(changed),
            expected_iter=3500,
        )
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    checkpoint["critic_state_dict"]["mlp.6.bias"] = torch.zeros(1, dtype=torch.float64)
    changed = tmp_path / "wrong_dtype.pt"
    torch.save(checkpoint, changed)
    with pytest.raises(ValueError, match="shape/dtype"):
        launcher.safe_load_checkpoint(
            changed,
            expected_sha256=launcher.sha256_file(changed),
            expected_iter=3500,
        )


def test_resume_counter_advances_past_saved_iteration() -> None:
    checkpoint = {
        "iter": 249,
        "infos": {
            "supported_idle": {
                "kind": launcher.CHECKPOINT_INFO_KIND,
                "schema_version": 1,
                "completed_updates": 250,
                "plan_payload_sha256": "a" * 64,
                "job_id": "static_model3500",
                "corpus_sha256": launcher.EXPECTED_CORPUS_SHA256,
                "source_checkpoint_sha256": launcher.SEED_CHECKPOINTS["model3500"][1],
            }
        },
    }
    assert launcher.completed_updates_from_checkpoint(checkpoint, resume=False) == 0
    assert launcher.completed_updates_from_checkpoint(checkpoint, resume=True) == 250
    checkpoint["iter"] = 250
    with pytest.raises(ValueError, match="off-by-one"):
        launcher.completed_updates_from_checkpoint(checkpoint, resume=True)
    checkpoint["iter"] = 249
    checkpoint["infos"]["supported_idle"]["schema_version"] = True
    with pytest.raises(ValueError, match="not bool"):
        launcher.completed_updates_from_checkpoint(checkpoint, resume=True)
    checkpoint["infos"]["supported_idle"]["schema_version"] = 1
    checkpoint["infos"]["supported_idle"]["unexpected"] = True
    with pytest.raises(ValueError, match="keys mismatch"):
        launcher.completed_updates_from_checkpoint(checkpoint, resume=True)


def test_unperturbed_contract_clears_every_randomization_surface() -> None:
    motion = SimpleNamespace(
        pose_range={"x": (-1.0, 1.0)},
        velocity_range={"x": (-1.0, 1.0)},
        joint_position_range=(-0.1, 0.1),
        debug_vis=True,
    )
    cfg = SimpleNamespace(
        seed=None,
        scene=SimpleNamespace(num_envs=1),
        observations={"actor": SimpleNamespace(enable_corruption=True)},
        commands={"motion": motion},
        events={"push_robot": object(), "foot_friction": object()},
    )
    assert launcher.apply_unperturbed_env_contract(cfg, num_envs=512, seed=20260806) is cfg
    assert cfg.seed == 20260806
    assert cfg.scene.num_envs == 512
    assert not cfg.observations["actor"].enable_corruption
    assert cfg.events == {}
    assert motion.pose_range == {}
    assert motion.velocity_range == {}
    assert motion.joint_position_range == (0.0, 0.0)
    assert not motion.debug_vis


def test_preflight_validates_plan_authorization_and_seed_without_mjlab(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = launcher.build_plan()
    plan_path = tmp_path / "plan.json"
    plan_file_sha256 = _write_json(plan_path, plan)
    auth_path = tmp_path / "authorization.json"
    auth_file_sha256 = _write_json(auth_path, _authorization(plan, "static_model3500"))

    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mjlab" or name.startswith("mjlab."):
            raise AssertionError("preflight imported MJLab")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    loaded_plan, job, authorization = launcher.preflight_run(
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha256,
        authorization_path=auth_path,
        authorization_file_sha256=auth_file_sha256,
        job_id="static_model3500",
    )
    assert loaded_plan == plan
    assert job["job_id"] == "static_model3500"
    assert authorization["authorized"] is True


def test_no_run_subcommand_exists_and_cli_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = launcher.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    with pytest.raises(SystemExit):
        launcher.main(["run"])
    assert set(tmp_path.iterdir()) == before


def test_payload_hash_is_canonical() -> None:
    left = {"b": 2, "a": [1, 3]}
    right = {"a": [1, 3], "b": 2}
    expected = hashlib.sha256(b'{"a":[1,3],"b":2}').hexdigest()
    assert launcher.payload_sha256(left) == expected
    assert launcher.payload_sha256(right) == expected


def test_json_rejects_duplicate_keys_and_non_finite_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        launcher.read_json(duplicate)
    non_finite = tmp_path / "nan.json"
    non_finite.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        launcher.read_json(non_finite)
