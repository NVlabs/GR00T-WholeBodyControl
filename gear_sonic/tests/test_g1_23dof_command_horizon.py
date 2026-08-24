"""Dependency-light tests for true23 command-horizon playback semantics.

Isaac Lab is unavailable in the unit-test environment, so these tests execute
the production method AST directly with small tensor-backed harnesses.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMANDS_PATH = REPO_ROOT / "gear_sonic/envs/manager_env/mdp/commands.py"
TERMINATIONS_PATH = REPO_ROOT / "gear_sonic/envs/manager_env/mdp/terminations.py"


def _production_command_harness() -> type:
    tree = ast.parse(COMMANDS_PATH.read_text(encoding="utf-8"))
    production_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TrackingCommand"
    )
    names = {
        "_enforce_complete_future_horizon",
        "_update_command",
        "tracking_time_out_steps",
        "future_time_steps",
    }
    methods = [
        node
        for node in production_class.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {method.name for method in methods} == names
    harness = ast.ClassDef(
        name="TrackingCommandHarness",
        bases=[],
        keywords=[],
        decorator_list=[],
        body=methods,
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            harness,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"Sequence": Sequence, "torch": torch}
    exec(compile(module, COMMANDS_PATH, "exec"), namespace)  # noqa: S102
    return namespace["TrackingCommandHarness"]


def _production_tracking_time_out():
    tree = ast.parse(TERMINATIONS_PATH.read_text(encoding="utf-8"))
    production_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "tracking_time_out"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            production_function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, TERMINATIONS_PATH, "exec"), namespace)  # noqa: S102
    return namespace["tracking_time_out"]


def _command(*, true23: bool = True):
    command_type = _production_command_harness()
    command = command_type()
    command._require_complete_future_horizon = true23
    command.device = torch.device("cpu")
    return command


def test_true23_future_reference_requires_real_velocity_proof_frame() -> None:
    command = _command()
    command.future_time_steps_init = torch.tensor([[0, 5, 10]])
    command.time_steps = torch.tensor([0])
    command.motion_start_time_steps = torch.tensor([0])
    command.motion_num_steps = torch.tensor([12])

    assert command.future_time_steps.tolist() == [0, 5, 10]

    command.motion_num_steps = torch.tensor([11])
    with pytest.raises(RuntimeError, match="terminal clamping is forbidden"):
        _ = command.future_time_steps


def test_legacy_future_reference_keeps_terminal_clamp() -> None:
    command = _command(true23=False)
    command.future_time_steps_init = torch.tensor([[0, 5, 10]])
    command.time_steps = torch.tensor([0])
    command.motion_start_time_steps = torch.tensor([0])
    command.motion_num_steps = torch.tensor([8])

    assert command.future_time_steps.tolist() == [0, 5, 7]


def test_true23_reset_start_reserves_horizon_proof_and_command_advance() -> None:
    command = _command()
    command._future_horizon_tail_steps = 11
    command.time_steps = torch.tensor([0, 0])
    command.motion_num_steps = torch.tensor([20, 20])
    command.motion_start_time_steps = torch.tensor([19, 7])

    command._enforce_complete_future_horizon(torch.tensor([0, 1]))

    assert 0 <= command.motion_start_time_steps[0].item() <= 7
    assert command.motion_start_time_steps[1].item() == 7


def test_true23_reset_rejects_clip_too_short_for_post_reset_advance() -> None:
    command = _command()
    command._future_horizon_tail_steps = 11
    command.time_steps = torch.tensor([0])
    command.motion_num_steps = torch.tensor([12])
    command.motion_start_time_steps = torch.tensor([0])

    with pytest.raises(RuntimeError, match="shortest=12, required=13"):
        command._enforce_complete_future_horizon(torch.tensor([0]))


def test_true23_timeout_boundary_preserves_last_complete_horizon() -> None:
    command = _command()
    command._future_horizon_tail_steps = 11
    command.motion_num_steps = torch.tensor([20, 20])
    command.time_steps = torch.tensor([7, 8])
    command.motion_start_time_steps = torch.tensor([0, 0])
    command.motion_ids = torch.tensor([0, 1])
    command.motion_lib = SimpleNamespace(
        get_time_step_total=lambda _motion_ids: torch.tensor([20, 20])
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _name: command)
    )

    assert command.tracking_time_out_steps.tolist() == [9, 9]
    tracking_time_out = _production_tracking_time_out()
    assert tracking_time_out(env, "motion").tolist() == [False, True]


def test_true23_command_fails_closed_if_timeout_reset_is_missing() -> None:
    command = _command()
    command.use_adaptive_sampling = False
    command._future_horizon_tail_steps = 11
    command.motion_num_steps = torch.tensor([20])
    command.time_steps = torch.tensor([8])
    command.motion_start_time_steps = torch.tensor([0])

    with pytest.raises(RuntimeError, match="horizon timeout did not reset"):
        command._update_command()


def test_legacy_timeout_boundary_remains_full_clip_length() -> None:
    command = _command(true23=False)
    command._future_horizon_tail_steps = 999
    command.motion_num_steps = torch.tensor([20])
    command.time_steps = torch.tensor([18])
    command.motion_start_time_steps = torch.tensor([0])
    command.motion_ids = torch.tensor([3])
    command.motion_lib = SimpleNamespace(
        get_time_step_total=lambda _motion_ids: torch.tensor([20])
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _name: command)
    )

    assert command.tracking_time_out_steps.tolist() == [20]
    tracking_time_out = _production_tracking_time_out()
    assert tracking_time_out(env, "motion").tolist() == [False]
