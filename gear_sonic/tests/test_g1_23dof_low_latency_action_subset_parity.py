"""Prove when 29-to-23 decoder distillation is a structural no-op.

The padded-H10 warm start keeps the complete decoder input and hidden trunk.
Its only decoder change is selecting the native-23 rows of the final affine
layer.  With the released SiLU activation, it must therefore equal those same
teacher outputs for every input, up to backend floating-point roundoff.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Mapping

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from gear_sonic.scripts.init_g1_23dof_checkpoint import (
    _load_pinned_legacy_release,
)
from gear_sonic.trl.mjlab.true23_actor import _ExactSiluMlp
from gear_sonic.trl.utils.g1_23dof_checkpoint import (
    initialize_policy_state_dict,
)
from gear_sonic.utils.g1_23dof_checkpoint_io import (
    load_safe_true23_checkpoint,
)
from gear_sonic.utils.g1_23dof_contract import (
    LOW_LATENCY_RELEASE_HF_REVISION,
    LOW_LATENCY_RELEASE_SHA256,
    NATIVE_IL23_TO_CANONICAL_IL29,
    REFERENCE_PROFILE_LOW_LATENCY,
    SOURCE_IL29_KEEP_INDICES,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ENV = "GEAR_SONIC_LOW_LATENCY_CHECKPOINT"
_DEFAULT_SOURCE = _REPOSITORY_ROOT / "low_latency" / "last.pt"
_MATERIALIZED_WARM_START = (
    _REPOSITORY_ROOT
    / "sonic_release"
    / "g1_23dof_rev_1_0_low_latency_init.pt"
)
_DECODER_PREFIX = "actor_module.decoders.g1_dyn.module."


def _decoder_linear_indices(state: Mapping[str, torch.Tensor]) -> tuple[int, ...]:
    return tuple(
        sorted(
            int(key.removeprefix(_DECODER_PREFIX).split(".", 1)[0])
            for key in state
            if key.startswith(_DECODER_PREFIX) and key.endswith(".weight")
        )
    )


def _forward_silu_decoder(
    state: Mapping[str, torch.Tensor], value: torch.Tensor
) -> torch.Tensor:
    indices = _decoder_linear_indices(state)
    for position, index in enumerate(indices):
        value = F.linear(
            value,
            state[f"{_DECODER_PREFIX}{index}.weight"],
            state[f"{_DECODER_PREFIX}{index}.bias"],
        )
        if position != len(indices) - 1:
            value = F.silu(value)
    return value


def _assert_native_subset_parity(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    decoder_inputs: torch.Tensor,
) -> None:
    source_indices = _decoder_linear_indices(source)
    target_indices = _decoder_linear_indices(target)
    assert target_indices == source_indices
    final_index = source_indices[-1]

    hidden_keys = [
        f"{_DECODER_PREFIX}{index}.{kind}"
        for index in source_indices[:-1]
        for kind in ("weight", "bias")
    ]
    assert all(torch.equal(target[key], source[key]) for key in hidden_keys)

    native_rows = torch.tensor(NATIVE_IL23_TO_CANONICAL_IL29)
    for kind in ("weight", "bias"):
        key = f"{_DECODER_PREFIX}{final_index}.{kind}"
        assert torch.equal(
            target[key],
            source[key].index_select(0, native_rows),
        )
    assert torch.equal(target["std"], source["std"].index_select(0, native_rows))

    with torch.inference_mode():
        teacher_subset = _forward_silu_decoder(source, decoder_inputs).index_select(
            1, native_rows
        )
        warm_start_output = _forward_silu_decoder(target, decoder_inputs)
    torch.testing.assert_close(
        warm_start_output,
        teacher_subset,
        rtol=1.0e-5,
        atol=5.0e-6,
    )


def test_subset_parity_is_structural_and_uses_native_action_order() -> None:
    generator = torch.Generator().manual_seed(20260806)
    source = {
        f"{_DECODER_PREFIX}0.weight": torch.randn(8, 994, generator=generator),
        f"{_DECODER_PREFIX}0.bias": torch.randn(8, generator=generator),
        f"{_DECODER_PREFIX}2.weight": torch.randn(29, 8, generator=generator),
        f"{_DECODER_PREFIX}2.bias": torch.randn(29, generator=generator),
        "std": torch.rand(29, generator=generator) + 0.1,
    }
    target, _ = initialize_policy_state_dict(source)
    decoder_inputs = torch.randn(7, 994, generator=generator)

    _assert_native_subset_parity(source, target, decoder_inputs)

    # Sorted retained canonical slots describe compact observations, not the
    # native PhysX action order.  Using them as output rows changes 16/23 joints.
    assert sum(
        native != compact
        for native, compact in zip(
            NATIVE_IL23_TO_CANONICAL_IL29,
            SOURCE_IL29_KEEP_INDICES,
            strict=True,
        )
    ) == 16
    runtime_probe = _ExactSiluMlp((2, 3, 1))
    assert isinstance(runtime_probe.module[1], nn.SiLU)
    assert not any(isinstance(layer, nn.ELU) for layer in runtime_probe.modules())


@pytest.fixture(scope="module")
def exact_low_latency_policy() -> tuple[dict[str, torch.Tensor], dict[str, str | None]]:
    source_path = Path(os.environ.get(_SOURCE_ENV, _DEFAULT_SOURCE)).expanduser()
    if not source_path.is_file():
        pytest.skip(
            f"exact low-latency release unavailable; set {_SOURCE_ENV}"
        )

    source_checkpoint, digest, release = _load_pinned_legacy_release(source_path)
    assert digest == LOW_LATENCY_RELEASE_SHA256
    assert release["source_revision"] == LOW_LATENCY_RELEASE_HF_REVISION
    assert release["reference_profile"] == REFERENCE_PROFILE_LOW_LATENCY
    policy = dict(source_checkpoint["policy_state_dict"])
    del source_checkpoint
    gc.collect()
    return policy, dict(release)


def test_exact_pinned_low_latency_warm_start_is_subset_equivalent(
    exact_low_latency_policy: tuple[
        dict[str, torch.Tensor], dict[str, str | None]
    ],
) -> None:
    source, _ = exact_low_latency_policy
    target, report = initialize_policy_state_dict(source)

    assert _decoder_linear_indices(source) == tuple(range(0, 17, 2))
    assert report["source_decoder_input_dim"] == 994
    assert report["target_decoder_input_dim"] == 994
    assert report["target_decoder_output_dim"] == 23

    generator = torch.Generator().manual_seed(20260806)
    decoder_inputs = torch.cat(
        (
            torch.zeros(1, 994),
            torch.randn(3, 994, generator=generator),
            3.0 * torch.randn(2, 994, generator=generator),
        )
    )
    _assert_native_subset_parity(source, target, decoder_inputs)

    # When present, also prove checked-in/generated warm-start material matches
    # a fresh conversion of the exact pinned source tensor-for-tensor.
    if _MATERIALIZED_WARM_START.is_file():
        materialized = load_safe_true23_checkpoint(_MATERIALIZED_WARM_START)
        materialized_policy = materialized["policy_state_dict"]
        assert materialized["g1_23dof_metadata"]["reference_profile"] == (
            REFERENCE_PROFILE_LOW_LATENCY
        )
        assert materialized_policy.keys() == target.keys()
        assert all(
            torch.equal(materialized_policy[key], target[key])
            for key in target
        )
