from functools import lru_cache
from pathlib import Path

import pytest
import torch

from gear_sonic.trl.mjlab.true23_actor import True23SonicCore
from gear_sonic.utils.g1_23dof_artifact import build_true23_policy_pair
from gear_sonic.utils.g1_23dof_checkpoint_io import (
    load_safe_true23_checkpoint,
)


@lru_cache(maxsize=1)
def _paths() -> tuple[Path, dict]:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = repo_root / "sonic_release/g1_23dof_rev_1_0_init.pt"
    return checkpoint_path, load_safe_true23_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )


def test_mjlab_core_has_exact_checkpoint_namespace_and_true23_head() -> None:
    checkpoint_path, checkpoint = _paths()
    core = True23SonicCore(checkpoint_path)
    expected = {
        key
        for key in checkpoint["policy_state_dict"]
        if key != "std"
    }

    assert set(core.state_dict()) == expected
    assert core.actor_module.encoders["teleop"].module[0].in_features == 267
    assert core.actor_module.decoders["g1_dyn"].module[0].in_features == 994
    assert core.actor_module.decoders["g1_dyn"].module[-1].out_features == 23


def test_mjlab_core_forward_is_bit_exact_with_promotion_reconstruction() -> None:
    torch.manual_seed(7)
    checkpoint_path, checkpoint = _paths()
    core = True23SonicCore(checkpoint_path).eval()
    encoder, decoder, _ = build_true23_policy_pair(checkpoint)
    teleop = torch.randn(4, 267)
    proprioception = torch.randn(4, 930)

    with torch.no_grad():
        actual = core(teleop, proprioception)
        expected = decoder(
            torch.cat((encoder(teleop), proprioception), dim=-1)
        )

    assert torch.equal(actual, expected)
    assert actual.shape == (4, 23)


def test_mjlab_fsq_keeps_training_gradient_but_exact_forward_values() -> None:
    torch.manual_seed(11)
    checkpoint_path, _ = _paths()
    core = True23SonicCore(checkpoint_path)
    teleop = torch.randn(2, 267)
    proprioception = torch.randn(2, 930)

    core(teleop, proprioception).square().mean().backward()

    gradient = core.actor_module.encoders["teleop"].module[0].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize(
    ("teleop_dim", "proprioception_dim", "message"),
    [
        (266, 930, "teleop observation"),
        (267, 929, "proprioception"),
    ],
)
def test_mjlab_core_rejects_contract_shape_drift(
    teleop_dim: int,
    proprioception_dim: int,
    message: str,
) -> None:
    checkpoint_path, _ = _paths()
    core = True23SonicCore(checkpoint_path)

    with pytest.raises(ValueError, match=message):
        core(
            torch.zeros(1, teleop_dim),
            torch.zeros(1, proprioception_dim),
        )
