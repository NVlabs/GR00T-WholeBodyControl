from pathlib import Path

import numpy as np
import torch

from gear_sonic.envs.mjlab.native124_multi_motion import (
    blend_reference_actions,
    blend_reference_leg_actions,
    deterministic_window_offsets,
    lead_reference_steps,
    weighted_span_indices,
)
from gear_sonic.utils.g1_23dof_incremental_corpus import ARRAY_NAMES, build_corpus, load_catalog


def _motion(path: Path, frames: int, marker: float) -> None:
    arrays = {
        "joint_pos": np.full((frames, 23), marker, np.float32),
        "joint_vel": np.full((frames, 23), marker, np.float32),
        "body_pos_w": np.full((frames, 24, 3), marker, np.float32),
        "body_quat_w": np.zeros((frames, 24, 4), np.float32),
        "body_lin_vel_w": np.full((frames, 24, 3), marker, np.float32),
        "body_ang_vel_w": np.full((frames, 24, 3), marker, np.float32),
    }
    arrays["body_quat_w"][..., 0] = 1.0
    np.savez(path, fps=np.asarray([50.0]), **arrays)


def test_build_corpus_pads_short_motion_without_crossing(tmp_path: Path) -> None:
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    _motion(first, 40, 1.0)
    _motion(second, 620, 2.0)
    catalog = build_corpus(
        (("a", first, 1.0), ("b", second, 4.0)),
        tmp_path / "corpus.npz",
        tmp_path / "corpus.spans.json",
    )
    restored = load_catalog(tmp_path / "corpus.spans.json")
    assert restored == catalog
    assert [span.stored_length for span in catalog.spans] == [500, 620]
    with np.load(catalog.corpus_path) as data:
        assert set(ARRAY_NAMES).issubset(data.files)
        assert np.all(data["joint_pos"][40:500] == 1.0)
        assert np.all(data["joint_vel"][40:500] == 0.0)
        assert np.all(data["joint_pos"][500:] == 2.0)


def test_weighted_selection_prefers_larger_weight() -> None:
    torch.manual_seed(7)
    selected = weighted_span_indices(torch.tensor([1.0, 9.0]), 10_000)
    assert float((selected == 1).float().mean()) > 0.87


def test_weighted_selection_can_mask_other_clips() -> None:
    selected = weighted_span_indices(torch.tensor([0.0, 1.0, 0.0]), 100)
    assert torch.equal(selected, torch.ones(100, dtype=torch.long))


def test_lead_reference_steps_clamps_to_window() -> None:
    current = torch.tensor([10, 25, 49])
    stops = torch.tensor([20, 30, 50])
    assert lead_reference_steps(current, stops, 5).tolist() == [15, 29, 49]


def test_deterministic_window_offsets_ignore_reset_order() -> None:
    env_ids = torch.tensor([7, 2, 11])
    counts = torch.tensor([0, 4, 1])
    widths = torch.tensor([19, 31, 7])
    direct = deterministic_window_offsets(env_ids, counts, widths, seed=424242)
    order = torch.tensor([2, 0, 1])
    reordered = deterministic_window_offsets(
        env_ids[order], counts[order], widths[order], seed=424242
    )
    assert torch.equal(direct[order], reordered)
    assert bool((direct >= 0).all())
    assert bool((direct < widths).all())


def test_reference_action_blend_uses_raw_pd_target() -> None:
    policy = torch.tensor([[1.0, -1.0]])
    reference = torch.tensor([[0.5, -0.5]])
    result = blend_reference_actions(policy, reference, 0.25, 0.0, 0.5)
    assert torch.allclose(result, torch.tensor([[1.5, -1.5]]))


def test_reference_leg_blend_leaves_upper_body_unchanged() -> None:
    policy = torch.zeros((1, 23))
    reference = torch.ones((1, 23))
    result = blend_reference_leg_actions(policy, reference, 0.5, 0.0, 1.0)
    assert torch.equal(result[:, :12], torch.full((1, 12), 2.0))
    assert torch.equal(result[:, 12:], torch.zeros((1, 11)))
