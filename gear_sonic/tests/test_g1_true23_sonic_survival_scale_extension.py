from __future__ import annotations

from pathlib import Path

from gear_sonic.scripts import trace_g1_true23_sonic_survival_scale_extension as extension

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _record(scale: float, completed: int, q9: int, termination: str = "ee_body_pos") -> dict[str, object]:
    return {
        "scale": scale,
        "completed_transitions": completed,
        "terminal_q9": q9,
        "termination_names": [termination],
        "episode_return": -100.0,
        "policy_state_sha256": f"{int(scale * 100):064x}",
        "nonfinite_count": 0,
        "raw_clip_required_count": 0,
        "action_semantics_mismatch_count": 0,
        "q9_discontinuity_count": 0,
    }


def test_extension_contract_is_hash_locked() -> None:
    contract = extension._load_contract(REPOSITORY_ROOT)
    assert contract["construction"]["scales"] == list(extension.SCALES)
    assert contract["evaluation_gate"]["minimum_terminal_q9"] == 190


def test_extension_selects_longest_clean_policy_across_allowed_terminations() -> None:
    records = [_record(scale, 200, 208) for scale in extension.SCALES]
    records[3] = _record(2.0, 358, 366, "anchor_pos")
    records[4] = _record(2.5, 300, 308, "anchor_ori")
    result = extension.assess(records)
    assert result["candidate_selected"] is True
    assert result["selected_scale"] == 2.0
    assert result["selected_completed_transitions"] == 358
    assert result["support_qualified"] is False


def test_extension_rejects_dirty_or_short_policies() -> None:
    records = [_record(scale, 181, 189) for scale in extension.SCALES]
    records[0] = _record(1.25, 300, 308)
    records[0]["raw_clip_required_count"] = 1
    result = extension.assess(records)
    assert result["candidate_selected"] is False
