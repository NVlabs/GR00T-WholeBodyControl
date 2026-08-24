"""Versioned late-actor phase-balance qualification entrypoint."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from gear_sonic.scripts import qualify_g1_true23_sonic_rank256_phase_balance as core

CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_phase_balance_qualification_v2.json"
)
CONTRACT_SHA256 = "b8d862c679c823f0f453982db7a9c738fc1ff005c8f892b6e692d8dfbc1addc6"
CONTRACT_KIND = "g1_true23_sonic_rank256_phase_balance_qualification_contract_v2"
RUN_DIR_DEFAULT = Path("/root/g1_true23_runs/sonic_rank256_phase_balance_qualification_v2")
RESULT_FILENAME = "rank256_phase_balance_qualification_result_v2.json"


@contextmanager
def _scope() -> Iterator[None]:
    names = (
        "CONTRACT_RELATIVE_PATH",
        "CONTRACT_SHA256",
        "CONTRACT_KIND",
        "RUN_DIR_DEFAULT",
        "RESULT_FILENAME",
        "SOURCE_RELATIVE_PATHS",
    )
    saved = {name: getattr(core, name) for name in names}
    try:
        core.CONTRACT_RELATIVE_PATH = CONTRACT_RELATIVE_PATH
        core.CONTRACT_SHA256 = CONTRACT_SHA256
        core.CONTRACT_KIND = CONTRACT_KIND
        core.RUN_DIR_DEFAULT = RUN_DIR_DEFAULT
        core.RESULT_FILENAME = RESULT_FILENAME
        core.SOURCE_RELATIVE_PATHS = (
            CONTRACT_RELATIVE_PATH,
            Path("gear_sonic/scripts/qualify_g1_true23_sonic_rank256_phase_balance_v2.py"),
            Path("gear_sonic/scripts/qualify_g1_true23_sonic_rank256_phase_balance.py"),
            Path("gear_sonic/scripts/screen_g1_true23_sonic_rank256_balance_residual.py"),
            Path("gear_sonic/scripts/collect_g1_true23_sonic_rank256_shifted_base_causal_recovery_score.py"),
            Path("gear_sonic/scripts/screen_g1_true23_sonic_rank256_shifted_base_causal_recovery.py"),
        )
        yield
    finally:
        for name, value in saved.items():
            setattr(core, name, value)


def main(argv: Sequence[str] | None = None) -> int:
    with _scope():
        return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
