"""Run exact-vector rank256 survival score by reusing the bounded parent engine."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from gear_sonic.scripts import train_g1_true23_sonic_rank256_disturbance_survival_score as parent
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_exact_impulse_survival_score_v1.json"
)
CONTRACT_SHA256 = "36dbd48fb1abc7b66ed25506091575a0b7a7e67694810a4676c9bc52b789f7ab"
SCALES = (-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
RESULT_FILENAME = "rank256_exact_impulse_survival_score_result.json"
FAILURE_FILENAME = "rank256_exact_impulse_survival_score_failure.json"
CHECKPOINT_FILENAME = "rank256_exact_impulse_survival_score_candidate.pt"
PARENT_RESULT_PATH = Path(
    "/root/g1_true23_runs/sonic_rank256_disturbance_survival_score_v1_retry2/"
    "rank256_disturbance_survival_score_result.json"
)
PARENT_RESULT_SHA256 = "aa6d25c8f6ec4d5e14b9ea7789bbd77d4353dda7269d5b0264c08cf32fdd7b7c"


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("exact impulse contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    if (
        body.get("kind") != "g1_true23_sonic_rank256_exact_impulse_survival_score_contract_v1"
        or body.get("gradient", {}).get("scales") != list(SCALES)
        or body.get("collection", {}).get("all_envs_receive_exact_vector") is not True
        or body.get("boundaries", {}).get("hardware_authorized") is not False
        or sha256_file(PARENT_RESULT_PATH) != PARENT_RESULT_SHA256
    ):
        raise ValueError("exact impulse contract semantic/input mismatch")
    return body


@contextmanager
def _scope(root: Path) -> Iterator[None]:
    contract = _load_contract(root)
    names = (
        "CONTRACT_RELATIVE_PATH",
        "CONTRACT_SHA256",
        "KIND",
        "SCALES",
        "RESULT_FILENAME",
        "FAILURE_FILENAME",
        "CHECKPOINT_FILENAME",
        "_load_contract",
        "_impulse_vectors",
    )
    saved = {name: getattr(parent, name) for name in names}
    base_contract = copy.deepcopy(saved["_load_contract"](root))
    base_contract["kind"] = contract["kind"]
    base_contract["role"] = contract["role"]
    base_contract["collection"] = {
        **dict(base_contract["collection"]),
        **dict(contract["collection"]),
    }
    base_contract["gradient"] = {
        **dict(base_contract["gradient"]),
        **dict(contract["gradient"]),
    }
    base_contract["evaluation"] = {
        **dict(base_contract["evaluation"]),
        **dict(contract["evaluation"]),
    }
    base_contract["boundaries"] = dict(contract["boundaries"])
    base_contract["parent_exact_impulse_contract"] = {
        "path": CONTRACT_RELATIVE_PATH.as_posix(),
        "sha256": CONTRACT_SHA256,
        "parent_result_sha256": PARENT_RESULT_SHA256,
    }

    def load_current(_root: Path) -> Mapping[str, Any]:
        return copy.deepcopy(base_contract)

    def exact_vectors(num_envs: int, device: Any) -> Any:
        import torch

        vector = np.asarray(contract["collection"]["exact_vector"], dtype=np.float32)
        values = np.repeat(vector.reshape(1, 6), num_envs, axis=0)
        return torch.from_numpy(values).to(device=device)

    try:
        parent.CONTRACT_RELATIVE_PATH = CONTRACT_RELATIVE_PATH
        parent.CONTRACT_SHA256 = CONTRACT_SHA256
        parent.KIND = "g1_true23_sonic_rank256_exact_impulse_survival_score_result_v1"
        parent.SCALES = SCALES
        parent.RESULT_FILENAME = RESULT_FILENAME
        parent.FAILURE_FILENAME = FAILURE_FILENAME
        parent.CHECKPOINT_FILENAME = CHECKPOINT_FILENAME
        parent._load_contract = load_current
        parent._impulse_vectors = exact_vectors
        yield
    finally:
        for name, value in saved.items():
            setattr(parent, name, value)


def preflight(repository_root: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        return parent.preflight(root)


def run(repository_root: Path, run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        return parent.run(root, run_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repository-root", type=Path, default=ROOT)
    train = sub.add_parser("train")
    train.add_argument("--repository-root", type=Path, default=ROOT)
    train.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight(args.repository_root)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    try:
        result = run(args.repository_root, args.run_dir)
    except Exception as error:
        print(f"exact impulse survival score failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result["assessment"], indent=2, sort_keys=True))  # noqa: T201
    return 0 if result.get("candidate") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
