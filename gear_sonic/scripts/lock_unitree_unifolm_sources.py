"""Discover immutable Unitree UniFoLM dataset revisions for operator review."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

from huggingface_hub import HfApi
import tyro

from gear_sonic.data.unitree_conversion.contracts import SourceLock
from gear_sonic.data.unitree_conversion.provenance import (
    discover_collection_lock,
    dump_source_lock,
)

DEFAULT_COLLECTION_SLUGS = (
    "unitreerobotics/unifolm-g1-dex3-dataset",
    "unitreerobotics/unifolm-wbt-dataset",
)


@dataclass(frozen=True)
class LockConfig:
    """CLI options for immutable collection discovery."""

    output_lock: Path
    """Review-draft YAML path to create."""

    force: bool = False
    """Atomically replace this exact output path if it already exists."""


def write_discovered_lock(
    lock: SourceLock,
    output_lock: Path,
    *,
    force: bool = False,
) -> Path:
    """Write a review draft exclusively, or atomically replace it with --force."""
    output_path = Path(output_lock)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_text = dump_source_lock(lock)

    if not force:
        try:
            with output_path.open("x", encoding="utf-8") as stream:
                stream.write(lock_text)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite existing source lock {output_path}; "
                "pass --force to replace this exact file"
            ) from None
        return output_path.resolve()

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(lock_text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path.resolve()


def main(config: LockConfig) -> None:
    lock = discover_collection_lock(
        api=HfApi(),
        collection_slugs=DEFAULT_COLLECTION_SLUGS,
    )
    write_discovered_lock(lock, config.output_lock, force=config.force)


if __name__ == "__main__":
    main(tyro.cli(LockConfig))
