"""Command-line interface for the BrainCo data exporter."""

import tyro

from .config import BraincoDataExporterConfig
from .main import main as run

def main() -> None:
    config = tyro.cli(BraincoDataExporterConfig)
    run(config)
