"""
Simple logging utilities for teleop evaluation.
"""

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

import decoupled_wbc


class EvaluationLogger:
    """Simple logger that writes evaluation metrics to a timestamped file."""

    def __init__(self, log_subdir: str = "logs_teleop"):
        """
        Initialize the evaluation logger.

        Args:
            log_subdir: Subdirectory name under project root for logs
        """
        self.log_subdir = log_subdir
        self._setup_logging()

    def _setup_logging(self):
        """Setup simple file logging with timestamp-based filename"""
        # Get project root directory
        project_root = Path(os.path.dirname(decoupled_wbc.__file__)).parent

        # Create logs directory if it doesn't exist
        self.logs_dir = project_root / self.log_subdir
        self.logs_dir.mkdir(exist_ok=True)

        # Create timestamp-based filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"eval_{timestamp}.log"
        self.log_file_path = self.logs_dir / log_filename

        # Open file for writing
        self.log_file = open(self.log_file_path, "w")

    def write(self, message: str):
        """Write a message to the log file."""
        self.log_file.write(message + "\n")
        self.log_file.flush()  # Ensure immediate write

    def log_metrics(self, metrics: Dict[str, Any]):
        """
        Log evaluation metrics to file - just like the original print statements.

        Args:
            metrics: Dictionary of metric names and values
        """
        for metric_name, value in metrics.items():
            if isinstance(value, np.ndarray):
                # Convert numpy arrays to readable format
                if value.size == 1:
                    self.write(f"{metric_name}: {value.item()}")
                else:
                    self.write(f"{metric_name}: {value}")
            else:
                self.write(f"{metric_name}: {value}")

    def get_log_file_path(self) -> Path:
        """Get the path to the current log file."""
        return self.log_file_path

    def print(self):
        """Print out the contents of the log file."""
        if self.log_file_path.exists():
            with open(self.log_file_path, "r") as f:
                content = f.read()
                print(content)
        else:
            print(f"Log file not found: {self.log_file_path}")

    def close(self):
        """Close the log file."""
        if hasattr(self, "log_file") and self.log_file:
            self.log_file.close()
