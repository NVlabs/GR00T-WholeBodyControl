import logging
import os
from typing import Optional

import colorlog


def setup_logging(
    name: Optional[str] = None,
    run_dir: Optional[str] = None,
    rank: Optional[int] = 0,
    level=logging.INFO,
):
    # Get the root logger
    root_logger = logging.getLogger()

    # Configure the root logger
    root_logger.setLevel(level)

    # Ensure hydra or other libraries aren't adding handlers
    root_logger.handlers.clear()

    # file handler only at rank 0, when we can create the path
    if rank == 0 and run_dir is not None and name is not None:
        # Create file handler: save to this file
        file_handler = logging.FileHandler(os.path.join(run_dir, f"{name}.log"))
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s  %(message)s",
            datefmt="%d/%m/%y %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # stdout logging, a bit more fancy
    formatter = colorlog.ColoredFormatter(
        "[%(white)s%(asctime)s%(reset)s] %(log_color)s%(levelname)s%(reset)s  %(message)s",
        datefmt="%d/%m/%y %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "purple",
            "INFO": "blue",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bg_white",
        },
        secondary_log_colors={},
        style="%",
    )
    stream_handler = colorlog.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    return root_logger
