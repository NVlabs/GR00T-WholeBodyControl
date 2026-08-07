"""General utility functions for RL training scripts.

Provides argument-conflict resolution for Hydra/CLI arg lists, colour-coded
console print helpers, timestamp generation, model-args loading, random-seed
initialisation, and a scalar-to-RGB colour mapper.
"""

# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from datetime import datetime
import os
import random
import sys

import numpy as np
import torch


def solve_argv_conflict(args_list):
    """Remove entries from ``args_list`` that are overridden by ``sys.argv``.

    When programmatic defaults conflict with user-provided command-line
    arguments, this function removes the duplicates from ``args_list`` so
    that the command-line value wins.

    Args:
        args_list: Mutable list of argument strings (modified in-place).
            Entries that also appear in ``sys.argv[1:]`` (along with their
            positional values) are removed.
    """
    arguments_to_be_removed = []
    arguments_size = []

    for argv in sys.argv[1:]:
        if argv.startswith("-"):
            size_count = 1
            for i, args in enumerate(args_list):
                if args == argv:
                    arguments_to_be_removed.append(args)
                    for more_args in args_list[i + 1 :]:
                        if not more_args.startswith("-"):
                            size_count += 1
                        else:
                            break
                    arguments_size.append(size_count)
                    break

    for args, size in zip(arguments_to_be_removed, arguments_size):
        args_index = args_list.index(args)
        for _ in range(size):
            args_list.pop(args_index)


def print_error(*message):
    """Print an error message in red and raise ``RuntimeError``.

    Args:
        *message: Message fragments passed to ``print``.

    Raises:
        RuntimeError: Always raised after printing.
    """
    print("\033[91m", "ERROR ", *message, "\033[0m")
    raise RuntimeError


def print_ok(*message):
    """Print a success message in green.

    Args:
        *message: Message fragments passed to ``print``.
    """
    print("\033[92m", *message, "\033[0m")


def print_warning(*message):
    """Print a warning message in yellow.

    Args:
        *message: Message fragments passed to ``print``.
    """
    print("\033[93m", *message, "\033[0m")


def print_info(*message):
    """Print an informational message in cyan.

    Args:
        *message: Message fragments passed to ``print``.
    """
    print("\033[96m", *message, "\033[0m")


def get_time_stamp():
    """Return the current date-time as a formatted string.

    Returns:
        String of the form ``"MM-DD-YYYY-HH-MM-SS"``.
    """
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    hour = now.strftime("%H")
    minute = now.strftime("%M")
    second = now.strftime("%S")
    return f"{month}-{day}-{year}-{hour}-{minute}-{second}"


def parse_model_args(model_args_path):
    """Load model arguments from a Python-literal file as an ``argparse.Namespace``.

    The file is expected to contain a single Python dict literal that is
    ``eval``-ed and wrapped in ``argparse.Namespace``.

    Args:
        model_args_path: Path to the model-args file.

    Returns:
        ``argparse.Namespace`` with one attribute per dict key.
    """
    fp = open(model_args_path)
    model_args = eval(fp.read())
    model_args = argparse.Namespace(**model_args)

    return model_args


def seeding(seed=0, torch_deterministic=False):
    """Set global random seeds for reproducibility.

    Seeds ``random``, ``numpy``, ``torch`` (CPU and all GPUs), and the
    ``PYTHONHASHSEED`` environment variable.  Optionally enables cuDNN
    deterministic mode at the cost of performance.

    Args:
        seed: Integer seed value.
        torch_deterministic: If True, enables fully deterministic CUDA
            operations (sets ``CUBLAS_WORKSPACE_CONFIG``, disables cuDNN
            benchmarking, and calls ``torch.use_deterministic_algorithms(True)``).

    Returns:
        The seed value that was applied.
    """
    print(f"Setting seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def distance_l2(root_pos, wp_pos):
    """Compute the L2 distance between two position tensors.

    Args:
        root_pos: Reference position tensor.
        wp_pos: Waypoint position tensor of the same shape as ``root_pos``.

    Returns:
        Scalar tensor with the Euclidean distance.
    """
    return torch.norm(wp_pos - root_pos, dim=0)


def value_to_color(value, min_value, max_value):
    """
    Converts a numerical value to an RGB color.
    The color will range from blue (low values) to red (high values).
    """
    # Ensure value is within the range [0, max_value]
    value = max(min_value, min(value, max_value))

    # Calculate the proportion of the value
    red = (value - min_value) / (max_value - min_value)

    # Map the proportion to the red channel for a red gradient
    # Blue for minimum value and red for maximum value
    blue = 1 - red
    green = 0  # Keep green constant for simplicity

    # Return the RGB color
    return red, green, blue
