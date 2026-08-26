# numpy / torch / pytorch_lightning should not be imported
# even implicitely in the imports
# so that the logging behave properly
import logging
import os
from typing import Optional

from motionbricks.motionlib.core.utils.custom_logging import setup_logging


# from lightning_fabric/utilities/rank_zero.py
# but return 0
def get_rank() -> Optional[int]:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return 0


def setup_train_logging(run_dir: str, rank: int, level=logging.INFO):
    return setup_logging(
        name="train",
        run_dir=run_dir,
        rank=rank,
        level=level,
    )
