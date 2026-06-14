import random
from typing import List, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticMotionDataset(Dataset):
    """Synthetic motion dataset for training without the full motion dataset.

    Each sample is a random tensor of shape [T, feat_dim] where T is drawn
    uniformly from [min_frames, max_frames] and feat_dim matches the motion
    representation dimensionality (e.g. 418 for G1Skeleton34).
    """

    def __init__(
        self,
        feat_dim: int,
        num_samples: int = 1000,
        min_frames: int = 80,
        max_frames: int = 300,
    ):
        self.feat_dim = feat_dim
        self.num_samples = num_samples
        self.lengths = [
            random.randint(min_frames, max_frames) for _ in range(num_samples)
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T = self.lengths[idx]
        motion = torch.randn(T, self.feat_dim)
        return {"keyid": idx, "motion": motion}


def collate_tensors(
    tensor_batch: List[Tensor],
    size: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pad variable-length tensors to a common length.

    Returns:
        - padded motions [B, T, D]
        - lengths [B]
        - pad_mask [B, T] (True where valid)
    """
    rep_dim = tensor_batch[0].shape[1]
    max_size = max(mo.shape[0] for mo in tensor_batch)
    if size is not None:
        assert size >= max_size
        max_size = size

    motion_batch = torch.zeros(len(tensor_batch), max_size, rep_dim)
    pad_mask = torch.zeros(len(tensor_batch), max_size, dtype=torch.bool)
    lengths = []
    for bi, mo in enumerate(tensor_batch):
        cur_len = mo.shape[0]
        lengths.append(cur_len)
        motion_batch[bi, :cur_len] = mo
        pad_mask[bi, :cur_len] = True

    len_batch = torch.tensor(lengths)
    return motion_batch, len_batch, pad_mask


def collate_batch(batch: List[Dict]) -> Dict:
    """Collate a batch of motion dicts into padded tensors."""
    motion = [bdict["motion"] for bdict in batch]
    motion, motion_len, motion_pad_mask = collate_tensors(motion)
    return {
        "motion": motion,
        "motion_len": motion_len,
        "motion_pad_mask": motion_pad_mask,
        "batch_size": len(motion),
    }
