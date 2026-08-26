from typing import List, Optional, Union

import torch


def length_to_mask(
    length: Union[torch.Tensor, List],
    max_len: Optional[int] = None,
    device=None,
) -> torch.Tensor:
    if isinstance(length, list):
        if device is None:
            device = "cpu"
        length = torch.tensor(length, device=device)

    if device is not None:
        assert device == length.device
    device = length.device

    if max_len is None:
        max_len = max(length)

    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask
