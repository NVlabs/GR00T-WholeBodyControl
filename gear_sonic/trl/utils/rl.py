"""RL-specific helpers: episode attention masks and (legacy) schedule utilities."""

import torch


def compute_episode_attnmask(dones):
    """
    Compute an attention mask that prevents the model from attending to observations from different episodes.

    Args:
        dones (torch.Tensor): A tensor of shape (num_envs, num_steps) indicating when each environment episode ends.
                                A value of 1.0 indicates the end of an episode.

    Returns:
        torch.Tensor: An attention mask of shape (num_envs, num_steps, num_steps) where True values indicate
                        positions that should be masked (i.e., the model should not attend to these positions).
    """
    # Create cumulative sum of dones to identify different episodes
    episode_starts = torch.roll(dones, 1, dims=1)
    episode_starts[:, 0] = True  # First step is always start of an episode
    episode_ids = torch.cumsum(episode_starts, dim=1)  # (num_envs, num_steps)

    # Expand episode_ids for broadcasting
    episode_ids_i = episode_ids.unsqueeze(2)  # (num_envs, num_steps, 1)
    episode_ids_j = episode_ids.unsqueeze(1)  # (num_envs, 1, num_steps)

    # Create mask where True indicates positions from different episodes
    attnmask = episode_ids_i != episode_ids_j
    return attnmask
