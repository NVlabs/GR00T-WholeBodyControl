"""Lightweight dataset helpers (dummy datasets, episode attention masks)."""

from datasets import Dataset


def create_dummy_dataset(num_samples: int = 100) -> Dataset:
    """Create a dummy dataset with the specified number of samples.

    Args:
        num_samples (int): Number of samples to create in the dataset.

    Returns:
        Dataset: A HuggingFace Dataset containing dummy prompts.
    """
    dummy_data = {"prompt": [f"Sample prompt {i}" for i in range(num_samples)]}
    return Dataset.from_dict(dummy_data)
