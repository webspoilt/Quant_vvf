"""
QUANT_VVF — Deep Learning Training Pipeline & Walk-Forward Cross-Validation.
Implements:
- Chronological train / validation / test splits (strict zero lookahead)
- Walk-forward rolling evaluation with purging and embargo
- Checkpointing and reproducibility seeding
"""

from __future__ import annotations
import os
import random
from typing import Dict, Any, Tuple, List, Generator
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for exact reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chronological_split(
    data: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Strict chronological dataset split without random shuffling.
    Train: [0 : N_train]
    Validation: [N_train : N_train + N_val]
    Test: [N_train + N_val : N]
    """
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = data[:n_train]
    val_data = data[n_train : n_train + n_val]
    test_data = data[n_train + n_val :]

    return train_data, val_data, test_data


def purged_walk_forward_splits(
    total_length: int,
    train_window: int,
    test_window: int,
    purge_window: int = 5,
    embargo_window: int = 2,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Purged & Embargoed walk-forward cross-validation splits (López de Prado 2018).
    Prevents leakage between overlapping prediction horizons.
    """
    start = 0
    while start + train_window + purge_window + test_window <= total_length:
        train_indices = np.arange(start, start + train_window)
        test_start = start + train_window + purge_window
        test_indices = np.arange(test_start, test_start + test_window)

        yield train_indices, test_indices

        # Advance start with embargo
        start += test_window + embargo_window
