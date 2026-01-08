"""
Time-series data splitting utilities (no information leakage).

All splits are chronological and contiguous (no shuffling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _sorted_df(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    if time_col not in df.columns:
        raise ValueError(f"df must contain '{time_col}' column")
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    return out.sort_values(time_col).reset_index(drop=True)


def train_val_test_split(
    df: pd.DataFrame,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    time_col: str = "timestamp",
    gap: int = 0,
) -> Split:
    """
    Chronological train/val/test split.

    Args:
        gap: number of rows to skip between splits (prevents leakage when features
             use future/rolling statistics).
    """
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("ratios must be non-negative")
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1")

    df = _sorted_df(df, time_col=time_col)
    n = len(df)
    if n == 0:
        return Split(df, df, df)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_end = max(0, n_train)
    val_start = min(n, train_end + gap)
    val_end = min(n, val_start + n_val)
    test_start = min(n, val_end + gap)

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[val_start:val_end].reset_index(drop=True)
    test = df.iloc[test_start:].reset_index(drop=True)
    return Split(train=train, val=val, test=test)


def rolling_window_splits(
    df: pd.DataFrame,
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
    time_col: str = "timestamp",
    gap: int = 0,
    max_windows: Optional[int] = None,
) -> Iterator[Tuple[int, Split]]:
    """
    Rolling-window splits for non-stationary evaluation.

    Windows are:
      [train][gap][val][gap][test]
    and then slide forward by step_size rows.
    """
    if min(train_size, val_size, test_size, step_size) <= 0:
        raise ValueError("sizes must be positive")

    df = _sorted_df(df, time_col=time_col)
    n = len(df)

    start = 0
    window_idx = 0
    while True:
        train_start = start
        train_end = train_start + train_size
        val_start = train_end + gap
        val_end = val_start + val_size
        test_start = val_end + gap
        test_end = test_start + test_size

        if test_end > n:
            break

        split = Split(
            train=df.iloc[train_start:train_end].reset_index(drop=True),
            val=df.iloc[val_start:val_end].reset_index(drop=True),
            test=df.iloc[test_start:test_end].reset_index(drop=True),
        )
        yield window_idx, split

        window_idx += 1
        if max_windows is not None and window_idx >= max_windows:
            break
        start += step_size


def slice_by_time(
    df: pd.DataFrame,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Convenience helper for slicing by timestamp (inclusive start, exclusive end)."""
    df = _sorted_df(df, time_col=time_col)
    t = df[time_col]

    if start is not None:
        df = df[t >= pd.to_datetime(start)]
    if end is not None:
        df = df[t < pd.to_datetime(end)]

    return df.reset_index(drop=True)

