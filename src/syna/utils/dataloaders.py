"""Data loader utilities.

Provides simple iterable data loaders used by training loops: a batched
DataLoader and a sequence-aware SeqDataLoader.
"""

import math
from typing import Any, Iterable, List, Tuple

import numpy as np


class DataLoader:
    """
    Simple iterable data loader.

    Args:
        dataset: Sequence of (input, target) pairs.
        batch_size: Number of samples per batch.
        shuffle: If True, shuffle dataset order at the start of each epoch.

    Behavior:
        - Iterates over dataset in batches. The last batch may be smaller.
        - Iteration raises StopIteration at epoch end and automatically resets
          for the next epoch.
    """

    def __init__(
        self, dataset: Iterable[Tuple[Any, Any]], batch_size: int, shuffle: bool = True
    ):
        self.dataset = list(dataset)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.data_size = len(self.dataset)
        self.max_iter = (
            math.ceil(self.data_size / self.batch_size) if self.batch_size > 0 else 0
        )
        self.reset()

    def reset(self) -> None:
        """Reset iteration counters and prepare indices for the next epoch."""
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the next batch (x, t) as NumPy arrays.
        Raises StopIteration at the end of an epoch and resets internally.
        """
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        start = self.iteration * self.batch_size
        end = start + self.batch_size
        batch_index = self.index[start:end]
        batch = [self.dataset[i] for i in batch_index]

        x = np.asarray([example[0] for example in batch])
        t = np.asarray([example[1] for example in batch])

        self.iteration += 1
        return x, t

    # Python 2 compatibility alias (optional)
    def next(self):
        return self.__next__()


class SeqDataLoader(DataLoader):
    """
    Sequence-aware loader that yields batch_size sequences in parallel.

    The loader divides the dataset into batch_size streams by computing a
    'jump' = data_size // batch_size and, for each iteration step k, takes
    the elements at positions (i*jump + k) % data_size for i in range(batch_size).

    Args:
        dataset: Sequence of (input, target) pairs.
        batch_size: Number of parallel sequences (streams) per batch.

    Behavior:
        - shuffle is always disabled for sequence loader.
        - Iteration yields exactly data_size // jump steps (i.e., max_iter inherited).
    """

    def __init__(self, dataset: Iterable[Tuple[Any, Any]], batch_size: int):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False)

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        # jump sets the offset between streams to evenly partition the data
        jump = max(1, self.data_size // max(1, self.batch_size))
        indices: List[int] = [
            (i * jump + self.iteration) % self.data_size for i in range(self.batch_size)
        ]
        batch = [self.dataset[i] for i in indices]

        x = np.asarray([example[0] for example in batch])
        t = np.asarray([example[1] for example in batch])

        self.iteration += 1
        return x, t
