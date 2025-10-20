"""Datasets — minimal dataset base class (MNIST behavior compatible with torchvision)."""

from typing import Any, Callable, Optional

import numpy as np


def _identity(x: Any) -> Any:
    return x


class Dataset:
    """Minimal base dataset.

    Subclasses should override prepare() to populate self.data (and optionally self.label).

    Args:
        train (bool): Whether this is a training split.
        transform (callable, optional): Function applied to each data item.
        target_transform (callable, optional): Function applied to each label.
    Attributes:
        data: Sequence of data items (must support indexing and len()).
        label: Sequence of labels or None.
    """

    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
    ):
        self.train = train
        self.transform = transform or _identity
        self.target_transform = target_transform or _identity

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index: int):
        assert np.isscalar(index), "Index must be a scalar"
        item = self.transform(self.data[index])
        if self.label is None:
            return item, None
        return item, self.target_transform(self.label[index])

    def __len__(self) -> int:
        return len(self.data) if self.data is not None else 0

    def prepare(self):
        """Populate self.data (and optionally self.label). Must be implemented by subclasses."""
        raise NotImplementedError
