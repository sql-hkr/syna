"""MNIST dataset"""

import gzip
import os
import shutil
import struct
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    with urllib.request.urlopen(url) as resp, open(dst, "wb") as out:
        shutil.copyfileobj(resp, out)


def _read_idx_images(gz_path: Path) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        (magic,) = struct.unpack(
            ">I", f.read(4)
        )  # not used except to advance buffer; magic should be 2051
        if magic != 2051:
            raise ValueError(f"Invalid magic number in {gz_path}: {magic}")
        n, rows, cols = struct.unpack(">III", f.read(12))
        data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        arr = arr.reshape(n, rows, cols)
        return arr


def _read_idx_labels(gz_path: Path) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        (magic,) = struct.unpack(">I", f.read(4))  # should be 2049
        if magic != 2049:
            raise ValueError(f"Invalid magic number in {gz_path}: {magic}")
        (n,) = struct.unpack(">I", f.read(4))
        data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return arr


class MNIST(Sequence):
    """Minimal MNIST dataset.

    Args:
        root: base directory to store/download MNIST files (defaults to ~/.syna/datasets)
        train: True for training split, False for test split
        download: whether to attempt download when files are missing
        transform: optional callable applied to images (receives numpy array HxW)
        target_transform: optional callable applied to labels
        flatten: whether to flatten images to (H*W,) vectors. If False, returns (1,H,W).
        normalize: if True, convert uint8 [0,255] to float32 [0,1]
    """

    def __init__(
        self,
        root: Optional[str] = None,
        train: bool = True,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        flatten: bool = True,
        normalize: bool = True,
    ) -> None:
        if root is None:
            root = os.path.join(Path.home(), ".syna", "datasets")
        self.root = Path(root) / "mnist"
        self.train = bool(train)
        self.download = bool(download)
        self.transform = transform
        self.target_transform = target_transform
        self.flatten = bool(flatten)
        self.normalize = bool(normalize)

        self._ensure_data()
        self._load()

    def _ensure_data(self) -> None:
        # ensure files exist (download if allowed)
        for key, fname in FILES.items():
            gz_path = self.root / fname
            if not gz_path.exists():
                if not self.download:
                    raise FileNotFoundError(
                        f"Missing MNIST file {gz_path}; set download=True"
                    )
                url = MNIST_URL + fname
                _download(url, gz_path)

    def _load(self) -> None:
        if self.train:
            images_path = self.root / FILES["train_images"]
            labels_path = self.root / FILES["train_labels"]
        else:
            images_path = self.root / FILES["test_images"]
            labels_path = self.root / FILES["test_labels"]

        imgs = _read_idx_images(images_path)
        labs = _read_idx_labels(labels_path)

        # preprocess
        if self.flatten:
            imgs = imgs.reshape(imgs.shape[0], -1)
        else:
            imgs = imgs.reshape(imgs.shape[0], 1, imgs.shape[1], imgs.shape[2])

        imgs = imgs.astype(np.float32)
        if self.normalize:
            imgs = imgs / 255.0

        self.data = imgs
        self.targets = labs.astype(np.int64)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        x = self.data[idx]
        y = int(self.targets[idx])
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


def load_mnist(
    root: Optional[str] = None, train: bool = True, download: bool = True
) -> MNIST:
    """Convenience factory for MNIST dataset."""
    return MNIST(root=root, train=train, download=download)
