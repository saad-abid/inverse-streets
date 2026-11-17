"""Load layouts + indicator labels per split, scale targets, and wrap them as a PyTorch Dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import PROJECT_ROOT

SPLITS = ("train", "val", "test")


@dataclass
class SplitData:
    """Images (N, H, W) uint8, targets (N, T) float32 and metadata (id, style) of one split."""
    images: np.ndarray
    targets: np.ndarray
    meta: pd.DataFrame


def load_splits(cfg: dict, targets: list[str]) -> dict[str, SplitData]:
    """Read data/layouts.npz and data/labels.csv and return train/val/test arrays."""
    data_dir = Path(PROJECT_ROOT) / cfg["paths"]["data_dir"]
    images = np.load(data_dir / "layouts.npz")["images"]
    labels = pd.read_csv(data_dir / "labels.csv")
    if not np.array_equal(labels["id"].to_numpy(), np.arange(len(images))):
        raise ValueError("labels.csv ids do not match layouts.npz; rerun src.build_labels")
    out = {}
    for split in SPLITS:
        mask = (labels["split"] == split).to_numpy()
        out[split] = SplitData(images=images[mask],
                               targets=labels.loc[mask, targets].to_numpy(np.float32),
                               meta=labels.loc[mask, ["id", "style"]].reset_index(drop=True))
    return out


@dataclass
class TargetScaler:
    """Standardize each target with the training mean and standard deviation."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, y: np.ndarray) -> "TargetScaler":
        std = y.std(axis=0)
        return cls(mean=y.mean(axis=0), std=np.where(std > 0, std, 1.0))

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mean) / self.std).astype(np.float32)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> "TargetScaler":
        return cls(mean=np.array(d["mean"]), std=np.array(d["std"]))


def random_dihedral(x: torch.Tensor) -> torch.Tensor:
    """Random rotation by 0/90/180/270 degrees and optional flip. The indicators of a
    layout do not change under these transforms (in expectation for the traffic ones)."""
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, dims=(-2, -1))
    if bool(torch.randint(0, 2, (1,))):
        x = torch.flip(x, dims=(-1,))
    return x


class LayoutDataset(Dataset):
    """Images as float tensors (1, H, W) in {0, 1} and scaled targets."""

    def __init__(self, images: np.ndarray, targets: np.ndarray, augment: bool = False) -> None:
        self.images = torch.from_numpy(images.astype(np.float32)).unsqueeze(1)
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.images[i]
        return (random_dihedral(x) if self.augment else x), self.targets[i]
