"""Data loading utilities for FE2MT.

This module provides unified data loading, preprocessing, splitting, and
patch extraction for HSI--LiDAR collaborative classification.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


def _load_mat_array(path: Path, key: str) -> np.ndarray:
    """Load an array from a MATLAB file."""
    data = loadmat(path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}")
    return data[key]


def _prepare_label_map(array: np.ndarray) -> np.ndarray:
    """Convert a label map to shape [H, W]."""
    array = np.squeeze(np.asarray(array))
    if array.ndim != 2:
        raise ValueError(f"Label map should be 2D, got {array.shape}")
    return array.astype(np.int64, copy=False)


def _ensure_hwc(
    array: np.ndarray,
    spatial_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    """Convert an input array to shape [H, W, C]."""
    array = np.asarray(array)

    if array.ndim == 2:
        array = array[..., None]
    elif array.ndim == 3:
        if array.shape[:2] == spatial_shape:
            pass
        elif array.shape[1:] == spatial_shape:
            array = np.transpose(array, (1, 2, 0))
        else:
            raise ValueError(
                f"{name} spatial shape mismatch: "
                f"{array.shape} vs {spatial_shape}"
            )
    else:
        raise ValueError(f"{name} should be 2D or 3D, got {array.shape}")

    return np.ascontiguousarray(array, dtype=np.float32)


def _min_max_normalize(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply per-band min-max normalization."""
    array = array.astype(np.float32, copy=False)
    array_min = array.min(axis=(0, 1), keepdims=True)
    array_max = array.max(axis=(0, 1), keepdims=True)
    return (array - array_min) / (array_max - array_min + eps)


def _random_per_class_split(
    gt: np.ndarray,
    num_classes: int,
    train_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly select a fixed number of training samples per class."""
    rng = np.random.default_rng(seed)
    train_label = np.zeros_like(gt)
    test_label = np.zeros_like(gt)

    for cls in range(1, num_classes + 1):
        coords = np.argwhere(gt == cls)
        if len(coords) < 2:
            raise ValueError(f"Class {cls} requires at least two samples.")

        n_train = min(train_per_class, len(coords) - 1)
        indices = rng.permutation(len(coords))

        train_coords = coords[indices[:n_train]]
        test_coords = coords[indices[n_train:]]

        train_label[train_coords[:, 0], train_coords[:, 1]] = cls
        test_label[test_coords[:, 0], test_coords[:, 1]] = cls

    return train_label, test_label


def _random_percent_split(
    gt: np.ndarray,
    num_classes: int,
    train_percent: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly select a percentage of training samples per class."""
    if not 0 < train_percent < 1:
        raise ValueError("train_percent should be in (0, 1).")

    rng = np.random.default_rng(seed)
    train_label = np.zeros_like(gt)
    test_label = np.zeros_like(gt)

    for cls in range(1, num_classes + 1):
        coords = np.argwhere(gt == cls)
        if len(coords) < 2:
            raise ValueError(f"Class {cls} requires at least two samples.")

        n_train = round(len(coords) * train_percent)
        n_train = max(1, min(n_train, len(coords) - 1))
        indices = rng.permutation(len(coords))

        train_coords = coords[indices[:n_train]]
        test_coords = coords[indices[n_train:]]

        train_label[train_coords[:, 0], train_coords[:, 1]] = cls
        test_label[test_coords[:, 0], test_coords[:, 1]] = cls

    return train_label, test_label


def _print_split_summary(
    train_label: np.ndarray,
    test_label: np.ndarray,
    num_classes: int,
) -> None:
    """Print the number of training and test samples."""
    print("Split summary:")
    print(f"  Train samples: {np.count_nonzero(train_label)}")
    print(f"  Test samples:  {np.count_nonzero(test_label)}")
    print("  Per-class train/test:")

    for cls in range(1, num_classes + 1):
        train_num = np.count_nonzero(train_label == cls)
        test_num = np.count_nonzero(test_label == cls)
        print(f"    Class {cls:02d}: {train_num:5d} / {test_num:5d}")


def load_data(args, project_root: Path, seed: int):
    """Load, normalize, and split an HSI--LiDAR dataset."""
    root = Path(args.dataset_root)
    if not root.is_absolute():
        root = project_root / root

    hsi_path = root / "hsi.mat"
    lidar_path = root / "lidar.mat"
    gt_path = root / "gt.mat"

    for path in (hsi_path, lidar_path, gt_path):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    print(f"Loading HSI:   {hsi_path}")
    print(f"Loading LiDAR: {lidar_path}")
    print(f"Loading GT:    {gt_path}")

    gt = _prepare_label_map(_load_mat_array(gt_path, "gt"))
    hsi = _ensure_hwc(_load_mat_array(hsi_path, "HSI"), gt.shape, "HSI")
    lidar = _ensure_hwc(_load_mat_array(lidar_path, "LiDAR"), gt.shape, "LiDAR")

    hsi = _min_max_normalize(hsi)
    lidar = _min_max_normalize(lidar)

    num_classes = int(gt.max())

    if args.protocol == "random_per_class":
        train_label, test_label = _random_per_class_split(
            gt,
            num_classes,
            args.train_per_class,
            seed,
        )
    elif args.protocol == "random_percent":
        train_label, test_label = _random_percent_split(
            gt,
            num_classes,
            args.train_percent,
            seed,
        )
    else:
        raise ValueError(f"Unknown protocol: {args.protocol}")

    return hsi, lidar, train_label, test_label, num_classes


def _pad_spatial(array: np.ndarray, radius: int, mode: str) -> np.ndarray:
    """Pad the spatial dimensions of an HWC array."""
    return np.pad(
        array,
        ((radius, radius), (radius, radius), (0, 0)),
        mode=mode,
    )


class HSILiDARDataset(Dataset):
    """Patch-based dataset for paired HSI and LiDAR samples."""

    def __init__(
        self,
        hsi: np.ndarray,
        lidar: np.ndarray,
        label_map: np.ndarray,
        patch_size: int,
        pad_mode: str = "symmetric",
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size should be a positive integer.")

        self.patch_size = patch_size
        self.label_map = label_map
        self.indices = np.argwhere(label_map > 0)

        radius = patch_size // 2
        self.hsi = _pad_spatial(hsi, radius, pad_mode)
        self.lidar = _pad_spatial(lidar, radius, pad_mode)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        row, col = self.indices[index]

        hsi_patch = self.hsi[
            row : row + self.patch_size,
            col : col + self.patch_size,
        ]
        lidar_patch = self.lidar[
            row : row + self.patch_size,
            col : col + self.patch_size,
        ]
        label = self.label_map[row, col] - 1

        hsi_patch = torch.from_numpy(
            np.ascontiguousarray(hsi_patch.transpose(2, 0, 1))
        ).float()
        lidar_patch = torch.from_numpy(
            np.ascontiguousarray(lidar_patch.transpose(2, 0, 1))
        ).float()
        label = torch.tensor(label, dtype=torch.long)

        return hsi_patch, lidar_patch, label


def get_dataloaders(args, project_root: Path, seed: int):
    """Build training and test data loaders."""
    hsi, lidar, train_label, test_label, num_classes = load_data(
        args,
        project_root,
        seed,
    )

    _print_split_summary(train_label, test_label, num_classes)

    train_set = HSILiDARDataset(
        hsi,
        lidar,
        train_label,
        args.patch_size,
        args.pad_mode,
    )
    test_set = HSILiDARDataset(
        hsi,
        lidar,
        test_label,
        args.patch_size,
        args.pad_mode,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return (
        train_loader,
        test_loader,
        hsi.shape[-1],
        lidar.shape[-1],
        num_classes,
    )
