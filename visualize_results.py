"""Visualize whole-scene FE2MT classification results."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.io import loadmat
from torch.utils.data import DataLoader

from data_loader import HSILiDARDataset
from fe2mt import FE2MT


def load_mat_array(path: Path, key: str) -> np.ndarray:
    """Load an array from a MATLAB file."""
    data = loadmat(path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}")
    return data[key]


def prepare_label_map(array: np.ndarray) -> np.ndarray:
    """Convert a label map to shape [H, W]."""
    array = np.squeeze(np.asarray(array))
    if array.ndim != 2:
        raise ValueError(f"Label map should be 2D, got {array.shape}")
    return array.astype(np.int64, copy=False)


def ensure_hwc(
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


def min_max_normalize(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply per-band min-max normalization."""
    array = array.astype(np.float32, copy=False)
    array_min = array.min(axis=(0, 1), keepdims=True)
    array_max = array.max(axis=(0, 1), keepdims=True)
    return (array - array_min) / (array_max - array_min + eps)


def load_scene(dataset_root: Path):
    """Load and normalize HSI, LiDAR, and ground-truth data."""
    hsi_path = dataset_root / "hsi.mat"
    lidar_path = dataset_root / "lidar.mat"
    gt_path = dataset_root / "gt.mat"

    for path in (hsi_path, lidar_path, gt_path):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    gt = prepare_label_map(load_mat_array(gt_path, "gt"))
    hsi = ensure_hwc(load_mat_array(hsi_path, "HSI"), gt.shape, "HSI")
    lidar = ensure_hwc(load_mat_array(lidar_path, "LiDAR"), gt.shape, "LiDAR")

    hsi = min_max_normalize(hsi)
    lidar = min_max_normalize(lidar)

    return hsi, lidar, gt


def find_best_run(results_path: Path) -> int:
    """Read the run with the highest OA from results.txt."""
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Specify --run explicitly."
        )

    text = results_path.read_text(encoding="utf-8")
    match = re.search(r"Best run by OA:\s*(\d+)", text)
    if match is None:
        raise ValueError(
            "Best run information was not found in results.txt. "
            "Specify --run explicitly."
        )

    return int(match.group(1))


@torch.no_grad()
def predict_scene(
    model: FE2MT,
    hsi: np.ndarray,
    lidar: np.ndarray,
    gt: np.ndarray,
    patch_size: int,
    pad_mode: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    """Predict all labeled pixels and reconstruct a classification map."""
    dataset = HSILiDARDataset(
        hsi,
        lidar,
        gt,
        patch_size=patch_size,
        pad_mode=pad_mode,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    prediction_map = np.zeros_like(gt, dtype=np.int64)
    offset = 0

    model.eval()

    for hsi_patch, lidar_patch, _ in data_loader:
        hsi_patch = hsi_patch.to(device)
        lidar_patch = lidar_patch.to(device)

        prediction = model(hsi_patch, lidar_patch).argmax(dim=1)
        prediction = prediction.cpu().numpy() + 1

        coords = dataset.indices[offset : offset + len(prediction)]
        prediction_map[coords[:, 0], coords[:, 1]] = prediction
        offset += len(prediction)

    return prediction_map


def make_pseudo_rgb(
    hsi: np.ndarray,
    bands: list[int] | None,
) -> np.ndarray:
    """Create a pseudo-RGB image from three HSI bands."""
    num_bands = hsi.shape[-1]

    if bands is None:
        bands = [
            round((num_bands - 1) * 0.75),
            round((num_bands - 1) * 0.50),
            round((num_bands - 1) * 0.25),
        ]

    if len(bands) != 3 or any(band < 0 or band >= num_bands for band in bands):
        raise ValueError(
            f"RGB band indices must contain three values in [0, {num_bands - 1}]."
        )

    rgb = hsi[..., bands].astype(np.float32, copy=True)

    for channel in range(3):
        values = rgb[..., channel]
        low, high = np.percentile(values, [1, 99])

        if high > low:
            values = (values - low) / (high - low)

        rgb[..., channel] = np.clip(values, 0.0, 1.0)

    return rgb


def build_label_colormap(num_classes: int):
    """Create a shared colormap for ground truth and predictions."""
    base = plt.get_cmap("tab20")
    colors = [(0.0, 0.0, 0.0, 1.0)]

    for index in range(num_classes):
        colors.append(base(index % 20))

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        np.arange(-0.5, num_classes + 1.5, 1.0),
        cmap.N,
    )
    return cmap, norm


def default_output_path(project_root: Path, dataset_name: str) -> Path:
    """Return the README-compatible result image path."""
    names = {
        "houston2013": "houston_result.jpg",
        "muufl": "muufl_result.jpg",
        "trento": "trento_result.jpg",
    }
    filename = names.get(
        dataset_name.lower(),
        f"{dataset_name.lower()}_result.jpg",
    )
    return project_root / "figures" / filename


def save_result_figure(
    hsi: np.ndarray,
    gt: np.ndarray,
    prediction_map: np.ndarray,
    output_path: Path,
    rgb_bands: list[int] | None,
) -> None:
    """Save HSI, ground-truth, and FE2MT prediction maps."""
    num_classes = int(gt.max())
    rgb = make_pseudo_rgb(hsi, rgb_bands)
    cmap, norm = build_label_colormap(num_classes)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("HSI")

    axes[1].imshow(gt, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(
        prediction_map,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    axes[2].set_title("FE2MT")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout(pad=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"quality": 95},
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize FE2MT whole-scene classification results."
    )

    parser.add_argument("--dataset_root", type=str, default="data/Houston2013")
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Run number to visualize. Defaults to the run with the highest OA.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pad_mode", type=str, default="symmetric")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--rgb_bands",
        type=int,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Optional zero-based HSI band indices for pseudo-RGB visualization.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    dataset_name = dataset_root.name
    output_dir = project_root / "outputs" / dataset_name

    run_number = args.run
    if run_number is None:
        run_number = find_best_run(output_dir / "results.txt")

    checkpoint_path = output_dir / f"best_model_run{run_number}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model_config = checkpoint["model_config"]
    model = FE2MT(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    hsi, lidar, gt = load_scene(dataset_root)

    if hsi.shape[-1] != model_config["hsi_channels"]:
        raise ValueError("HSI channel count does not match the checkpoint.")
    if lidar.shape[-1] != model_config["lidar_channels"]:
        raise ValueError("LiDAR channel count does not match the checkpoint.")
    if int(gt.max()) != model_config["num_classes"]:
        raise ValueError("Number of classes does not match the checkpoint.")

    print(f"Dataset:    {dataset_name}")
    print(f"Run:        {run_number}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best OA:    {checkpoint['metrics']['OA'] * 100:.2f}")

    prediction_map = predict_scene(
        model=model,
        hsi=hsi,
        lidar=lidar,
        gt=gt,
        patch_size=model_config["patch_size"],
        pad_mode=args.pad_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    if args.output is None:
        output_path = default_output_path(project_root, dataset_name)
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path

    save_result_figure(
        hsi=hsi,
        gt=gt,
        prediction_map=prediction_map,
        output_path=output_path,
        rgb_bands=args.rgb_bands,
    )

    print(f"Saved:      {output_path}")


if __name__ == "__main__":
    main()
