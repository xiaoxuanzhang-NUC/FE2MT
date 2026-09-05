"""Training script for FE2MT."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

from data_loader import get_dataloaders
from fe2mt import FE2MT


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute OA, AA, and Cohen's kappa."""
    return {
        "OA": float(accuracy_score(y_true, y_pred)),
        "AA": float(balanced_accuracy_score(y_true, y_pred)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def train_one_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Train the model for one epoch."""
    model.train()

    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []

    for hsi, lidar, labels in data_loader:
        hsi = hsi.to(device)
        lidar = lidar.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(hsi, lidar)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        y_true.append(labels.detach().cpu().numpy())
        y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())

    metrics = compute_metrics(np.concatenate(y_true), np.concatenate(y_pred))
    return total_loss / total_samples, metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Evaluate the model on a data loader."""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []

    for hsi, lidar, labels in data_loader:
        hsi = hsi.to(device)
        lidar = lidar.to(device)
        labels = labels.to(device)

        logits = model(hsi, lidar)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        y_true.append(labels.cpu().numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())

    metrics = compute_metrics(np.concatenate(y_true), np.concatenate(y_pred))
    return total_loss / total_samples, metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train FE2MT.")

    parser.add_argument("--dataset_root", type=str, default="data/Houston2013")
    parser.add_argument(
        "--protocol",
        type=str,
        default="random_per_class",
        choices=["random_per_class", "random_percent"],
    )
    parser.add_argument("--train_per_class", type=int, default=20)
    parser.add_argument("--train_percent", type=float, default=0.2)

    parser.add_argument("--patch_size", type=int, default=12)
    parser.add_argument("--pad_mode", type=str, default="symmetric")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    dataset_name = Path(args.dataset_root).name

    output_dir = project_root / "outputs" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Dataset: {dataset_name}")
    print(f"Device:  {device}")

    run_metrics = []
    best_epochs = []

    for run_idx in range(args.runs):
        run_number = run_idx + 1
        run_seed = args.seed + run_idx
        set_seed(run_seed)

        log_path = output_dir / f"training_log_run{run_number}.txt"
        checkpoint_path = output_dir / f"best_model_run{run_number}.pth"

        print("=" * 72)
        print(
            f"Run {run_number}/{args.runs} | "
            f"protocol={args.protocol} | seed={run_seed}"
        )
        print("=" * 72)

        (
            train_loader,
            test_loader,
            hsi_channels,
            lidar_channels,
            num_classes,
        ) = get_dataloaders(args, project_root, run_seed)

        model_config = {
            "hsi_channels": hsi_channels,
            "lidar_channels": lidar_channels,
            "num_classes": num_classes,
            "patch_size": args.patch_size,
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
        }

        model = FE2MT(**model_config).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_oa = -1.0
        best_epoch = 0
        best_metrics = None

        with open(log_path, "w", encoding="utf-8") as log_file:
            for epoch in range(1, args.epochs + 1):
                train_loss, train_metrics = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                )
                test_loss, test_metrics = evaluate(
                    model,
                    test_loader,
                    criterion,
                    device,
                )

                if test_metrics["OA"] > best_oa:
                    best_oa = test_metrics["OA"]
                    best_epoch = epoch
                    best_metrics = test_metrics.copy()

                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "model_config": model_config,
                            "epoch": best_epoch,
                            "metrics": best_metrics,
                            "dataset": dataset_name,
                            "seed": run_seed,
                        },
                        checkpoint_path,
                    )

                log = (
                    f"Epoch {epoch:03d}/{args.epochs} | "
                    f"Train loss {train_loss:.4f} | "
                    f"OA {train_metrics['OA'] * 100:.2f} | "
                    f"AA {train_metrics['AA'] * 100:.2f} | "
                    f"Kappa {train_metrics['Kappa'] * 100:.2f} || "
                    f"Test loss {test_loss:.4f} | "
                    f"OA {test_metrics['OA'] * 100:.2f} | "
                    f"AA {test_metrics['AA'] * 100:.2f} | "
                    f"Kappa {test_metrics['Kappa'] * 100:.2f} | "
                    f"Best OA {best_oa * 100:.2f}@{best_epoch}"
                )

                print(log)
                log_file.write(log + "\n")

            best_log = (
                f"Best result | Epoch {best_epoch} | "
                f"OA {best_metrics['OA'] * 100:.2f} | "
                f"AA {best_metrics['AA'] * 100:.2f} | "
                f"Kappa {best_metrics['Kappa'] * 100:.2f}"
            )
            print(best_log)
            log_file.write(best_log + "\n")

        run_metrics.append(best_metrics)
        best_epochs.append(best_epoch)

    oa = np.array([metrics["OA"] for metrics in run_metrics])
    aa = np.array([metrics["AA"] for metrics in run_metrics])
    kappa = np.array([metrics["Kappa"] for metrics in run_metrics])
    best_run = int(np.argmax(oa)) + 1

    final_lines = [
        f"Dataset: {dataset_name}",
        f"Runs: {args.runs}",
        f"Best epochs: {best_epochs}",
    ]

    for idx, metrics in enumerate(run_metrics, start=1):
        final_lines.append(
            f"Run {idx}: "
            f"OA {metrics['OA'] * 100:.2f} | "
            f"AA {metrics['AA'] * 100:.2f} | "
            f"Kappa {metrics['Kappa'] * 100:.2f} | "
            f"Best epoch {best_epochs[idx - 1]}"
        )

    final_lines.extend(
        [
            f"Best run by OA: {best_run}",
            f"OA:    {oa.mean() * 100:.2f} ± {oa.std() * 100:.2f}",
            f"AA:    {aa.mean() * 100:.2f} ± {aa.std() * 100:.2f}",
            f"Kappa: {kappa.mean() * 100:.2f} ± {kappa.std() * 100:.2f}",
        ]
    )

    print("=" * 72)
    print(f"Final results over {args.runs} run(s)")
    for line in final_lines[2:]:
        print(line)

    results_path = output_dir / "results.txt"
    with open(results_path, "w", encoding="utf-8") as results_file:
        results_file.write("\n".join(final_lines) + "\n")


if __name__ == "__main__":
    main()
