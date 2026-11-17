"""CNN predictor: layout image -> several mobility indicators at once (multi-output regression).

Compared with two baselines:
  (a) always predict the training mean,
  (b) a linear model on road density only (one straight-line fit per indicator).

Run:  python -m src.predictor [--config configs/default.yaml] [--epochs 20]
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.build_labels import layout_indicators
from src.config import PROJECT_ROOT, get_device, load_config, set_seed
from src.dataset import LayoutDataset, SplitData, TargetScaler, load_splits
from src.evaluate import metrics_table


# --------------------------------------------------------------------------- model
class LayoutCNN(nn.Module):
    """Four conv blocks (conv 3x3 -> batch norm -> ReLU -> 2x2 max pool), then two linear layers."""

    def __init__(self, n_outputs: int, channels: tuple[int, ...] = (16, 32, 64, 64),
                 image_size: int = 64, hidden: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        layers, c_in = [], 1
        for c_out in channels:
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        side = image_size // 2 ** len(channels)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c_in * side * side, hidden),
                                  nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- training
def _run_epoch(model: nn.Module, loader: DataLoader, device: str,
               optimizer: torch.optim.Optimizer | None = None) -> float:
    """One pass over the loader. Trains if an optimizer is given. Returns the mean MSE loss."""
    model.train(optimizer is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.mse_loss(model(x), y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / n


def train_model(model: nn.Module, train_ds: LayoutDataset, val_ds: LayoutDataset,
                train_cfg: dict, device: str, seed: int) -> list[dict[str, float]]:
    """Train with Adam on the MSE of scaled targets. Keeps the weights of the epoch with the
    lowest validation loss. Returns the per-epoch history."""
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                              generator=gen, num_workers=train_cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=256, num_workers=train_cfg["num_workers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    model.to(device)
    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = _run_epoch(model, train_loader, device, optimizer)
        val_loss = _run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict(model: nn.Module, images: np.ndarray, scaler: TargetScaler, device: str,
            batch_size: int = 256) -> np.ndarray:
    """Predictions in the original indicator units."""
    model.eval().to(device)
    x_all = torch.from_numpy(images.astype(np.float32)).unsqueeze(1)
    out = [model(x_all[i:i + batch_size].to(device)).cpu().numpy()
           for i in range(0, len(x_all), batch_size)]
    return scaler.inverse(np.concatenate(out))


def load_predictor(path: Path, device: str = "cpu") -> tuple[LayoutCNN, TargetScaler, list[str]]:
    """Load a trained predictor checkpoint: model, target scaler, target names."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LayoutCNN(len(ckpt["targets"]), tuple(ckpt["channels"]), ckpt["image_size"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), TargetScaler.from_dict(ckpt["scaler"]), ckpt["targets"]


# --------------------------------------------------------------------------- baselines
def mean_baseline(y_train: np.ndarray, n: int) -> np.ndarray:
    """Baseline (a): the training mean of every indicator, for n samples."""
    return np.tile(y_train.mean(axis=0), (n, 1))


def fit_density_linear(density: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Baseline (b): least-squares fit y = a + b * density for every indicator.
    Returns coefficients of shape (2, T)."""
    design = np.stack([np.ones_like(density), density], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def predict_density_linear(coef: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Apply the density-only linear baseline."""
    return np.stack([np.ones_like(density), density], axis=1) @ coef


# --------------------------------------------------------------------------- latency
@torch.no_grad()
def measure_latency(model: nn.Module, image_size: int, n_warmup: int = 20,
                    n_runs: int = 200, batch_size: int = 256) -> dict[str, float]:
    """CPU inference time per image: single-image latency (mean, p95) after warm-up,
    and per-image time when processing a batch."""
    model = copy.deepcopy(model).cpu().eval()
    x1 = torch.zeros(1, 1, image_size, image_size)
    for _ in range(n_warmup):
        model(x1)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model(x1)
        times.append((time.perf_counter() - start) * 1000)
    xb = torch.zeros(batch_size, 1, image_size, image_size)
    model(xb)
    start = time.perf_counter()
    model(xb)
    batch_ms = (time.perf_counter() - start) * 1000 / batch_size
    return {"single_mean_ms": float(np.mean(times)), "single_p95_ms": float(np.percentile(times, 95)),
            "batched_ms_per_image": batch_ms, "torch_threads": torch.get_num_threads()}


# --------------------------------------------------------------------------- plots
def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, styles: pd.Series,
                 targets: list[str], metrics: pd.DataFrame, path: Path) -> None:
    """Predicted vs true value for every indicator, coloured by layout style."""
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4.2))
    for j, (ax, target) in enumerate(zip(axes, targets)):
        for style in sorted(styles.unique()):
            m = (styles == style).to_numpy()
            ax.scatter(y_true[m, j], y_pred[m, j], s=5, alpha=0.6, label=style)
        lo, hi = y_true[:, j].min(), y_true[:, j].max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        r2 = metrics.query("model == 'cnn' and indicator == @target")["r2"].iloc[0]
        ax.set_title(f"{target}\nCNN test R² = {r2:.3f}", fontsize=9)
        ax.set_xlabel("true (graph method)")
        ax.set_ylabel("predicted (CNN)")
    axes[0].legend(fontsize=7, markerscale=2)
    fig.suptitle("CNN predictions on the test set (synthetic layouts; congestion from a simplified "
                 "all-or-nothing assignment)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    """Training and validation loss per epoch."""
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(h["epoch"], h["train_loss"], label="train")
    ax.plot(h["epoch"], h["val_loss"], label="validation")
    ax.set_xlabel("epoch"), ax.set_ylabel("MSE (scaled targets)"), ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- label noise
def label_repeatability(split: SplitData, cfg: dict, targets: list[str],
                        alt_seed: int = 999) -> pd.DataFrame:
    """Recompute the labels of a split with another random seed (other random trips and
    source nodes) and compare with the stored labels. The R² obtained is a ceiling for
    any predictor: the random part of a label cannot be predicted from the image."""
    redo = np.array([[layout_indicators(img, alt_seed, int(i), cfg)[t] for t in targets]
                     for img, i in zip(split.images, split.meta["id"])])
    return metrics_table(split.targets, {"same_layout_other_seed": redo}, targets)


# --------------------------------------------------------------------------- main
def evaluate_all(model: nn.Module, data: dict[str, SplitData], scaler: TargetScaler,
                 targets: list[str], device: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Test metrics of the CNN and of both baselines. Also returns the CNN test predictions."""
    train, test = data["train"], data["test"]
    density_train = train.images.reshape(len(train.images), -1).mean(axis=1)
    density_test = test.images.reshape(len(test.images), -1).mean(axis=1)
    coef = fit_density_linear(density_train, train.targets)
    cnn_pred = predict(model, test.images, scaler, device)
    preds = {"mean_baseline": mean_baseline(train.targets, len(test.targets)),
             "density_linear": predict_density_linear(coef, density_test),
             "cnn": cnn_pred}
    return metrics_table(test.targets, preds, targets), cnn_pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    set_seed(cfg["seed"])
    targets, device = cfg["predictor"]["targets"], get_device()
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]
    models_dir.mkdir(exist_ok=True)

    data = load_splits(cfg, targets)
    scaler = TargetScaler.fit(data["train"].targets)
    train_ds = LayoutDataset(data["train"].images, scaler.transform(data["train"].targets), augment=True)
    val_ds = LayoutDataset(data["val"].images, scaler.transform(data["val"].targets))
    model = LayoutCNN(len(targets), tuple(cfg["predictor"]["channels"]), cfg["dataset"]["image_size"])
    print(f"Device: {device}. Parameters: {count_parameters(model):,}")

    start = time.perf_counter()
    history = train_model(model, train_ds, val_ds, cfg["training"], device, cfg["seed"])
    train_seconds = time.perf_counter() - start
    torch.save({"state_dict": model.state_dict(), "scaler": scaler.to_dict(), "targets": targets,
                "channels": cfg["predictor"]["channels"], "image_size": cfg["dataset"]["image_size"]},
               models_dir / "predictor.pt")

    metrics, cnn_pred = evaluate_all(model, data, scaler, targets, device)
    metrics.to_csv(results_dir / "predictor_metrics.csv", index=False)
    by_style = pd.concat([metrics_table(data["test"].targets[m], {"cnn": cnn_pred[m]}, targets)
                          .assign(style=s) for s in sorted(data["test"].meta["style"].unique())
                          for m in [(data["test"].meta["style"] == s).to_numpy()]])
    by_style.to_csv(results_dir / "predictor_metrics_by_style.csv", index=False)
    latency = measure_latency(model, cfg["dataset"]["image_size"])
    latency.update({"device": "cpu", "parameters": count_parameters(model),
                    "train_seconds": round(train_seconds, 1), "train_device": device})
    pd.DataFrame([latency]).to_csv(results_dir / "predictor_latency.csv", index=False)
    plot_scatter(data["test"].targets, cnn_pred, data["test"].meta["style"], targets, metrics,
                 results_dir / "predictor_scatter.png")
    plot_history(history, results_dir / "predictor_training.png")
    repeat = label_repeatability(data["test"], cfg, targets)
    repeat.to_csv(results_dir / "label_repeatability.csv", index=False)

    print(f"\nTraining time: {train_seconds:.1f} s on {device}")
    print(metrics.round(4).to_string(index=False))
    print("\nCNN R² per style:")
    print(by_style.pivot(index="indicator", columns="style", values="r2").round(3).to_string())
    print("\nLabel repeatability (same test layouts, other random seed) = R² ceiling:")
    print(repeat.round(4).to_string(index=False))
    print("\nLatency (CPU):", {k: round(v, 3) if isinstance(v, float) else v for k, v in latency.items()})


if __name__ == "__main__":
    main()
