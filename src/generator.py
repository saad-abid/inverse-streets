"""Conditional VAE: target indicators -> new layout image.

Idea in short:
- The encoder sees a layout AND its indicators and compresses the layout into a small latent
  vector z. Because the indicators are given separately, z only has to store the rest
  (style, orientation, spacing, ...).
- The decoder rebuilds the layout from z AND the indicators.
- A KL term pushes z towards a standard normal distribution, so after training we can draw
  z ~ N(0, I), choose target indicators, and decode new layouts. Different z -> different
  layouts for the same target.

Generated images are probabilities; post-processing makes them binary and removes tiny
artefacts before indicators are recomputed with the graph method.

Run:  python -m src.generator [--config configs/default.yaml] [--epochs 30]
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
from skimage.morphology import remove_small_holes, remove_small_objects
from torch import nn
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, get_device, load_config, set_seed
from src.dataset import LayoutDataset, SplitData, TargetScaler, load_splits
from src.graph_metrics import graph_indicators, road_graph


# --------------------------------------------------------------------------- model
class ConditionalVAE(nn.Module):
    """Convolutional encoder/decoder; the condition vector is concatenated to the flattened
    encoder features and to the latent vector before decoding."""

    def __init__(self, n_cond: int, latent_dim: int = 32,
                 channels: tuple[int, ...] = (32, 64, 128, 128), image_size: int = 64,
                 hidden: int = 256) -> None:
        super().__init__()
        self.latent_dim, self.channels = latent_dim, tuple(channels)
        self.side = image_size // 2 ** len(channels)
        flat = channels[-1] * self.side * self.side

        enc, c_in = [], 1
        for c_out in channels:  # each block halves the image size
            enc += [nn.Conv2d(c_in, c_out, 4, stride=2, padding=1), nn.BatchNorm2d(c_out),
                    nn.LeakyReLU(0.2, inplace=True)]
            c_in = c_out
        self.encoder = nn.Sequential(*enc, nn.Flatten())
        self.enc_fc = nn.Sequential(nn.Linear(flat + n_cond, hidden), nn.LeakyReLU(0.2, inplace=True))
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

        self.dec_fc = nn.Sequential(nn.Linear(latent_dim + n_cond, flat), nn.ReLU(inplace=True))
        dec, rev = [], list(reversed(channels))
        for c_in, c_out in zip(rev, rev[1:]):  # each block doubles the image size
            dec += [nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1),
                    nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
        dec.append(nn.ConvTranspose2d(rev[-1], 1, 4, stride=2, padding=1))  # logits
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_fc(torch.cat([self.encoder(x), cond], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(torch.cat([z, cond], dim=1))
        return self.decoder(h.view(-1, self.channels[-1], self.side, self.side))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        mu, logvar = self.encode(x, cond)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # reparameterization trick
        return self.decode(z, cond), mu, logvar


def vae_loss(logits: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
             beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample loss = reconstruction (pixel BCE, summed) + beta * KL(q(z|x,c) || N(0, I))."""
    n = len(x)
    recon = nn.functional.binary_cross_entropy_with_logits(logits, x, reduction="sum") / n
    kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / n
    return recon + beta * kl, recon, kl


def beta_schedule(epoch: int, beta: float, warmup_epochs: int) -> float:
    """KL weight rises linearly from 0 to beta over the first warm-up epochs (epoch starts at 1).
    This helps the model use z instead of ignoring it."""
    if warmup_epochs <= 0:
        return beta
    return beta * min(1.0, epoch / warmup_epochs)


# --------------------------------------------------------------------------- training
def _run_epoch(model: ConditionalVAE, loader: DataLoader, device: str, beta: float,
               optimizer: torch.optim.Optimizer | None = None) -> dict[str, float]:
    """One pass over the loader (trains if an optimizer is given). Returns mean losses."""
    model.train(optimizer is not None)
    sums, n = np.zeros(3), 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, c in loader:
            x, c = x.to(device), c.to(device)
            logits, mu, logvar = model(x, c)
            loss, recon, kl = vae_loss(logits, x, mu, logvar, beta)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            sums += np.array([loss.item(), recon.item(), kl.item()]) * len(x)
            n += len(x)
    return dict(zip(("loss", "recon", "kl"), sums / n))


def train_generator(model: ConditionalVAE, train_ds: LayoutDataset, val_ds: LayoutDataset,
                    gen_cfg: dict, device: str, seed: int) -> list[dict[str, float]]:
    """Train with Adam and KL warm-up. Keeps the weights of the epoch with the lowest
    validation loss (computed with the full beta). Returns the per-epoch history."""
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=gen_cfg["batch_size"], shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=gen_cfg["learning_rate"])
    model.to(device)
    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(1, gen_cfg["epochs"] + 1):
        beta = beta_schedule(epoch, gen_cfg["beta"], gen_cfg["kl_warmup_epochs"])
        tr = _run_epoch(model, train_loader, device, beta, optimizer)
        va = _run_epoch(model, val_loader, device, gen_cfg["beta"])
        history.append({"epoch": epoch, "beta": beta, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        print(f"epoch {epoch:3d}  beta {beta:.2f}  train recon {tr['recon']:.1f} kl {tr['kl']:.1f}"
              f"  |  val loss {va['loss']:.1f} (recon {va['recon']:.1f}, kl {va['kl']:.1f})")
        if epoch >= gen_cfg["kl_warmup_epochs"] and va["loss"] < best_loss:
            best_loss, best_state = va["loss"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state or model.state_dict())
    return history


# --------------------------------------------------------------------------- sampling
@torch.no_grad()
def sample_probabilities(model: ConditionalVAE, target: np.ndarray, scaler: TargetScaler,
                         n: int, seed: int, device: str = "cpu") -> np.ndarray:
    """Decode n random latent vectors for one target (raw indicator values).
    Returns road probabilities of shape (n, H, W). Same seed -> same samples."""
    model.eval().to(device)
    z = torch.randn(n, model.latent_dim, generator=torch.Generator().manual_seed(seed))
    cond = torch.from_numpy(scaler.transform(np.asarray(target, float)[None])).repeat(n, 1)
    return torch.sigmoid(model.decode(z.to(device), cond.to(device))).cpu().numpy()[:, 0]


def postprocess(probs: np.ndarray, threshold: float = 0.5, max_fragment: int = 10,
                max_hole: int = 4) -> np.ndarray:
    """Binarize, then remove road fragments of <= max_fragment pixels and fill holes of
    <= max_hole pixels. Works on one image (H, W) or a batch (N, H, W)."""
    if probs.ndim == 3:
        return np.stack([postprocess(p, threshold, max_fragment, max_hole) for p in probs])
    roads = probs > threshold
    roads = remove_small_objects(roads, max_size=max_fragment, connectivity=2)
    roads = remove_small_holes(roads, max_size=max_hole)
    return roads.astype(np.uint8)


def generate_layouts(model: ConditionalVAE, target: np.ndarray, scaler: TargetScaler, n: int,
                     seed: int, gen_cfg: dict, device: str = "cpu") -> np.ndarray:
    """Binary, cleaned layouts (n, H, W) for one target."""
    probs = sample_probabilities(model, target, scaler, n, seed, device)
    return postprocess(probs, gen_cfg["threshold"], gen_cfg["max_fragment_size"], gen_cfg["max_hole_size"])


def calibrate_threshold(model: ConditionalVAE, scaler: TargetScaler, val: SplitData,
                        conditions: list[str], candidates: list[float], seed: int,
                        n_targets: int = 200) -> tuple[float, pd.DataFrame]:
    """Pick the binarization threshold on VALIDATION targets: the candidate whose binary
    road density is closest (mean absolute error) to the target density.
    Thin roads get probabilities below 0.5 where the model is unsure of their exact
    position, so a fixed 0.5 threshold breaks them."""
    j = conditions.index("road_density")
    n = min(n_targets, len(val.targets))
    probs = [sample_probabilities(model, val.targets[k], scaler, 1, seed + k)[0] for k in range(n)]
    rows = [{"threshold": t, "density_mae": float(np.mean(
        [abs((p > t).mean() - val.targets[k][j]) for k, p in enumerate(probs)]))} for t in candidates]
    table = pd.DataFrame(rows)
    return float(table.loc[table["density_mae"].idxmin(), "threshold"]), table


def load_generator(path: Path, device: str = "cpu") -> tuple[ConditionalVAE, TargetScaler, list[str], float]:
    """Load a trained generator checkpoint: model, condition scaler, condition names and the
    binarization threshold (calibrated on validation data if available, else 0.5)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ConditionalVAE(len(ckpt["conditions"]), ckpt["latent_dim"], tuple(ckpt["channels"]),
                           ckpt["image_size"])
    model.load_state_dict(ckpt["state_dict"])
    return (model.eval(), TargetScaler.from_dict(ckpt["scaler"]), ckpt["conditions"],
            float(ckpt.get("threshold", 0.5)))


# --------------------------------------------------------------------------- figures
def recomputed(img: np.ndarray, cfg: dict) -> dict[str, float]:
    """Indicators of a generated image with the ground-truth graph method."""
    return graph_indicators(img, road_graph(img, cfg["graph"]), np.random.default_rng(0), cfg["graph"])


def plot_generated(model: ConditionalVAE, scaler: TargetScaler, test: SplitData,
                   conditions: list[str], cfg: dict, path: Path, n_per_target: int = 6) -> None:
    """For one real test layout per style: the real layout, then layouts generated for its
    indicator values, each titled with the indicators recomputed by the graph method."""
    styles = sorted(test.meta["style"].unique())
    fig, axes = plt.subplots(len(styles), n_per_target + 1,
                             figsize=(1.9 * (n_per_target + 1), 2.2 * len(styles)))
    for r, style in enumerate(styles):
        k = int(np.flatnonzero((test.meta["style"] == style).to_numpy())[0])
        target = test.targets[k]
        images = generate_layouts(model, target, scaler, n_per_target, cfg["seed"] + r,
                                  cfg["generator"])
        panels = [(test.images[k], f"REAL #{test.meta['id'][k]}")] + \
                 [(img, "generated") for img in images]
        for c, (img, label) in enumerate(panels):
            ind = recomputed(img, cfg)
            ax = axes[r, c]
            ax.imshow(img, cmap="gray_r", interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{label}\nd={ind['road_density']:.2f} int={ind['n_intersections']}",
                         fontsize=7, color="black" if c == 0 else "tab:blue")
        target_txt = ", ".join(f"{n}={v:.2f}" if v < 1 else f"{n}={v:.0f}"
                               for n, v in zip(conditions, target))
        axes[r, 0].set_ylabel(f"target from\n{style}\n({target_txt})", fontsize=7)
    fig.suptitle("cVAE samples for the indicator values of real test layouts "
                 "(d = road density, int = intersections, recomputed with the graph method)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    """Reconstruction and KL terms per epoch."""
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key in zip(axes, ("recon", "kl")):
        ax.plot(h["epoch"], h[f"train_{key}"], label="train")
        ax.plot(h["epoch"], h[f"val_{key}"], label="validation")
        ax.set_title({"recon": "reconstruction (BCE per image)", "kl": "KL per image"}[key])
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def quick_check(model: ConditionalVAE, scaler: TargetScaler, test: SplitData, conditions: list[str],
                cfg: dict, n_targets: int = 100) -> pd.DataFrame:
    """One generated layout for each of the first n test targets, with recomputed indicators.
    (The full evaluation is Phase 5.)"""
    rows = []
    for k in range(min(n_targets, len(test.targets))):
        img = generate_layouts(model, test.targets[k], scaler, 1, cfg["seed"] + 1000 + k, cfg["generator"])[0]
        ind = recomputed(img, cfg)
        rows.append({**{f"target_{c}": test.targets[k][j] for j, c in enumerate(conditions)},
                     **{f"generated_{c}": ind[c] for c in conditions},
                     "connected": ind["connectivity"] == 1.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main
def _train_and_save(cfg: dict, data: dict[str, SplitData], scaler: TargetScaler, device: str,
                    path: Path, results_dir: Path) -> ConditionalVAE:
    """Train a new cVAE, save the checkpoint, the history CSV and the loss plot."""
    gen_cfg, conditions = cfg["generator"], cfg["generator"]["conditions"]
    train_ds = LayoutDataset(data["train"].images, scaler.transform(data["train"].targets), augment=True)
    val_ds = LayoutDataset(data["val"].images, scaler.transform(data["val"].targets))
    model = ConditionalVAE(len(conditions), gen_cfg["latent_dim"], tuple(gen_cfg["channels"]),
                           cfg["dataset"]["image_size"])
    print(f"Device: {device}. Conditions: {conditions}. "
          f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    start = time.perf_counter()
    history = train_generator(model, train_ds, val_ds, gen_cfg, device, cfg["seed"])
    print(f"Training time: {time.perf_counter() - start:.1f} s on {device}")
    torch.save({"state_dict": model.state_dict(), "scaler": scaler.to_dict(),
                "conditions": conditions, "latent_dim": gen_cfg["latent_dim"],
                "channels": gen_cfg["channels"], "image_size": cfg["dataset"]["image_size"]}, path)
    pd.DataFrame(history).to_csv(results_dir / "generator_training.csv", index=False)
    plot_history(history, results_dir / "generator_training.png")
    return model.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--from-checkpoint", action="store_true",
                        help="skip training; reuse models/generator.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gen_cfg = cfg["generator"]
    if args.epochs:
        gen_cfg["epochs"] = args.epochs
    set_seed(cfg["seed"])
    conditions, device = gen_cfg["conditions"], get_device()
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    path = PROJECT_ROOT / cfg["paths"]["models_dir"] / "generator.pt"
    path.parent.mkdir(exist_ok=True)

    data = load_splits(cfg, conditions)
    if args.from_checkpoint:
        model, scaler, conditions, _ = load_generator(path)
    else:
        scaler = TargetScaler.fit(data["train"].targets)
        model = _train_and_save(cfg, data, scaler, device, path, results_dir)

    if gen_cfg["calibrate_threshold"] and "road_density" in conditions:
        threshold, table = calibrate_threshold(model, scaler, data["val"], conditions,
                                               gen_cfg["threshold_candidates"], cfg["seed"] + 5000)
        table.to_csv(results_dir / "generator_threshold_calibration.csv", index=False)
        gen_cfg["threshold"] = threshold
        ckpt = torch.load(path, weights_only=False)
        ckpt["threshold"] = threshold
        torch.save(ckpt, path)
        print("Threshold calibration on validation targets:")
        print(table.round(4).to_string(index=False))
        print(f"-> threshold {threshold}")

    plot_generated(model, scaler, data["test"], conditions, cfg, results_dir / "generated_examples.png")
    check = quick_check(model, scaler, data["test"], conditions, cfg)
    summary = pd.DataFrame([{"threshold": gen_cfg["threshold"], "n_targets": len(check),
                             **{f"{c}_mae": float((check[f"generated_{c}"] - check[f"target_{c}"]).abs().mean())
                                for c in conditions},
                             "share_connected": float(check["connected"].mean())}])
    summary.to_csv(results_dir / "generator_quickcheck.csv", index=False)
    print("\nQuick check (100 test targets, 1 sample each; full evaluation in Phase 5):")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
