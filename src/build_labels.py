"""Compute mobility indicators for every layout and save them to data/labels.csv.

Indicators come from the road graph (graph_metrics.py) and from a simplified
all-or-nothing traffic assignment (traffic.py). Traffic numbers are NOT the output of a
calibrated traffic simulation.

Run:  python -m src.build_labels [--config configs/default.yaml] [--n 200] [--workers 4]
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.graph_metrics import graph_indicators, road_graph
from src.traffic import edge_key, traffic_indicators

INDICATORS = ["road_density", "n_intersections", "n_dead_ends", "connectivity",
              "avg_shortest_path", "detour_index", "avg_trip_length",
              "congestion_index", "share_congested"]


def layout_indicators(img: np.ndarray, seed: int, index: int, cfg: dict) -> dict[str, float]:
    """All indicators for one layout. The random parts are seeded with (seed, index)."""
    rng = np.random.default_rng([seed, index, 2])
    g = road_graph(img, cfg["graph"])
    traffic, _ = traffic_indicators(img, g, rng, cfg["traffic"])
    return {**graph_indicators(img, g, rng, cfg["graph"]), **traffic}


def _worker(args: tuple[np.ndarray, int, int, dict]) -> dict[str, float]:
    return layout_indicators(*args)


def compute_all(images: np.ndarray, cfg: dict, workers: int) -> pd.DataFrame:
    """Indicators for every image, computed in parallel processes."""
    jobs = [(img, cfg["seed"], i, cfg) for i, img in enumerate(images)]
    if workers <= 1:
        rows = [_worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_worker, jobs, chunksize=50))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plots
def plot_distributions(labels: pd.DataFrame, path: Path) -> None:
    """Histogram of every indicator, one colour per layout style."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, col in zip(axes.flat, INDICATORS):
        data = [labels.loc[labels["style"] == s, col].dropna() for s in sorted(labels["style"].unique())]
        ax.hist(data, bins=30, stacked=True, label=sorted(labels["style"].unique()))
        ax.set_title(col, fontsize=10)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Indicator distributions (synthetic layouts; traffic = simplified all-or-nothing "
                 "assignment, not calibrated)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_edge_loads(images: np.ndarray, labels: pd.DataFrame, cfg: dict, path: Path) -> None:
    """Edge-load heatmap and detected nodes for one test layout of each style."""
    styles = sorted(labels["style"].unique())
    fig, axes = plt.subplots(1, len(styles), figsize=(4.2 * len(styles), 4.4))
    for ax, style in zip(axes, styles):
        i = int(labels.loc[(labels["style"] == style) & (labels["split"] == "test"), "id"].iloc[0])
        img = images[i]
        g = road_graph(img, cfg["graph"])
        _, loads = traffic_indicators(img, g, np.random.default_rng([cfg["seed"], i, 2]), cfg["traffic"])
        ax.imshow(img, cmap="Greys", vmax=4, interpolation="nearest")
        vmax = max(loads.values()) if loads else 1
        for u, v, data in g.edges(data=True):
            px = np.array(data["pixels"])
            sc = ax.scatter(px[:, 1], px[:, 0], c=[loads.get(edge_key(u, v), 0)] * len(px),
                            cmap="inferno_r", vmin=0, vmax=vmax, s=4, marker="s")
        for kind, colour in (("intersection", "tab:blue"), ("dead_end", "tab:green"),
                             ("border_exit", "tab:gray")):
            pts = np.array([g.nodes[n]["pos"] for n in g.nodes if g.nodes[n]["kind"] == kind]).reshape(-1, 2)
            ax.scatter(pts[:, 1], pts[:, 0], s=14, c=colour, label=kind, edgecolors="white", linewidths=0.4)
        ax.set_title(f"{style} #{i}\ncongestion index = {labels.loc[i, 'congestion_index']:.2f}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, label="trips per segment")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Edge loads from a simplified all-or-nothing assignment of 200 random trips "
                 "(not a calibrated simulation)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--n", type=int, default=None, help="only the first N layouts")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = PROJECT_ROOT / cfg["paths"]["data_dir"]
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    images = np.load(data_dir / "layouts.npz")["images"][: args.n]
    params = pd.read_csv(data_dir / "params.csv").iloc[: len(images)]

    start = time.perf_counter()
    indicators = compute_all(images, cfg, args.workers)
    elapsed = time.perf_counter() - start
    labels = pd.concat([params[["id", "style", "split"]], indicators], axis=1)
    labels.to_csv(data_dir / "labels.csv", index=False)

    summary = labels.groupby("style")[INDICATORS].mean().T
    summary.to_csv(results_dir / "indicators_by_style.csv")
    plot_distributions(labels, results_dir / "indicator_distributions.png")
    plot_edge_loads(images, labels, cfg, results_dir / "edge_load_heatmaps.png")

    print(f"Computed indicators for {len(labels)} layouts in {elapsed:.1f} s "
          f"({args.workers} workers)")
    print(f"Rows with any missing indicator: {int(labels[INDICATORS].isna().any(axis=1).sum())}")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
