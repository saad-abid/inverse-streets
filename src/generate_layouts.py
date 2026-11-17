"""Generate synthetic binary street-layout images (road pixel = 1, city block = 0).

Four layout styles are mixed so the dataset is varied:
- regular_grid:   evenly spaced horizontal and vertical streets,
- irregular_grid: streets shifted by a few pixels and some segments removed
                  (a random spanning tree is always kept, so the network stays connected),
- radial:         concentric rings crossed by straight spokes from a centre,
- cul_de_sac:     one main road with dead-end branches ending in a turning bulb.

Every sample uses its own random generator seeded with (global seed, sample index),
so the whole dataset is reproducible and any single sample can be regenerated.

Run:  python -m src.generate_layouts [--config configs/default.yaml] [--n 5000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from skimage.draw import disk, line

from src.config import PROJECT_ROOT, load_config

Image = np.ndarray
Params = dict[str, Any]
Point = tuple[int, int]  # (row, col)


# --------------------------------------------------------------------------- drawing
def _stamp_offsets(width: int) -> range:
    """Pixel offsets that turn a 1-pixel line into a line of the given width."""
    start = -((width - 1) // 2)
    return range(start, start + width)


def draw_segment(img: Image, p0: Point, p1: Point, width: int) -> None:
    """Draw a straight road of `width` pixels from p0 to p1 (clipped to the image)."""
    rr, cc = line(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
    h, w = img.shape
    for dr in _stamp_offsets(width):
        for dc in _stamp_offsets(width):
            r, c = rr + dr, cc + dc
            inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            img[r[inside], c[inside]] = 1


def _sample_spacing_width(rng: np.random.Generator, g: dict) -> tuple[int, int]:
    """Draw a street spacing and a road width from the config ranges."""
    spacing = int(rng.integers(g["spacing_min"], g["spacing_max"] + 1))
    width = int(rng.integers(g["road_width_min"], g["road_width_max"] + 1))
    return spacing, width


# --------------------------------------------------------------------------- styles
def regular_grid(size: int, rng: np.random.Generator, cfg: dict) -> tuple[Image, Params]:
    """Evenly spaced streets crossing the whole image."""
    spacing, width = _sample_spacing_width(rng, cfg["grid"])
    ox, oy = (int(v) for v in rng.integers(0, spacing, size=2))
    img = np.zeros((size, size), dtype=np.uint8)
    for x in range(ox, size, spacing):
        draw_segment(img, (0, x), (size - 1, x), width)
    for y in range(oy, size, spacing):
        draw_segment(img, (y, 0), (y, size - 1), width)
    return img, {"spacing": spacing, "road_width": width, "offset_x": ox, "offset_y": oy}


def _shifted_positions(size: int, spacing: int, shift: int, rng: np.random.Generator) -> list[int]:
    """Street positions every `spacing` pixels, each moved by up to +/- `shift` pixels."""
    base = np.arange(int(rng.integers(0, spacing)), size, spacing)
    moved = base + rng.integers(-shift, shift + 1, size=base.size)
    return sorted(set(np.clip(moved, 0, size - 1).tolist()))


def irregular_grid(size: int, rng: np.random.Generator, cfg: dict) -> tuple[Image, Params]:
    """Grid with shifted streets and randomly removed segments, always connected."""
    g_cfg = cfg["grid"]
    spacing, width = _sample_spacing_width(rng, g_cfg)
    p_remove = float(rng.uniform(g_cfg["removal_prob_min"], g_cfg["removal_prob_max"]))
    xs = _shifted_positions(size, spacing, g_cfg["max_shift"], rng)
    ys = _shifted_positions(size, spacing, g_cfg["max_shift"], rng)

    graph = nx.grid_2d_graph(len(xs), len(ys))  # nodes (i, j) = crossing of street xs[i], ys[j]
    for u, v in graph.edges:
        graph.edges[u, v]["weight"] = float(rng.random())
    tree = nx.minimum_spanning_tree(graph)  # random spanning tree -> keeps connectivity

    kept, removed = [], 0
    for u, v in sorted(graph.edges):
        if tree.has_edge(u, v) or rng.random() >= p_remove:
            kept.append(((ys[u[1]], xs[u[0]]), (ys[v[1]], xs[v[0]])))
        else:
            removed += 1

    # Stubs from the outer streets to the image border (leaves, safe to remove).
    stubs = []
    for y in ys:
        stubs += [((y, xs[0]), (y, 0)), ((y, xs[-1]), (y, size - 1))]
    for x in xs:
        stubs += [((ys[0], x), (0, x)), ((ys[-1], x), (size - 1, x))]
    for stub in stubs:
        if rng.random() >= p_remove:
            kept.append(stub)
        else:
            removed += 1

    img = np.zeros((size, size), dtype=np.uint8)
    for p0, p1 in kept:
        draw_segment(img, p0, p1, width)
    params = {"spacing": spacing, "road_width": width, "removal_prob": round(p_remove, 4),
              "max_shift": g_cfg["max_shift"], "n_removed_segments": removed}
    return img, params


def radial(size: int, rng: np.random.Generator, cfg: dict) -> tuple[Image, Params]:
    """Concentric ring roads crossed by spokes from a (slightly off-centre) centre."""
    r_cfg = cfg["radial"]
    spacing, width = _sample_spacing_width(rng, cfg["grid"])
    cy, cx = size / 2 + rng.uniform(-size / 8, size / 8, size=2)
    n_spokes = int(rng.integers(r_cfg["n_spokes_min"], r_cfg["n_spokes_max"] + 1))
    rotation = float(rng.uniform(0, 2 * np.pi / n_spokes))

    img = np.zeros((size, size), dtype=np.uint8)
    rows, cols = np.mgrid[0:size, 0:size]
    dist = np.hypot(rows - cy, cols - cx)
    max_r = min(cy, cx, size - 1 - cy, size - 1 - cx) - width  # rings fully inside image
    radii = np.arange(spacing, max_r + 1e-9, spacing)
    for r in radii:
        img[np.abs(dist - r) <= width / 2 + 0.25] = 1

    centre = (int(round(cy)), int(round(cx)))
    for k in range(n_spokes):
        angle = rotation + 2 * np.pi * k / n_spokes
        end = (int(round(cy + 2 * size * np.sin(angle))), int(round(cx + 2 * size * np.cos(angle))))
        draw_segment(img, centre, end, width)

    params = {"spacing": spacing, "road_width": width, "n_spokes": n_spokes,
              "n_rings": int(radii.size), "centre_row": round(float(cy), 2),
              "centre_col": round(float(cx), 2)}
    return img, params


def cul_de_sac(size: int, rng: np.random.Generator, cfg: dict) -> tuple[Image, Params]:
    """One main road with perpendicular dead-end branches (and optional side spurs)."""
    c_cfg = cfg["cul_de_sac"]
    spacing, width = _sample_spacing_width(rng, cfg["grid"])
    img = np.zeros((size, size), dtype=np.uint8)
    y0 = int(rng.integers(size // 3, 2 * size // 3))
    draw_segment(img, (y0, 0), (y0, size - 1), width)

    margin = width + 3  # keep bulbs away from the border
    spur_dir = int(rng.choice([-1, 1]))  # all spurs point the same way -> no collisions
    n_branches = n_spurs = 0
    for x in range(int(rng.integers(2, spacing)), size - 2, spacing):
        for direction in (-1, 1):
            available = (y0 - margin) if direction < 0 else (size - 1 - margin - y0)
            if available < spacing or rng.random() >= c_cfg["branch_prob"]:
                continue
            length = int(rng.integers(spacing, available + 1))
            end_y = y0 + direction * length
            draw_segment(img, (y0, x), (end_y, x), width)
            rr, cc = disk((end_y, x), width + 1, shape=img.shape)  # turning bulb
            img[rr, cc] = 1
            n_branches += 1
            if rng.random() < c_cfg["spur_prob"]:
                spur_y = y0 + direction * int(length * rng.uniform(0.4, 0.8))
                spur_len = int(rng.integers(2, max(3, spacing // 2) + 1))
                draw_segment(img, (spur_y, x), (spur_y, x + spur_dir * spur_len), width)
                n_spurs += 1

    transposed = bool(rng.random() < 0.5)  # random orientation
    if transposed:
        img = np.ascontiguousarray(img.T)
    params = {"spacing": spacing, "road_width": width, "main_road_pos": y0,
              "n_branches": n_branches, "n_spurs": n_spurs, "transposed": transposed}
    return img, params


STYLES: dict[str, Callable[[int, np.random.Generator, dict], tuple[Image, Params]]] = {
    "regular_grid": regular_grid,
    "irregular_grid": irregular_grid,
    "radial": radial,
    "cul_de_sac": cul_de_sac,
}


# --------------------------------------------------------------------------- dataset
def generate_layout(style: str, size: int, seed: int, index: int, cfg: dict) -> tuple[Image, Params]:
    """Generate one layout. The same (seed, index) always gives the same image."""
    rng = np.random.default_rng([seed, index])
    return STYLES[style](size, rng, cfg)


def assign_splits(groups: np.ndarray, fractions: dict[str, float], seed: int) -> np.ndarray:
    """Stratified train/val/test split: each group (style) is split with the same fractions."""
    rng = np.random.default_rng(seed)
    splits = np.empty(groups.size, dtype=object)
    for group in np.unique(groups):
        idx = rng.permutation(np.flatnonzero(groups == group))
        n_train = int(round(fractions["train"] * idx.size))
        n_val = int(round(fractions["val"] * idx.size))
        splits[idx[:n_train]] = "train"
        splits[idx[n_train:n_train + n_val]] = "val"
        splits[idx[n_train + n_val:]] = "test"
    return splits


def generate_dataset(cfg: dict, n: int | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate N layouts (styles in equal proportion) and their parameter table."""
    d_cfg = cfg["dataset"]
    n = n or d_cfg["n_samples"]
    size, seed, styles = d_cfg["image_size"], cfg["seed"], d_cfg["styles"]
    images = np.zeros((n, size, size), dtype=np.uint8)
    rows = []
    for i in range(n):
        style = styles[i % len(styles)]
        images[i], params = generate_layout(style, size, seed, i, cfg)
        common = {k: params.pop(k) for k in ("spacing", "road_width")}
        rows.append({"id": i, "style": style, **common,
                     "road_density": float(images[i].mean()), "extra": json.dumps(params)})
    df = pd.DataFrame(rows)
    df["split"] = assign_splits(df["style"].to_numpy(), d_cfg["splits"], seed)
    return images, df


def save_dataset(images: np.ndarray, df: pd.DataFrame, data_dir: Path) -> None:
    """Save images to layouts.npz and parameters to params.csv."""
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / "layouts.npz", images=images)
    df.to_csv(data_dir / "params.csv", index=False)


def plot_examples(images: np.ndarray, df: pd.DataFrame, path: Path, per_style: int = 6) -> None:
    """Save a figure with `per_style` example layouts for every style (one row per style)."""
    styles = list(df["style"].unique())
    fig, axes = plt.subplots(len(styles), per_style, figsize=(1.6 * per_style, 1.8 * len(styles)))
    for r, style in enumerate(styles):
        ids = df.loc[df["style"] == style, "id"].to_numpy()[:per_style]
        for c, i in enumerate(ids):
            ax = axes[r, c]
            ax.imshow(images[i], cmap="gray_r", interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"#{i}  d={images[i].mean():.2f}", fontsize=7)
        axes[r, 0].set_ylabel(style, fontsize=9)
    fig.suptitle("Synthetic layouts (black = road). d = road density", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--n", type=int, default=None, help="number of layouts (default: config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    images, df = generate_dataset(cfg, args.n)
    data_dir = PROJECT_ROOT / cfg["paths"]["data_dir"]
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    save_dataset(images, df, data_dir)
    plot_examples(images, df, results_dir / "example_layouts.png")
    summary = df.groupby("style")["road_density"].describe()[["count", "mean", "min", "max"]]
    summary.to_csv(results_dir / "dataset_summary.csv")
    print(f"Saved {len(df)} layouts to {data_dir}")
    print(df["split"].value_counts().to_string())
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
