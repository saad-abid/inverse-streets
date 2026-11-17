"""Evaluation metrics.

Phase 3: regression metrics for the predictor (MAE, RMSE, R² per indicator).
Phase 5: evaluation of generated layouts:
  - target agreement: indicators of generated layouts are recomputed with the graph method
    (the ground truth) and compared with the requested targets,
  - cross-check with the CNN predictor,
  - diversity between layouts generated for the same target,
  - plausibility: connected network, no small isolated fragments, realistic road width,
    and indicator distributions compared with real layouts.
A reference is computed the same way from the k real TRAINING layouts closest to each target.

Run:  python -m src.evaluate [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import wasserstein_distance
from skimage.measure import label
from skimage.morphology import skeletonize

from src.config import PROJECT_ROOT, load_config

INDICATORS = ["road_density", "n_intersections", "n_dead_ends", "avg_shortest_path",
              "detour_index", "congestion_index"]


# --------------------------------------------------------------------------- regression
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and R² for one indicator, on the pairs where both values exist
    (n = number of pairs used). R² is NaN if the true values are constant."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
    err = y_pred - y_true
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan"),
        "n": int(y_true.size),
    }


def metrics_table(y_true: np.ndarray, predictions: dict[str, np.ndarray],
                  target_names: list[str]) -> pd.DataFrame:
    """One row per (model, indicator) with MAE, RMSE and R²."""
    rows = []
    for model_name, y_pred in predictions.items():
        for j, target in enumerate(target_names):
            rows.append({"model": model_name, "indicator": target,
                         **regression_metrics(y_true[:, j], y_pred[:, j])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plausibility
def road_width(img: np.ndarray) -> float:
    """Typical road width in pixels: 2 * median(distance to the nearest block pixel, measured
    on the road centre lines) - 1. Gives 1 for 1-px roads and 3 for 3-px roads."""
    skel = skeletonize(img.astype(bool))
    if not skel.any():
        return float("nan")
    dist = distance_transform_edt(img)
    return float(2 * np.median(dist[skel]) - 1)


def fragment_sizes(img: np.ndarray) -> np.ndarray:
    """Pixel counts of the 8-connected road components, largest first."""
    labels = label(img, connectivity=2)
    return np.sort(np.bincount(labels.ravel())[1:])[::-1]


def plausibility(img: np.ndarray, width_range: tuple[float, float],
                 small_fragment_share: float = 0.1) -> dict[str, float]:
    """Plausibility checks for one layout:
    connected = a single road component; no_small_fragments = every extra component holds at
    least `small_fragment_share` of the road pixels; realistic_width = width inside the range
    measured on real layouts."""
    sizes = fragment_sizes(img)
    width = road_width(img)
    total = sizes.sum() if sizes.size else 0
    return {
        "n_fragments": int(sizes.size),
        "connected": bool(sizes.size == 1),
        "no_small_fragments": bool(sizes.size > 0 and (sizes >= small_fragment_share * total).all()),
        "road_width": width,
        "realistic_width": bool(width_range[0] <= width <= width_range[1]),
    }


def width_range_from_real(images: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    """Range (percentiles) of road widths measured on real layouts."""
    widths = [road_width(img) for img in images]
    return float(np.nanpercentile(widths, low)), float(np.nanpercentile(widths, high))


# --------------------------------------------------------------------------- diversity
def pairwise_difference(images: np.ndarray) -> float:
    """Mean share of differing pixels over all pairs of images (0 = identical)."""
    if len(images) < 2:
        return float("nan")
    flat = images.reshape(len(images), -1).astype(bool)
    return float(np.mean([np.mean(a != b) for a, b in itertools.combinations(flat, 2)]))


# --------------------------------------------------------------------------- targets
def select_targets(styles: pd.Series, n_per_style: int, seed: int) -> np.ndarray:
    """Indices of n_per_style random layouts of every style (their indicators become targets)."""
    rng = np.random.default_rng(seed)
    idx = [rng.choice(np.flatnonzero((styles == s).to_numpy()), n_per_style, replace=False)
           for s in sorted(styles.unique())]
    return np.concatenate(idx)


def nearest_real(train_targets: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k training layouts closest to the target (standardized distance)."""
    std = train_targets.std(axis=0)
    std[std == 0] = 1.0
    dist = np.linalg.norm((train_targets - target) / std, axis=1)
    return np.argsort(dist)[:k]


# --------------------------------------------------------------------------- per-sample table
def describe_layouts(images: np.ndarray, cfg: dict, width_range: tuple[float, float],
                     seed: int) -> pd.DataFrame:
    """Recomputed indicators (graph method + simplified traffic) and plausibility of layouts."""
    from src.build_labels import layout_indicators

    rows = []
    for i, img in enumerate(images):
        ind = layout_indicators(img, seed, i, cfg)
        rows.append({**{k: ind[k] for k in INDICATORS}, **plausibility(img, width_range)})
    return pd.DataFrame(rows)


def agreement(desc: pd.DataFrame, targets: np.ndarray, conditions: list[str]) -> pd.DataFrame:
    """Absolute and relative error of every condition, per sample."""
    out = pd.DataFrame(index=desc.index)
    for j, c in enumerate(conditions):
        out[f"target_{c}"] = targets[:, j]
        out[f"abs_err_{c}"] = (desc[c] - targets[:, j]).abs()
        out[f"rel_err_{c}"] = out[f"abs_err_{c}"] / np.maximum(np.abs(targets[:, j]), 1e-9)
    return out


def summary_rows(name: str, desc: pd.DataFrame, agree: pd.DataFrame, conditions: list[str],
                 diversity: float, real_desc: pd.DataFrame) -> list[dict]:
    """Rows (source, metric, value) for generation_metrics.csv."""
    rows = []
    for c in conditions:
        rows += [{"source": name, "metric": f"mae_{c}", "value": agree[f"abs_err_{c}"].mean()},
                 {"source": name, "metric": f"median_rel_err_{c}", "value": agree[f"rel_err_{c}"].median()}]
    rows.append({"source": name, "metric": "share_any_indicator_missing",
                 "value": desc[INDICATORS].isna().any(axis=1).mean()})
    for m in ("connected", "no_small_fragments", "realistic_width"):
        rows.append({"source": name, "metric": f"share_{m}", "value": desc[m].mean()})
    rows.append({"source": name, "metric": "mean_pairwise_pixel_difference", "value": diversity})
    for ind in INDICATORS:
        rows.append({"source": name, "metric": f"wasserstein_{ind}_vs_real_test",
                     "value": wasserstein_distance(desc[ind].dropna(), real_desc[ind].dropna())})
    return rows


# --------------------------------------------------------------------------- figures
def plot_target_agreement(gen: pd.DataFrame, ref: pd.DataFrame, conditions: list[str], path: Path) -> None:
    """Recomputed value vs target for each condition: mean and spread per target."""
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4.4))
    for ax, c in zip(np.atleast_1d(axes), conditions):
        for df, colour, label_txt in ((ref, "tab:gray", "10 nearest real layouts"),
                                      (gen, "tab:blue", "10 generated layouts")):
            g = df.groupby("target_id").agg(t=(f"target_{c}", "first"), m=(c, "mean"), s=(c, "std"))
            ax.errorbar(g["t"], g["m"], yerr=g["s"], fmt="o", ms=3, alpha=0.7, color=colour,
                        label=label_txt, elinewidth=0.8)
        lo, hi = gen[f"target_{c}"].min(), gen[f"target_{c}"].max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"target {c}"), ax.set_ylabel("recomputed with the graph method")
        ax.set_title(c)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Target agreement per target (mean ± std over 10 layouts)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_distributions(gen: pd.DataFrame, real: pd.DataFrame, path: Path) -> None:
    """Indicator histograms of generated layouts vs real test layouts."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, ind in zip(axes.flat, INDICATORS):
        data = [real[ind].dropna(), gen[ind].dropna()]
        bins = np.histogram_bin_edges(np.concatenate(data), bins=30)
        ax.hist(data[0], bins=bins, density=True, alpha=0.5, label="real test layouts")
        ax.hist(data[1], bins=bins, density=True, alpha=0.5, label="generated")
        ax.set_title(ind, fontsize=10)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Indicator distributions: generated vs real (congestion from a simplified "
                 "all-or-nothing assignment)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_grid(images: np.ndarray, gen: pd.DataFrame, target_rows: list[int], conditions: list[str],
              path: Path, n_cols: int = 8) -> None:
    """Generated layouts for a few targets (one row per target)."""
    fig, axes = plt.subplots(len(target_rows), n_cols, figsize=(1.7 * n_cols, 2.0 * len(target_rows)))
    for r, t in enumerate(target_rows):
        idx = gen.index[gen["target_id"] == t][:n_cols]
        first = gen.loc[idx[0]]
        txt = ", ".join(f"{c}={first[f'target_{c}']:.2f}" if first[f"target_{c}"] < 1
                        else f"{c}={first[f'target_{c}']:.0f}" for c in conditions)
        for c, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(images[i], cmap="gray_r", interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            colour = "tab:green" if gen.loc[i, "connected"] else "tab:red"
            ax.set_title(f"d={gen.loc[i, 'road_density']:.2f} int={gen.loc[i, 'n_intersections']}",
                         fontsize=7, color=colour)
        axes[r, 0].set_ylabel(f"{first['style']}\n{txt}", fontsize=6)
    fig.suptitle("Generated layouts per target (title green = connected network, red = not)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def run_generation_evaluation(cfg: dict, n_per_style: int = 15, n_samples: int = 10,
                              k_ref: int = 10) -> dict[str, pd.DataFrame]:
    """Generate n_samples layouts for each selected target and evaluate them."""
    import torch

    from src.dataset import load_splits
    from src.generator import generate_layouts, load_generator
    from src.predictor import load_predictor, predict

    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]
    generator, g_scaler, conditions, threshold = load_generator(models_dir / "generator.pt")
    predictor, p_scaler, p_targets = load_predictor(models_dir / "predictor.pt")
    gen_cfg = {**cfg["generator"], "threshold": threshold}
    data = load_splits(cfg, conditions)
    train, test = data["train"], data["test"]
    labels = pd.read_csv(PROJECT_ROOT / cfg["paths"]["data_dir"] / "labels.csv")
    real_test = labels[labels["split"] == "test"]
    width_range = width_range_from_real(train.images[:1000])

    chosen = select_targets(test.meta["style"], n_per_style, cfg["seed"])
    gen_images, ref_images, gen_meta, ref_meta, div = [], [], [], [], []
    torch.manual_seed(cfg["seed"])
    for t, k in enumerate(chosen):
        target = test.targets[k]
        imgs = generate_layouts(generator, target, g_scaler, n_samples, cfg["seed"] + 10_000 + t, gen_cfg)
        refs = train.images[nearest_real(train.targets, target, k_ref)]
        gen_images.append(imgs)
        ref_images.append(refs)
        meta = {"target_id": t, "test_id": int(test.meta["id"][k]), "style": test.meta["style"][k]}
        gen_meta += [meta] * len(imgs)
        ref_meta += [meta] * len(refs)
        div.append({"target_id": t, "generated": pairwise_difference(imgs),
                    "nearest_real": pairwise_difference(refs)})
    gen_images, ref_images = np.concatenate(gen_images), np.concatenate(ref_images)
    targets_gen = np.repeat(test.targets[chosen], n_samples, axis=0)
    targets_ref = np.repeat(test.targets[chosen], k_ref, axis=0)

    gen = pd.concat([pd.DataFrame(gen_meta), describe_layouts(gen_images, cfg, width_range, cfg["seed"] + 1)], axis=1)
    gen = pd.concat([gen, agreement(gen, targets_gen, conditions)], axis=1)
    ref = pd.concat([pd.DataFrame(ref_meta), describe_layouts(ref_images, cfg, width_range, cfg["seed"] + 1)], axis=1)
    ref = pd.concat([ref, agreement(ref, targets_ref, conditions)], axis=1)
    diversity = pd.DataFrame(div)

    cnn_gen = predict(predictor, gen_images, p_scaler, "cpu")
    cnn_check = metrics_table(gen[p_targets].to_numpy(float), {"cnn_on_generated": cnn_gen}, p_targets)
    for j, c in enumerate(p_targets):
        gen[f"cnn_{c}"] = cnn_gen[:, j]

    summary = pd.DataFrame(
        summary_rows("generated", gen, gen, conditions, diversity["generated"].mean(), real_test)
        + summary_rows(f"nearest_{k_ref}_real_train", ref, ref, conditions,
                       diversity["nearest_real"].mean(), real_test)
        + [{"source": "real_train", "metric": "road_width_p1", "value": width_range[0]},
           {"source": "real_train", "metric": "road_width_p99", "value": width_range[1]}])
    return {"generated": gen, "reference": ref, "diversity": diversity, "summary": summary,
            "cnn_check": cnn_check, "images": gen_images, "real_test": real_test,
            "conditions": conditions}


def per_target_table(gen: pd.DataFrame, diversity: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """One row per target: target values, mean recomputed values, errors, plausibility, diversity."""
    agg = {"style": ("style", "first"), "test_id": ("test_id", "first"),
           "share_connected": ("connected", "mean"), "share_realistic_width": ("realistic_width", "mean")}
    for c in conditions:
        agg.update({f"target_{c}": (f"target_{c}", "first"), f"mean_generated_{c}": (c, "mean"),
                    f"mae_{c}": (f"abs_err_{c}", "mean")})
    table = gen.groupby("target_id").agg(**agg).reset_index()
    return table.merge(diversity.rename(columns={"generated": "pairwise_diff_generated",
                                                 "nearest_real": "pairwise_diff_nearest_real"}),
                       on="target_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--per-style", type=int, default=15, help="targets per layout style")
    parser.add_argument("--samples", type=int, default=10, help="generated layouts per target")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    out = run_generation_evaluation(cfg, args.per_style, args.samples)
    gen, conditions = out["generated"], out["conditions"]

    out["summary"].to_csv(results_dir / "generation_metrics.csv", index=False)
    per_target_table(gen, out["diversity"], conditions).to_csv(results_dir / "generation_per_target.csv", index=False)
    out["cnn_check"].to_csv(results_dir / "generation_cnn_crosscheck.csv", index=False)
    by_style = gen.groupby("style").agg(
        **{f"mae_{c}": (f"abs_err_{c}", "mean") for c in conditions},
        share_connected=("connected", "mean"), share_realistic_width=("realistic_width", "mean"))
    by_style.to_csv(results_dir / "generation_by_style.csv")

    plot_target_agreement(gen, out["reference"], conditions, results_dir / "generation_target_agreement.png")
    plot_distributions(gen, out["real_test"], results_dir / "generation_distributions.png")
    rows = [int(t) for t in gen.drop_duplicates("style")["target_id"]]
    rows += [int(t) + 1 for t in rows]
    plot_grid(out["images"], gen, sorted(rows), conditions, results_dir / "generated_grid.png")

    pd.set_option("display.width", 140)
    print(out["summary"].pivot(index="metric", columns="source", values="value").round(4).to_string())
    print("\nBy target style (generated):")
    print(by_style.round(3).to_string())
    print("\nCNN predictor on generated layouts vs graph method:")
    print(out["cnn_check"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
