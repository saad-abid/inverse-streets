"""Tests for the synthetic layout generator (Phase 1)."""

import numpy as np
import pytest
from skimage.measure import label

from src.config import load_config
from src.generate_layouts import (
    STYLES,
    assign_splits,
    draw_segment,
    generate_dataset,
    generate_layout,
)

CFG = load_config()
SIZE = CFG["dataset"]["image_size"]
SEEDS = range(40)


def _n_components(img: np.ndarray) -> int:
    """Number of 8-connected road components."""
    return int(label(img, connectivity=2).max())


@pytest.mark.parametrize("style", list(STYLES))
def test_images_are_binary_with_right_shape(style: str) -> None:
    for i in SEEDS:
        img, _ = generate_layout(style, SIZE, seed=0, index=i, cfg=CFG)
        assert img.shape == (SIZE, SIZE)
        assert img.dtype == np.uint8
        assert set(np.unique(img)) <= {0, 1}


@pytest.mark.parametrize("style", list(STYLES))
def test_road_network_is_connected(style: str) -> None:
    for i in SEEDS:
        img, _ = generate_layout(style, SIZE, seed=0, index=i, cfg=CFG)
        assert _n_components(img) == 1, f"{style} sample {i} is not connected"


@pytest.mark.parametrize("style", list(STYLES))
def test_density_is_reasonable(style: str) -> None:
    for i in SEEDS:
        img, _ = generate_layout(style, SIZE, seed=0, index=i, cfg=CFG)
        assert 0.02 < img.mean() < 0.8


def test_dataset_is_reproducible() -> None:
    a_img, a_df = generate_dataset(CFG, n=40)
    b_img, b_df = generate_dataset(CFG, n=40)
    assert np.array_equal(a_img, b_img)
    assert a_df.equals(b_df)


def test_different_seed_gives_different_dataset() -> None:
    other = {**CFG, "seed": CFG["seed"] + 1}
    a_img, _ = generate_dataset(CFG, n=20)
    b_img, _ = generate_dataset(other, n=20)
    assert not np.array_equal(a_img, b_img)


def test_styles_are_balanced() -> None:
    _, df = generate_dataset(CFG, n=40)
    assert df["style"].value_counts().nunique() == 1
    assert set(df["style"]) == set(STYLES)


def test_splits_are_disjoint_and_have_right_sizes() -> None:
    groups = np.array(["a"] * 100 + ["b"] * 100)
    splits = assign_splits(groups, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    for g in ("a", "b"):
        counts = {s: int(np.sum((groups == g) & (splits == s))) for s in ("train", "val", "test")}
        assert counts == {"train": 70, "val": 15, "test": 15}
    assert set(splits) == {"train", "val", "test"}


@pytest.mark.parametrize("width", [1, 2, 3])
def test_draw_segment_width(width: int) -> None:
    img = np.zeros((20, 20), dtype=np.uint8)
    draw_segment(img, (10, 0), (10, 19), width)
    assert img.sum(axis=0).tolist() == [width] * 20  # every column has `width` road pixels


def test_draw_segment_clips_outside_image() -> None:
    img = np.zeros((10, 10), dtype=np.uint8)
    draw_segment(img, (5, -20), (5, 30), 1)
    assert img.sum() == 10
