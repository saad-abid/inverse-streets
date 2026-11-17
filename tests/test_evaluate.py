"""Tests for the generation-evaluation helpers (Phase 5)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    agreement,
    fragment_sizes,
    nearest_real,
    pairwise_difference,
    plausibility,
    road_width,
    select_targets,
    width_range_from_real,
)
from src.generate_layouts import draw_segment


def _road(width: int, size: int = 32) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    draw_segment(img, (16, 4), (16, 27), width)
    return img


@pytest.mark.parametrize("width", [1, 3])
def test_road_width_of_straight_road(width: int) -> None:
    assert road_width(_road(width)) == pytest.approx(width)


def test_road_width_empty_is_nan() -> None:
    assert np.isnan(road_width(np.zeros((8, 8), np.uint8)))


def test_fragment_sizes_sorted() -> None:
    img = _road(1)
    img[2, 2:5] = 1  # 3-pixel fragment
    assert fragment_sizes(img).tolist() == [24, 3]


def test_plausibility_connected_road() -> None:
    out = plausibility(_road(1), width_range=(1.0, 3.0))
    assert out["connected"] and out["no_small_fragments"] and out["realistic_width"]
    assert out["n_fragments"] == 1


def test_plausibility_detects_small_fragment_and_bad_width() -> None:
    img = _road(1)
    img[2, 2:4] = 1  # 2 of 26 road pixels -> below 10 %
    out = plausibility(img, width_range=(2.0, 3.0))
    assert not out["connected"] and not out["no_small_fragments"] and not out["realistic_width"]


def test_two_large_parts_are_not_small_fragments() -> None:
    img = _road(1)
    img[28, 4:28] = 1  # second long road, not connected
    out = plausibility(img, width_range=(1.0, 3.0))
    assert not out["connected"] and out["no_small_fragments"]


def test_pairwise_difference() -> None:
    a = np.zeros((4, 4), np.uint8)
    b = np.ones((4, 4), np.uint8)
    assert pairwise_difference(np.stack([a, a, a])) == 0.0
    assert pairwise_difference(np.stack([a, b])) == 1.0
    assert pairwise_difference(np.stack([a, a, b])) == pytest.approx(2 / 3)
    assert np.isnan(pairwise_difference(a[None]))


def test_select_targets_balanced_and_reproducible() -> None:
    styles = pd.Series(["a"] * 10 + ["b"] * 10)
    idx = select_targets(styles, 3, seed=0)
    assert len(idx) == 6 and (styles[idx] == "a").sum() == 3
    assert np.array_equal(idx, select_targets(styles, 3, seed=0))


def test_nearest_real_uses_standardized_distance() -> None:
    train = np.array([[0.1, 10.0], [0.5, 10.0], [0.1, 50.0]])
    assert nearest_real(train, np.array([0.12, 12.0]), 1).tolist() == [0]
    assert nearest_real(train, np.array([0.45, 11.0]), 2).tolist()[0] == 1


def test_agreement_errors() -> None:
    desc = pd.DataFrame({"road_density": [0.2, 0.3], "n_intersections": [10, 30]})
    out = agreement(desc, np.array([[0.25, 20], [0.25, 20]]), ["road_density", "n_intersections"])
    assert out["abs_err_road_density"].tolist() == pytest.approx([0.05, 0.05])
    assert out["rel_err_n_intersections"].tolist() == pytest.approx([0.5, 0.5])


def test_width_range_from_real() -> None:
    lo, hi = width_range_from_real(np.stack([_road(1), _road(3)]), 0, 100)
    assert (lo, hi) == pytest.approx((1.0, 3.0))
