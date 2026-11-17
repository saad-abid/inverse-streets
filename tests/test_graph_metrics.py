"""Tests for road-graph extraction and graph indicators on hand-made layouts."""

import math

import numpy as np
import pytest

from src.config import load_config
from src.generate_layouts import STYLES, draw_segment, generate_layout
from src.graph_metrics import (
    connectivity,
    graph_indicators,
    path_length,
    pixel_graph,
    road_graph,
)

CFG = load_config()


def _canvas(size: int = 41) -> np.ndarray:
    return np.zeros((size, size), dtype=np.uint8)


def _indicators(img: np.ndarray) -> dict:
    return graph_indicators(img, road_graph(img), np.random.default_rng(0))


@pytest.mark.parametrize("width", [1, 2, 3])
def test_plus_shape(width: int) -> None:
    img = _canvas()
    draw_segment(img, (20, 5), (20, 35), width)
    draw_segment(img, (5, 20), (35, 20), width)
    ind = _indicators(img)
    assert (ind["n_intersections"], ind["n_dead_ends"], ind["n_border_exits"]) == (1, 4, 0)
    assert ind["n_edges"] == 4


@pytest.mark.parametrize("width", [1, 2, 3])
def test_hash_shape_has_four_intersections(width: int) -> None:
    img = _canvas()
    for k in (13, 27):
        draw_segment(img, (k, 5), (k, 35), width)
        draw_segment(img, (5, k), (35, k), width)
    ind = _indicators(img)
    assert (ind["n_intersections"], ind["n_dead_ends"], ind["n_border_exits"]) == (4, 8, 0)


@pytest.mark.parametrize("width", [1, 2, 3])
def test_full_grid_three_by_three(width: int) -> None:
    img = _canvas()
    for k in (10, 20, 30):
        draw_segment(img, (k, 0), (k, 40), width)
        draw_segment(img, (0, k), (40, k), width)
    ind = _indicators(img)
    assert (ind["n_intersections"], ind["n_dead_ends"], ind["n_border_exits"]) == (9, 0, 12)
    assert ind["connectivity"] == 1.0


def test_straight_line_has_detour_one_and_known_length() -> None:
    img = _canvas()
    draw_segment(img, (20, 5), (20, 25), 1)
    ind = _indicators(img)
    assert ind["n_dead_ends"] == 2
    assert ind["avg_shortest_path"] == pytest.approx(20.0)
    assert ind["detour_index"] == pytest.approx(1.0)


def test_plus_average_path_matches_hand_computation() -> None:
    # centre->arm = 15, arm->arm = 30; all 5 nodes as sources: (4*15 + 4*(15 + 3*30)) / 20 = 24
    img = _canvas()
    draw_segment(img, (20, 5), (20, 35), 1)
    draw_segment(img, (5, 20), (35, 20), 1)
    assert _indicators(img)["avg_shortest_path"] == pytest.approx(24.0)


def test_l_shape_detour_close_to_sqrt_two() -> None:
    img = _canvas()
    draw_segment(img, (5, 5), (5, 25), 1)
    draw_segment(img, (5, 25), (25, 25), 1)
    ind = _indicators(img)
    assert ind["n_intersections"] == 0 and ind["n_dead_ends"] == 2  # a corner is not a junction
    assert ind["detour_index"] == pytest.approx(math.sqrt(2), abs=0.05)


def test_closed_loop_has_no_intersections_or_dead_ends() -> None:
    img = _canvas()
    corners = [(10, 10), (10, 30), (30, 30), (30, 10), (10, 10)]
    for a, b in zip(corners, corners[1:]):
        draw_segment(img, a, b, 1)
    ind = _indicators(img)
    assert (ind["n_intersections"], ind["n_dead_ends"], ind["n_border_exits"]) == (0, 0, 0)
    assert ind["n_edges"] >= 3 and ind["connectivity"] == 1.0


def test_two_separate_roads_give_half_connectivity() -> None:
    img = _canvas()
    draw_segment(img, (10, 5), (10, 35), 1)
    draw_segment(img, (30, 5), (30, 35), 1)
    g = road_graph(img)
    assert g.number_of_nodes() == 4
    assert connectivity(g) == 0.5


def test_empty_image_gives_nan_paths() -> None:
    ind = _indicators(_canvas())
    assert ind["n_nodes"] == 0
    assert math.isnan(ind["avg_shortest_path"]) and math.isnan(ind["connectivity"])


def test_pixel_graph_corner_is_not_a_junction() -> None:
    img = _canvas(10)
    img[2, 2:6] = 1
    img[2:6, 5] = 1
    pg = pixel_graph(img)
    assert max(dict(pg.degree).values()) == 2


def test_path_length() -> None:
    assert path_length([(0, 0), (0, 1), (1, 2)]) == pytest.approx(1 + math.sqrt(2))


@pytest.mark.parametrize("style", list(STYLES))
def test_generated_layouts_give_finite_connected_graphs(style: str) -> None:
    for i in range(10):
        img, _ = generate_layout(style, 64, seed=0, index=i, cfg=CFG)
        ind = _indicators(img)
        assert ind["connectivity"] == 1.0
        assert np.isfinite(ind["avg_shortest_path"]) and ind["detour_index"] >= 1.0 - 1e-9


def test_cul_de_sac_dead_ends_match_generator() -> None:
    for i in range(60):
        img, params = generate_layout("cul_de_sac", 64, seed=0, index=i, cfg=CFG)
        expected = params["n_branches"] + params["n_spurs"]
        assert _indicators(img)["n_dead_ends"] == expected, f"sample {i}"


def test_radial_detour_never_below_one() -> None:
    """Regression: edge lengths must include the distance to a large junction's centre."""
    for i in range(80):
        img, _ = generate_layout("radial", 64, seed=0, index=i, cfg=CFG)
        assert _indicators(img)["detour_index"] >= 1.0


def test_regular_grid_has_no_dead_ends() -> None:
    for i in range(10):
        img, _ = generate_layout("regular_grid", 64, seed=0, index=i, cfg=CFG)
        assert _indicators(img)["n_dead_ends"] == 0
