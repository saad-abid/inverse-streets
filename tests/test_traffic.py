"""Tests for OD sampling, all-or-nothing assignment and congestion indicators."""

import networkx as nx
import numpy as np
import pytest

from src.generate_layouts import draw_segment
from src.graph_metrics import road_graph
from src.traffic import (
    assign_all_or_nothing,
    block_centres,
    congestion_indicators,
    edge_key,
    nearest_nodes,
    sample_land_points,
    traffic_indicators,
)


def _path_graph() -> nx.Graph:
    """a(0) --2-- b(1) --3-- c(2), positions on a line."""
    g = nx.Graph()
    for n, col in ((0, 0.0), (1, 2.0), (2, 5.0)):
        g.add_node(n, pos=(0.0, col))
    g.add_edge(0, 1, length=2.0)
    g.add_edge(1, 2, length=3.0)
    return g


def test_assignment_loads_and_lengths() -> None:
    loads, lengths = assign_all_or_nothing(_path_graph(), [(0, 2), (0, 1), (2, 1)])
    assert loads == {(0, 1): 2, (1, 2): 2}
    assert sorted(lengths) == [2.0, 3.0, 5.0]


def test_assignment_takes_shortest_route() -> None:
    g = _path_graph()
    g.add_edge(0, 2, length=10.0)  # long direct road: shortest path still goes through b
    loads, _ = assign_all_or_nothing(g, [(0, 2)])
    assert loads[(0, 2)] == 0 and loads[(0, 1)] == 1 and loads[(1, 2)] == 1


def test_assignment_skips_same_node_and_unreachable() -> None:
    g = _path_graph()
    g.add_node(9, pos=(9.0, 9.0))
    loads, lengths = assign_all_or_nothing(g, [(0, 0), (0, 9)])
    assert lengths == [] and sum(loads.values()) == 0


def test_congestion_indicators() -> None:
    # loads 4, 1, 1 -> mean 2, max/mean = 2, share above 1.5*mean (=3) = 1/3
    out = congestion_indicators({(0, 1): 4, (1, 2): 1, (2, 3): 1}, threshold=1.5)
    assert out["congestion_index"] == pytest.approx(2.0)
    assert out["share_congested"] == pytest.approx(1 / 3)


def test_uniform_loads_mean_no_congestion() -> None:
    out = congestion_indicators({(0, 1): 3, (1, 2): 3}, threshold=2.0)
    assert out == {"congestion_index": 1.0, "share_congested": 0.0}


def test_block_centres_of_plus_crossing_image() -> None:
    img = np.zeros((21, 21), dtype=np.uint8)
    draw_segment(img, (10, 0), (10, 20), 1)
    draw_segment(img, (0, 10), (20, 10), 1)
    centres = block_centres(img)
    assert len(centres) == 4
    assert sorted(map(tuple, centres.round(1).tolist())) == [(4.5, 4.5), (4.5, 15.5), (15.5, 4.5), (15.5, 15.5)]


def test_nearest_nodes() -> None:
    assert nearest_nodes(_path_graph(), np.array([[0.0, 4.4], [1.0, 0.2]])) == [2, 0]


def test_land_points_are_never_on_roads() -> None:
    img = np.zeros((21, 21), dtype=np.uint8)
    draw_segment(img, (10, 0), (10, 20), 3)
    pts = sample_land_points(img, 500, np.random.default_rng(0)).astype(int)
    assert pts.shape == (500, 2)
    assert img[pts[:, 0], pts[:, 1]].sum() == 0


def test_cul_de_sac_trips_use_many_nodes() -> None:
    """Regression: with block centres, a cul-de-sac has only 2 blocks -> all trips on 1 pair."""
    from src.config import load_config
    from src.generate_layouts import generate_layout

    img, _ = generate_layout("cul_de_sac", 64, seed=0, index=0, cfg=load_config())
    g = road_graph(img)
    out, loads = traffic_indicators(img, g, np.random.default_rng(0))
    assert sum(v > 0 for v in loads.values()) > 0.5 * len(loads)
    assert out["n_trips_assigned"] > 150


def test_edge_key_is_order_independent() -> None:
    assert edge_key(3, 1) == edge_key(1, 3) == (1, 3)


def test_traffic_on_grid_layout_is_reproducible() -> None:
    img = np.zeros((41, 41), dtype=np.uint8)
    for k in (10, 20, 30):
        draw_segment(img, (k, 0), (k, 40), 1)
        draw_segment(img, (0, k), (40, k), 1)
    g = road_graph(img)
    a, loads_a = traffic_indicators(img, g, np.random.default_rng(1))
    b, loads_b = traffic_indicators(img, g, np.random.default_rng(1))
    assert a == b and loads_a == loads_b
    assert a["n_blocks"] == 16 and a["n_trips_assigned"] > 0
    assert a["congestion_index"] >= 1.0
