"""Simplified traffic assignment on a road graph.

Random origin-destination (OD) trips are drawn between random points of land (non-road
pixels, i.e. inside the city blocks), each linked to its nearest graph node. Each trip is
assigned to its single shortest path (all-or-nothing assignment): no road capacity, no
travel-time feedback from congestion, no calibrated demand. This is only a first step
towards what tools such as PTV Visum do with calibrated demand, capacities and
equilibrium assignment. The resulting indicators are relative measures of how much a
layout concentrates traffic on a few segments, not real traffic predictions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops

Edge = tuple[int, int]

DEFAULT_TRAFFIC_CFG: dict[str, Any] = {
    "n_trips": 200,               # OD trips per layout
    "congestion_threshold": 2.0,  # a segment is congested if load > threshold * mean load
    "min_block_area": 4,          # blocks smaller than this (pixels) are not counted
}


def edge_key(u: int, v: int) -> Edge:
    """Order-independent key for an undirected edge."""
    return (u, v) if u <= v else (v, u)


def block_centres(img: np.ndarray, min_area: int = 4) -> np.ndarray:
    """Centroids (row, col) of city blocks = 4-connected regions of non-road pixels."""
    regions = regionprops(label(img == 0, connectivity=1))
    centres = [r.centroid for r in regions if r.area >= min_area]
    return np.array(centres, dtype=float).reshape(-1, 2)


def sample_land_points(img: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` random non-road pixels (row, col): trip ends spread uniformly over the blocks."""
    land = np.argwhere(img == 0)
    if len(land) == 0:
        return np.empty((0, 2), dtype=float)
    return land[rng.integers(0, len(land), size=n)].astype(float)


def nearest_nodes(g: nx.Graph, points: np.ndarray) -> list[int]:
    """Graph node closest to each point (straight-line distance)."""
    nodes = sorted(g.nodes)
    tree = cKDTree(np.array([g.nodes[n]["pos"] for n in nodes]))
    _, idx = tree.query(points)
    return [nodes[i] for i in np.atleast_1d(idx)]


def assign_all_or_nothing(g: nx.Graph, od_nodes: list[tuple[int, int]]) -> tuple[dict[Edge, int], list[float]]:
    """Put every trip on its shortest path. Returns trips per edge and trip lengths.
    Trips whose origin and destination share a node, or are unreachable, are skipped."""
    loads: dict[Edge, int] = {edge_key(u, v): 0 for u, v in g.edges}
    lengths: list[float] = []
    by_origin: dict[int, list[int]] = defaultdict(list)
    for o, d in od_nodes:
        if o != d:
            by_origin[o].append(d)
    for o, dests in by_origin.items():
        dist, paths = nx.single_source_dijkstra(g, o, weight="length")
        for d in dests:
            if d not in paths:
                continue
            path = paths[d]
            for u, v in zip(path, path[1:]):
                loads[edge_key(u, v)] += 1
            lengths.append(float(dist[d]))
    return loads, lengths


def congestion_indicators(loads: dict[Edge, int], threshold: float) -> dict[str, float]:
    """Congestion index (max load / mean load) and share of segments with
    load > threshold * mean load. The mean is over all segments, including unused ones."""
    values = np.array(list(loads.values()), dtype=float)
    if values.size == 0 or values.mean() == 0:
        return {"congestion_index": float("nan"), "share_congested": float("nan")}
    mean = values.mean()
    return {"congestion_index": float(values.max() / mean),
            "share_congested": float(np.mean(values > threshold * mean))}


def traffic_indicators(img: np.ndarray, g: nx.Graph, rng: np.random.Generator,
                       cfg: dict | None = None) -> tuple[dict[str, float], dict[Edge, int]]:
    """Run the OD sampling + all-or-nothing assignment for one layout.
    Returns the indicators and the edge loads (for plotting)."""
    cfg = {**DEFAULT_TRAFFIC_CFG, **(cfg or {})}
    n_blocks = len(block_centres(img, cfg["min_block_area"]))
    if g.number_of_nodes() == 0 or not (img == 0).any():
        nan = float("nan")
        return {"n_blocks": n_blocks, "n_trips_assigned": 0, "avg_trip_length": nan,
                "congestion_index": nan, "share_congested": nan}, {}
    ends = sample_land_points(img, 2 * cfg["n_trips"], rng)
    access = nearest_nodes(g, ends)
    od_nodes = list(zip(access[0::2], access[1::2]))
    loads, lengths = assign_all_or_nothing(g, od_nodes)
    return {
        "n_blocks": n_blocks,
        "n_trips_assigned": len(lengths),
        "avg_trip_length": float(np.mean(lengths)) if lengths else float("nan"),
        **congestion_indicators(loads, cfg["congestion_threshold"]),
    }, loads
