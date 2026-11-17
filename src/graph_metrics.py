"""Turn a binary layout image into a road graph and compute mobility indicators.

Pipeline:
1. Skeletonize the road pixels (roads become 1-pixel-wide centre lines).
2. Build a pixel graph on the skeleton (8-neighbourhood, redundant diagonals removed).
3. Pixels with exactly 2 neighbours are "road", all others are graph nodes:
   1 neighbour = road end, 3+ neighbours = intersection (adjacent ones are merged).
4. Trace the road between nodes to get edges with their length in pixels.
5. Clean skeleton artefacts: drop tiny spurs, merge junctions that are very close.

Road ends that touch the image border are "border exits" (the street continues outside
the tile), not dead ends.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

Pixel = tuple[int, int]

DEFAULT_GRAPH_CFG: dict[str, Any] = {
    "min_spur_length": 3.0,   # dead-end edges shorter than this (px) are skeleton artefacts
    "merge_length": 3.0,      # intersections closer than this (px) are merged into one
    "border_margin": 2,       # road ends within this distance (px) of the border are exits
    "n_sources": 20,          # random source nodes for path-length / detour statistics
    "min_euclid": 5.0,        # ignore node pairs closer than this for the detour index
}


# --------------------------------------------------------------------------- pixel graph
def pixel_graph(img: np.ndarray) -> nx.Graph:
    """Graph of skeleton pixels. Diagonal links are skipped when a 4-neighbour path exists,
    so a simple corner does not look like a junction."""
    skel = skeletonize(img.astype(bool))
    pixels = set(map(tuple, np.argwhere(skel).tolist()))
    g = nx.Graph()
    g.add_nodes_from(pixels)
    for r, c in pixels:
        for dr, dc in ((0, 1), (1, 0)):
            if (r + dr, c + dc) in pixels:
                g.add_edge((r, c), (r + dr, c + dc), weight=1.0)
        for dc in (1, -1):
            diag = (r + 1, c + dc)
            if diag in pixels and (r + 1, c) not in pixels and (r, c + dc) not in pixels:
                g.add_edge((r, c), diag, weight=math.sqrt(2))
    return g


def path_length(pixels: list[Pixel]) -> float:
    """Length of a pixel path (1 per straight step, sqrt(2) per diagonal step)."""
    return float(sum(math.dist(a, b) for a, b in zip(pixels, pixels[1:])))


# --------------------------------------------------------------------------- road graph
def _node_clusters(pg: nx.Graph) -> dict[Pixel, int]:
    """Map every node pixel to a node id. Adjacent junction pixels share one id."""
    node_id: dict[Pixel, int] = {}
    junctions = [p for p in pg if pg.degree[p] >= 3]
    clusters = list(nx.connected_components(pg.subgraph(junctions)))
    clusters += [{p} for p in pg if pg.degree[p] in (0, 1)]
    for comp in nx.connected_components(pg):  # pure loops have no node pixel: pick one
        if all(pg.degree[p] == 2 for p in comp):
            clusters.append({min(comp)})
    for nid, cluster in enumerate(clusters):
        node_id.update({p: nid for p in cluster})
    return node_id


def _add_road(g: nx.Graph, u: int, v: int, pixels: list[Pixel]) -> None:
    """Add a road between nodes u and v. Self-loops and parallel roads are split with
    extra degree-2 nodes so the graph stays simple."""
    if u != v and not g.has_edge(u, v):
        g.add_edge(u, v, length=_road_length(g, u, v, pixels), pixels=pixels)
        return
    n_parts = 3 if u == v else 2
    if len(pixels) < n_parts + 1:
        return  # too short to matter
    cuts = [round(i * (len(pixels) - 1) / n_parts) for i in range(n_parts + 1)]
    ids = [u] + [max(g.nodes) + 1 + i for i in range(n_parts - 1)] + [v]
    for i, cut in enumerate(cuts[1:-1]):
        g.add_node(ids[i + 1], pos=tuple(map(float, pixels[cut])))
    for i in range(n_parts):
        part = pixels[cuts[i]:cuts[i + 1] + 1]
        g.add_edge(ids[i], ids[i + 1], length=_road_length(g, ids[i], ids[i + 1], part), pixels=part)


def _road_length(g: nx.Graph, u: int, v: int, pixels: list[Pixel]) -> float:
    """Pixel path length plus the distance from each node centre to the path end.
    (A large junction is a blob of pixels; its centre is not on the path itself.)"""
    return (math.dist(g.nodes[u]["pos"], pixels[0]) + path_length(pixels)
            + math.dist(pixels[-1], g.nodes[v]["pos"]))


def _trace_roads(pg: nx.Graph, node_id: dict[Pixel, int], g: nx.Graph) -> None:
    """Walk along degree-2 pixels from every node pixel to the next node pixel."""
    walked: set[frozenset] = set()
    for start in node_id:
        for nb in pg[start]:
            step = frozenset((start, nb))
            if step in walked or node_id.get(nb) == node_id[start]:
                continue
            walked.add(step)
            path, prev, cur = [start, nb], start, nb
            while cur not in node_id:
                nxt = next(p for p in pg[cur] if p != prev)
                walked.add(frozenset((cur, nxt)))
                path.append(nxt)
                prev, cur = cur, nxt
            _add_road(g, node_id[start], node_id[cur], path)


def _raw_road_graph(img: np.ndarray) -> nx.Graph:
    """Road graph straight from the skeleton, before artefact cleaning."""
    pg = pixel_graph(img)
    node_id = _node_clusters(pg)
    members: dict[int, list[Pixel]] = {}
    for p, nid in node_id.items():
        members.setdefault(nid, []).append(p)
    g = nx.Graph()
    for nid, pixels in members.items():
        g.add_node(nid, pos=tuple(np.mean(pixels, axis=0).tolist()))
    _trace_roads(pg, node_id, g)
    return g


def _is_border(pos: tuple[float, float], size: int, margin: int) -> bool:
    """True if a position lies within `margin` pixels of the image border."""
    return min(pos[0], pos[1], size - 1 - pos[0], size - 1 - pos[1]) <= margin


def _prune_spurs(g: nx.Graph, size: int, cfg: dict) -> bool:
    """Remove short dead-end edges attached to an intersection. Returns True if changed."""
    for n in list(g.nodes):
        if g.degree[n] != 1 or _is_border(g.nodes[n]["pos"], size, cfg["border_margin"]):
            continue
        (nb,) = g[n]
        if g[n][nb]["length"] < cfg["min_spur_length"] and g.degree[nb] >= 3:
            g.remove_node(n)
            return True
    return False


def _merge_close_junctions(g: nx.Graph, cfg: dict) -> bool:
    """Contract one edge between two intersections shorter than merge_length."""
    for u, v, data in g.edges(data=True):
        if data["length"] < cfg["merge_length"] and g.degree[u] >= 3 and g.degree[v] >= 3:
            pu, pv = np.array(g.nodes[u]["pos"]), np.array(g.nodes[v]["pos"])
            nx.contracted_nodes(g, u, v, self_loops=False, copy=False)
            g.nodes[u]["pos"] = tuple((pu + pv) / 2)
            g.nodes[u].pop("contraction", None)
            return True
    return False


def _dissolve_degree_two(g: nx.Graph) -> bool:
    """Replace a degree-2 node by one longer edge when that keeps the graph simple."""
    for n in list(g.nodes):
        if g.degree[n] != 2:
            continue
        a, b = g[n]
        if g.has_edge(a, b):
            continue
        pixels = g[a][n]["pixels"] + g[n][b]["pixels"]
        length = g[a][n]["length"] + g[n][b]["length"]
        g.remove_node(n)
        g.add_edge(a, b, length=length, pixels=pixels)
        return True
    return False


def road_graph(img: np.ndarray, cfg: dict | None = None) -> nx.Graph:
    """Build the cleaned road graph. Node attributes: pos (row, col) and kind
    ('intersection', 'dead_end', 'border_exit', 'road' or 'isolated')."""
    cfg = {**DEFAULT_GRAPH_CFG, **(cfg or {})}
    g = _raw_road_graph(img)
    while _prune_spurs(g, img.shape[0], cfg) or _merge_close_junctions(g, cfg) \
            or _dissolve_degree_two(g):
        pass
    for n in g.nodes:
        g.nodes[n]["kind"] = _node_kind(g, n, img.shape[0], cfg["border_margin"])
    return g


def _node_kind(g: nx.Graph, n: int, size: int, margin: int) -> str:
    """Classify a node by its degree and position."""
    deg = g.degree[n]
    if deg >= 3:
        return "intersection"
    if deg == 1:
        return "border_exit" if _is_border(g.nodes[n]["pos"], size, margin) else "dead_end"
    return "road" if deg == 2 else "isolated"


# --------------------------------------------------------------------------- indicators
def count_kinds(g: nx.Graph, kinds: Iterable[str]) -> dict[str, int]:
    """Number of nodes of each kind."""
    values = [g.nodes[n]["kind"] for n in g.nodes]
    return {k: values.count(k) for k in kinds}


def connectivity(g: nx.Graph) -> float:
    """Share of nodes that belong to the largest connected component."""
    if g.number_of_nodes() == 0:
        return float("nan")
    return max(len(c) for c in nx.connected_components(g)) / g.number_of_nodes()


def path_statistics(g: nx.Graph, rng: np.random.Generator, cfg: dict | None = None) -> dict[str, float]:
    """Mean shortest-path length and mean detour index, from random source nodes to all
    other nodes of the largest component. Detour = network distance / straight distance."""
    cfg = {**DEFAULT_GRAPH_CFG, **(cfg or {})}
    nan = {"avg_shortest_path": float("nan"), "detour_index": float("nan")}
    if g.number_of_nodes() < 2:
        return nan
    main = g.subgraph(max(nx.connected_components(g), key=len))
    nodes = sorted(main.nodes)
    sources = rng.choice(nodes, size=min(cfg["n_sources"], len(nodes)), replace=False)
    dists, detours = [], []
    for s in sources:
        lengths = nx.single_source_dijkstra_path_length(main, int(s), weight="length")
        for t, d in lengths.items():
            if t == s:
                continue
            dists.append(d)
            euclid = math.dist(main.nodes[s]["pos"], main.nodes[t]["pos"])
            if euclid >= cfg["min_euclid"]:
                detours.append(d / euclid)
    if not dists:
        return nan
    return {"avg_shortest_path": float(np.mean(dists)),
            "detour_index": float(np.mean(detours)) if detours else float("nan")}


def graph_indicators(img: np.ndarray, g: nx.Graph, rng: np.random.Generator,
                     cfg: dict | None = None) -> dict[str, float]:
    """All graph-based mobility indicators for one layout."""
    kinds = count_kinds(g, ("intersection", "dead_end", "border_exit"))
    return {
        "road_density": float(img.mean()),
        "n_intersections": kinds["intersection"],
        "n_dead_ends": kinds["dead_end"],
        "n_border_exits": kinds["border_exit"],
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "total_road_length": float(sum(d["length"] for _, _, d in g.edges(data=True))),
        "connectivity": connectivity(g),
        **path_statistics(g, rng, cfg),
    }
