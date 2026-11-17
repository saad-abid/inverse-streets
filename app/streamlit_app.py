"""InverseStreets demo: predict indicators, generate layouts for targets, show the edge benchmark.

Run from the project root:  streamlit run app/streamlit_app.py
Needs the outputs of Phases 1-4 (data/, models/) and Phase 7 (results/edge_benchmark.*).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # streamlit puts app/ on the path, the code lives in src/
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.build_labels import layout_indicators
from src.config import load_config
from src.constraints import ConstraintChecker, RuleResult, all_passed
from src.graph_metrics import road_graph
from src.traffic import edge_key, traffic_indicators

HONESTY = ("All layouts are **synthetic** 64x64 street patterns. Traffic indicators come from a "
           "**simplified all-or-nothing shortest-path assignment** of random trips, not from a "
           "calibrated traffic simulation.")
INDICATOR_NAMES = {"road_density": "road density", "n_intersections": "intersections",
                   "avg_shortest_path": "avg shortest path (px)", "congestion_index": "congestion index"}


# --------------------------------------------------------------------------- pure helpers (tested)
def binarize_upload(img: np.ndarray, size: int, roads_dark: bool = True) -> np.ndarray:
    """Turn an uploaded image (H, W) or (H, W, C) into a binary size x size layout (road = 1)."""
    gray = img.astype(float)
    if gray.ndim == 3:
        gray = gray[..., :3].mean(axis=2)
    resized = np.asarray(Image.fromarray(gray.astype(np.uint8)).resize((size, size), Image.NEAREST),
                         dtype=float)
    threshold = (resized.min() + resized.max()) / 2
    roads = resized < threshold if roads_dark else resized > threshold
    return roads.astype(np.uint8)


def comparison_table(predicted: dict[str, float], true: dict[str, float]) -> pd.DataFrame:
    """Predicted vs graph-method value for every predicted indicator."""
    rows = [{"indicator": INDICATOR_NAMES.get(k, k), "CNN prediction": predicted[k],
             "graph method": true[k], "difference": predicted[k] - true[k]} for k in predicted]
    return pd.DataFrame(rows)


def edge_load_figure(img: np.ndarray, cfg: dict, seed: int = 0) -> plt.Figure:
    """Layout with its road graph nodes and edge loads of the simplified assignment."""
    g = road_graph(img, cfg["graph"])
    _, loads = traffic_indicators(img, g, np.random.default_rng(seed), cfg["traffic"])
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(img, cmap="Greys", vmax=4, interpolation="nearest")
    vmax = max(loads.values()) if loads else 1
    sc = None
    for u, v, data in g.edges(data=True):
        px = np.array(data["pixels"])
        sc = ax.scatter(px[:, 1], px[:, 0], c=[loads.get(edge_key(u, v), 0)] * len(px),
                        cmap="inferno_r", vmin=0, vmax=vmax, s=5, marker="s")
    for kind, colour in (("intersection", "tab:blue"), ("dead_end", "tab:green")):
        pts = np.array([g.nodes[n]["pos"] for n in g.nodes if g.nodes[n]["kind"] == kind]).reshape(-1, 2)
        ax.scatter(pts[:, 1], pts[:, 0], s=14, c=colour, label=kind, edgecolors="white", linewidths=0.4)
    if sc is not None:
        fig.colorbar(sc, ax=ax, fraction=0.046, label="trips per segment")
    ax.set_xticks([]), ax.set_yticks([])
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


def describe_generated(images: np.ndarray, cfg: dict, checker: ConstraintChecker,
                       spec: dict, seed: int) -> pd.DataFrame:
    """Recomputed indicators and layout-rule outcome for generated layouts."""
    rows = []
    for i, img in enumerate(images):
        ind = layout_indicators(img, seed, i, cfg)
        checks = checker.check_layout(ind, spec)
        rows.append({"road_density": ind["road_density"], "n_intersections": ind["n_intersections"],
                     "congestion_index": ind["congestion_index"], "connectivity": ind["connectivity"],
                     "valid": all_passed(checks),
                     "failed_rules": ", ".join(c.rule for c in checks if not c.passed)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- cached resources
@st.cache_resource
def load_resources() -> dict:
    """Config, data, trained models and the constraint checker (loaded once per session)."""
    from src.generator import load_generator
    from src.llm_spec import indicator_stats
    from src.predictor import load_predictor

    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir, models_dir = ROOT / cfg["paths"]["data_dir"], ROOT / cfg["paths"]["models_dir"]
    labels = pd.read_csv(data_dir / "labels.csv")
    return {
        "cfg": cfg,
        "images": np.load(data_dir / "layouts.npz")["images"],
        "labels": labels,
        "predictor": load_predictor(models_dir / "predictor.pt"),
        "generator": load_generator(models_dir / "generator.pt"),
        "checker": ConstraintChecker(labels, cfg["constraints"]),
        "stats": indicator_stats(labels),
    }


def show_rules(results: list[RuleResult]) -> None:
    """One coloured line per rule."""
    for r in results:
        (st.success if r.passed else st.error)(f"**{r.rule}**: {r.message}")


# --------------------------------------------------------------------------- tabs
def predict_tab(res: dict) -> None:
    from src.predictor import predict

    cfg, labels = res["cfg"], res["labels"]
    model, scaler, targets = res["predictor"]
    source = st.radio("Layout", ["pick a test layout", "upload an image"], horizontal=True)
    if source == "pick a test layout":
        test = labels[labels["split"] == "test"]
        style = st.selectbox("Style", sorted(test["style"].unique()))
        ids = test.loc[test["style"] == style, "id"].tolist()
        layout_id = st.select_slider("Test layout id", options=ids)
        img = res["images"][layout_id]
    else:
        upload = st.file_uploader("Image of a street layout (it is resized to 64x64)", type=["png", "jpg", "jpeg"])
        roads_dark = st.checkbox("Roads are darker than blocks", value=True)
        if upload is None:
            st.info("Upload an image, or pick a test layout.")
            return
        img = binarize_upload(np.asarray(Image.open(upload)), cfg["dataset"]["image_size"], roads_dark)

    start = time.perf_counter()
    pred = predict(model, img[None], scaler, "cpu")[0]
    ms = (time.perf_counter() - start) * 1000
    true = layout_indicators(img, cfg["seed"], 0, cfg)

    left, right = st.columns([1, 1])
    with left:
        st.pyplot(edge_load_figure(img, cfg))
        st.caption("Edge loads: 200 random trips, each on its shortest path (simplified assignment).")
    with right:
        st.dataframe(comparison_table(dict(zip(targets, pred)), true).round(3), hide_index=True)
        st.caption(f"CNN inference: {ms:.1f} ms on CPU (PyTorch, includes Python overhead). "
                   f"Graph method = skeleton -> road graph -> indicators.")
        st.metric("dead ends (graph method)", true["n_dead_ends"])
        st.metric("detour index (graph method)", f"{true['detour_index']:.2f}")


def generate_tab(res: dict) -> None:
    from src.generator import generate_layouts
    from src.llm_spec import interpret_request, ollama_llm

    cfg, checker = res["cfg"], res["checker"]
    c = cfg["constraints"]
    mode = st.radio("Targets from", ["sliders", "text request (local LLM via Ollama)"], horizontal=True)
    if mode == "sliders":
        spec = {"road_density": st.slider("road density", 0.0, 0.7, 0.30, 0.01),
                "intersections": st.slider("intersections", 0, 70, 25),
                "max_congestion_index": (st.slider("max congestion index", 1.0, 7.0, 3.5, 0.1)
                                         if st.checkbox("limit congestion") else None)}
        st.caption(f"Allowed density range: [{c['density_min']:.2f}, {c['density_max']:.2f}]. "
                   "Other limits come from the training data.")
        checks = checker.check_targets(spec)
    else:
        request = st.text_input("Request", "a medium density grid with about 20 intersections")
        if not st.button("Send to the LLM"):
            st.info("The LLM answer is checked with a JSON schema and the consistency rules before anything is generated.")
            return
        with st.spinner("Asking the local LLM..."):
            result = interpret_request(request, ollama_llm(cfg["llm"]), checker, res["stats"])
        st.code(result.llm_answer or "(no answer)", language="json")
        spec, checks = result.spec, result.checks
    show_rules(checks)
    if spec is None or not all_passed(checks):
        st.warning("Rejected: nothing is generated.")
        return

    n = st.slider("layouts to generate", 4, 16, 8, 4)
    seed = int(st.number_input("seed", value=0, step=1))
    model, scaler, conditions, threshold = res["generator"]
    values = {"road_density": spec["road_density"], "n_intersections": spec["intersections"]}
    target = np.array([values[k] for k in conditions], dtype=float)
    images = generate_layouts(model, target, scaler, n, seed, {**cfg["generator"], "threshold": threshold})
    table = describe_generated(images, cfg, checker, spec, seed)
    st.write(f"**{int(table['valid'].sum())} of {n}** generated layouts pass all layout rules "
             "(recomputed with the graph method).")
    cols = st.columns(4)
    for i, img in enumerate(images):
        row = table.iloc[i]
        with cols[i % 4]:
            st.image(255 - img * 255, clamp=True, width="stretch")
            mark = "valid" if row["valid"] else f"invalid: {row['failed_rules']}"
            st.caption(f"d={row['road_density']:.2f}, int={row['n_intersections']}, "
                       f"cong={row['congestion_index']:.2f} - {mark}")


def benchmark_tab() -> None:
    results = ROOT / "results"
    csv = results / "edge_benchmark.csv"
    if not csv.exists():
        st.info("Run `python -m src.edge_benchmark` first.")
        return
    table = pd.read_csv(csv)
    cols = ["version", "threads", "size_mb", "latency_mean_ms", "latency_p95_ms", "mean_r2", "device"]
    st.dataframe(table[cols].round(4), hide_index=True)
    st.image(str(results / "edge_benchmark.png"))
    st.caption("Measured on a laptop CPU. On an embedded board (e.g. a Jetson) the ranking can differ.")


def main() -> None:
    st.set_page_config(page_title="InverseStreets", layout="wide")
    st.title("InverseStreets")
    st.sidebar.markdown("### About\n" + HONESTY)
    st.sidebar.markdown("Predict mobility indicators from a street layout, generate layouts for "
                        "target indicators, and check the results with explicit rules.")
    try:
        res = load_resources()
    except FileNotFoundError as err:
        st.error(f"Missing file: {err}. Run phases 1-4 first (see README).")
        return
    predict_tab_, generate_tab_, benchmark_tab_ = st.tabs(["Predict", "Generate", "Benchmark"])
    with predict_tab_:
        predict_tab(res)
    with generate_tab_:
        generate_tab(res)
    with benchmark_tab_:
        benchmark_tab()


if __name__ == "__main__":
    main()
