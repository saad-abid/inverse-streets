"""Tests for the demo helpers, and a headless run of the whole app (Phase 8)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.streamlit_app import binarize_upload, comparison_table, describe_generated, edge_load_figure
from src.config import load_config
from src.constraints import ConstraintChecker
from src.generate_layouts import draw_segment, generate_layout

ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
HAVE_ARTIFACTS = all((ROOT / p).exists() for p in
                     ("data/labels.csv", "data/layouts.npz", "models/predictor.pt", "models/generator.pt"))


def test_binarize_upload_dark_roads_rgb() -> None:
    img = np.full((128, 128, 3), 255, dtype=np.uint8)
    img[60:68, :, :] = 0  # dark horizontal road
    out = binarize_upload(img, 64, roads_dark=True)
    assert out.shape == (64, 64) and out.dtype == np.uint8
    assert out[32].all() and out[5].sum() == 0


def test_binarize_upload_bright_roads() -> None:
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, 10:13] = 255
    out = binarize_upload(img, 64, roads_dark=False)
    assert out[:, 11].all() and out.sum() == 64 * 3


def test_comparison_table() -> None:
    table = comparison_table({"road_density": 0.3}, {"road_density": 0.25, "other": 1})
    assert table["indicator"].tolist() == ["road density"]
    assert table["difference"].iloc[0] == pytest.approx(0.05)


def test_edge_load_figure_runs() -> None:
    img, _ = generate_layout("regular_grid", 64, 0, 0, CFG)
    fig = edge_load_figure(img, CFG)
    assert len(fig.axes) == 2  # map + colour bar


def test_describe_generated_flags_broken_layouts() -> None:
    labels = pd.DataFrame({"road_density": np.linspace(0.1, 0.5, 50),
                           "n_intersections": np.linspace(2, 50, 50),
                           "congestion_index": np.full(50, 3.0), "split": "train"})
    checker = ConstraintChecker(labels, CFG["constraints"])
    good, _ = generate_layout("regular_grid", 64, 0, 0, CFG)
    broken = np.zeros((64, 64), np.uint8)
    draw_segment(broken, (10, 0), (10, 63), 3)
    draw_segment(broken, (40, 0), (40, 63), 3)  # two separate roads -> not connected
    table = describe_generated(np.stack([good, broken]), CFG, checker, {"max_congestion_index": None}, 0)
    assert table["valid"].tolist() == [True, False]
    assert "layout_connected" in table["failed_rules"].iloc[1]


@pytest.mark.skipif(not HAVE_ARTIFACTS, reason="needs data/ and models/ from phases 1-4")
def test_app_runs_headless() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Predict", "Generate", "Benchmark"]
    assert any("synthetic" in m.value for m in at.sidebar.markdown)
    assert len(at.dataframe) >= 1  # the prediction table
