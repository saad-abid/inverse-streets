"""Tests for metrics, dataset utilities, the CNN, baselines and latency measurement (Phase 3)."""

import numpy as np
import pytest
import torch

from src.config import set_seed
from src.dataset import LayoutDataset, TargetScaler, random_dihedral
from src.evaluate import metrics_table, regression_metrics
from src.predictor import (
    LayoutCNN,
    count_parameters,
    fit_density_linear,
    mean_baseline,
    measure_latency,
    predict,
    predict_density_linear,
    train_model,
)


# --------------------------------------------------------------------------- metrics
def test_perfect_prediction_metrics() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert regression_metrics(y, y) == {"mae": 0.0, "rmse": 0.0, "r2": 1.0, "n": 3}


def test_mean_prediction_has_r2_zero() -> None:
    y = np.array([1.0, 2.0, 3.0, 6.0])
    assert regression_metrics(y, np.full(4, y.mean()))["r2"] == pytest.approx(0.0)


def test_known_errors() -> None:
    out = regression_metrics(np.array([0.0, 0.0]), np.array([1.0, -3.0]))
    assert out["mae"] == pytest.approx(2.0)
    assert out["rmse"] == pytest.approx(np.sqrt(5.0))
    assert np.isnan(out["r2"])  # constant truth


def test_metrics_skip_missing_pairs() -> None:
    out = regression_metrics(np.array([1.0, np.nan, 3.0]), np.array([2.0, 5.0, np.nan]))
    assert out["n"] == 1 and out["mae"] == 1.0
    assert regression_metrics(np.array([np.nan]), np.array([1.0]))["n"] == 0


def test_metrics_table_shape() -> None:
    y = np.random.default_rng(0).random((10, 2))
    table = metrics_table(y, {"a": y, "b": y * 0}, ["t1", "t2"])
    assert len(table) == 4 and set(table["model"]) == {"a", "b"}


# --------------------------------------------------------------------------- dataset
def test_scaler_roundtrip_and_standardizes() -> None:
    y = np.random.default_rng(0).normal(5, 3, size=(100, 3))
    scaler = TargetScaler.fit(y)
    z = scaler.transform(y)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-5) and np.allclose(z.std(axis=0), 1, atol=1e-4)
    assert np.allclose(scaler.inverse(z), y, atol=1e-4)
    assert np.allclose(TargetScaler.from_dict(scaler.to_dict()).mean, scaler.mean)


def test_scaler_handles_constant_column() -> None:
    scaler = TargetScaler.fit(np.ones((5, 1)))
    assert np.all(np.isfinite(scaler.transform(np.ones((5, 1)))))


def test_dihedral_keeps_road_pixels() -> None:
    torch.manual_seed(0)
    x = (torch.rand(1, 16, 16) > 0.7).float()
    for _ in range(20):
        y = random_dihedral(x)
        assert y.shape == x.shape and y.sum() == x.sum()


def test_dataset_items() -> None:
    images = np.zeros((4, 64, 64), dtype=np.uint8)
    ds = LayoutDataset(images, np.zeros((4, 3), dtype=np.float32), augment=True)
    x, y = ds[2]
    assert len(ds) == 4 and x.shape == (1, 64, 64) and y.shape == (3,)


# --------------------------------------------------------------------------- model
def test_cnn_output_shape_and_size() -> None:
    model = LayoutCNN(n_outputs=4)
    assert model(torch.zeros(5, 1, 64, 64)).shape == (5, 4)
    assert count_parameters(model) < 500_000


def test_training_reduces_loss_on_tiny_data() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    images = (rng.random((32, 64, 64)) < rng.uniform(0.1, 0.6, (32, 1, 1))).astype(np.uint8)
    y = images.reshape(32, -1).mean(axis=1, keepdims=True).astype(np.float32)  # density
    scaler = TargetScaler.fit(y)
    ds = LayoutDataset(images, scaler.transform(y))
    model = LayoutCNN(n_outputs=1, channels=(8, 8))
    cfg = {"batch_size": 8, "epochs": 15, "learning_rate": 3e-3, "num_workers": 0}
    history = train_model(model, ds, ds, cfg, "cpu", seed=0)
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    pred = predict(model, images, scaler, "cpu")
    assert pred.shape == (32, 1) and np.isfinite(pred).all()


# --------------------------------------------------------------------------- baselines
def test_mean_baseline() -> None:
    y = np.array([[1.0, 10.0], [3.0, 30.0]])
    assert np.array_equal(mean_baseline(y, 3), np.array([[2.0, 20.0]] * 3))


def test_density_linear_recovers_exact_line() -> None:
    density = np.linspace(0.1, 0.6, 20)
    y = np.stack([2 + 5 * density, -1 + 0.5 * density], axis=1)
    coef = fit_density_linear(density, y)
    assert np.allclose(coef, [[2, -1], [5, 0.5]])
    assert np.allclose(predict_density_linear(coef, density), y)


# --------------------------------------------------------------------------- label noise
def test_label_repeatability_is_perfect_for_deterministic_indicators() -> None:
    import pandas as pd

    from src.build_labels import layout_indicators
    from src.config import load_config
    from src.dataset import SplitData
    from src.generate_layouts import generate_layout
    from src.predictor import label_repeatability

    cfg = load_config()
    targets = ["road_density", "n_intersections", "congestion_index"]
    images = np.stack([generate_layout(s, 64, 0, i, cfg)[0]
                       for i, s in enumerate(["regular_grid", "radial", "cul_de_sac"] * 3)])
    y = np.array([[layout_indicators(img, cfg["seed"], i, cfg)[t] for t in targets]
                  for i, img in enumerate(images)])
    split = SplitData(images, y, pd.DataFrame({"id": range(len(images)), "style": "x"}))
    table = label_repeatability(split, cfg, targets).set_index("indicator")
    assert table.loc["road_density", "r2"] == 1.0
    assert table.loc["n_intersections", "r2"] == 1.0
    assert table.loc["congestion_index", "mae"] > 0  # random trips differ between seeds


# --------------------------------------------------------------------------- latency
def test_latency_is_positive() -> None:
    out = measure_latency(LayoutCNN(2, channels=(4, 4)), 64, n_warmup=2, n_runs=5, batch_size=4)
    assert out["single_mean_ms"] > 0 and out["single_p95_ms"] >= 0 and out["batched_ms_per_image"] > 0
