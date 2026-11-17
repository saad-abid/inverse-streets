"""Tests for ONNX export, INT8 quantization and the latency / size helpers (Phase 7)."""

import numpy as np
import pytest
import torch

from src.edge_benchmark import (
    cpu_name,
    export_onnx,
    file_size_mb,
    measure_latency,
    onnx_runner,
    predict_all,
    preprocess_for_quantization,
    quantize_int8,
    quantize_int8_static,
    torch_runner,
    torch_size_mb,
)
from src.predictor import LayoutCNN


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    model = LayoutCNN(n_outputs=3, channels=(8, 16), image_size=32, hidden=256).eval()
    folder = tmp_path_factory.mktemp("onnx")
    fp32 = export_onnx(model, folder / "m_fp32.onnx", 32)
    int8 = quantize_int8(fp32, folder / "m_int8.onnx")
    return model, fp32, int8


def _images(n: int = 10) -> np.ndarray:
    return (np.random.default_rng(0).random((n, 32, 32)) > 0.7).astype(np.uint8)


def test_onnx_fp32_matches_pytorch(exported) -> None:
    model, fp32, _ = exported
    a = predict_all(torch_runner(model), _images())
    b = predict_all(onnx_runner(fp32, threads=1), _images())
    assert a.shape == b.shape == (10, 3)
    assert np.abs(a - b).max() < 1e-4


def test_onnx_dynamic_batch(exported) -> None:
    _, fp32, _ = exported
    run = onnx_runner(fp32, threads=1)
    assert run(np.zeros((1, 1, 32, 32), np.float32)).shape == (1, 3)
    assert run(np.zeros((7, 1, 32, 32), np.float32)).shape == (7, 3)


def test_int8_is_smaller_and_close(exported) -> None:
    model, fp32, int8 = exported
    assert file_size_mb(int8) < 0.5 * file_size_mb(fp32)
    a = predict_all(torch_runner(model), _images())
    b = predict_all(onnx_runner(int8, threads=1), _images())
    assert np.abs(a - b).max() < 0.1 * (np.abs(a).max() + 1e-6) + 0.05


def test_export_is_self_contained(exported) -> None:
    _, fp32, _ = exported
    assert not list(fp32.parent.glob("*.data"))  # weights are inside the .onnx file
    assert file_size_mb(fp32) == pytest.approx(torch_size_mb(exported[0]), rel=0.2)


def test_latency_and_cpu_name(exported) -> None:
    _, fp32, _ = exported
    out = measure_latency(onnx_runner(fp32, 1), 32, n_warmup=2, n_runs=10, batch_size=4)
    assert 0 < out["latency_mean_ms"] <= out["latency_p95_ms"] * 10
    assert out["batched_ms_per_image"] > 0
    assert isinstance(cpu_name(), str) and cpu_name()


def test_static_int8_close_to_fp32(exported, tmp_path) -> None:
    model, fp32, _ = exported
    pre = preprocess_for_quantization(fp32, tmp_path / "pre.onnx")
    static = quantize_int8_static(pre, tmp_path / "static.onnx", _images(16))
    assert file_size_mb(static) < 0.5 * file_size_mb(fp32)
    a = predict_all(torch_runner(model), _images())
    b = predict_all(onnx_runner(static, threads=1), _images())
    assert b.shape == a.shape and np.abs(a - b).max() < 0.1 * (np.abs(a).max() + 1e-6) + 0.05


def test_dynamic_fc_only_quantizes_linear_layers(exported, tmp_path) -> None:
    import onnx

    _, fp32, int8_all = exported
    fc_only = quantize_int8(fp32, tmp_path / "fc.onnx", op_types=["MatMul", "Gemm"])
    ops_all = {n.op_type for n in onnx.load(str(int8_all)).graph.node}
    ops_fc = {n.op_type for n in onnx.load(str(fc_only)).graph.node}
    assert "ConvInteger" in ops_all and "ConvInteger" not in ops_fc and "Conv" in ops_fc
    assert file_size_mb(fc_only) < file_size_mb(fp32)
