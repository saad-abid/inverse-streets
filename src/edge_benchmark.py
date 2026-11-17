"""Edge efficiency of the CNN predictor: ONNX export, dynamic INT8 quantization, benchmark.

Versions compared:
  pytorch_fp32   the trained PyTorch model
  onnx_fp32      the same model exported to ONNX, run with onnxruntime
  onnx_int8_dynamic     dynamic INT8 quantization of all supported operators (weights stored as
                        8-bit integers, activations quantized on the fly), as in the project plan
  onnx_int8_dynamic_fc  dynamic INT8 on the linear layers only (convolutions stay FP32)
  onnx_int8_static      static INT8 (QDQ, per-channel), activation ranges calibrated on 256
                        training images; the usual method for CNNs
For each version and CPU thread count: test accuracy (MAE, R²), single-image latency (mean and
p95 after warm-up), batched time per image, model file size, largest output difference to
PyTorch, and the device.

Run:  python -m src.edge_benchmark [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import io
import os
import platform
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from src.config import PROJECT_ROOT, load_config
from src.dataset import load_splits
from src.evaluate import metrics_table

Runner = Callable[[np.ndarray], np.ndarray]  # float32 (N, 1, H, W) -> scaled outputs (N, T)


# --------------------------------------------------------------------------- export
def export_onnx(model: torch.nn.Module, path: Path, image_size: int) -> Path:
    """Export to a single self-contained ONNX file with a dynamic batch dimension."""
    model = model.cpu().eval()
    example = torch.zeros(1, 1, image_size, image_size)
    batch = torch.export.Dim("batch")
    torch.onnx.export(model, (example,), str(path), input_names=["image"], output_names=["indicators"],
                      dynamic_shapes={"x": {0: batch}}, dynamo=True, external_data=False, verbose=False)
    return path


def preprocess_for_quantization(fp32_path: Path, out_path: Path) -> Path:
    """Shape inference and graph optimization recommended by onnxruntime before quantizing."""
    quant_pre_process(str(fp32_path), str(out_path))
    return out_path


def quantize_int8(fp32_path: Path, int8_path: Path, op_types: list[str] | None = None) -> Path:
    """Dynamic INT8 quantization (onnxruntime): weights stored as int8, activations quantized
    at run time. `op_types` limits which operators are quantized (None = all supported)."""
    quantize_dynamic(str(fp32_path), str(int8_path), op_types_to_quantize=op_types,
                     weight_type=QuantType.QInt8)
    return int8_path


class ImageCalibrationReader(CalibrationDataReader):
    """Feeds calibration images (uint8, (N, H, W)) one at a time to the static quantizer."""

    def __init__(self, images: np.ndarray, input_name: str = "image") -> None:
        self._items = iter([{input_name: img.astype(np.float32)[None, None]} for img in images])

    def get_next(self) -> dict | None:
        return next(self._items, None)


def quantize_int8_static(fp32_path: Path, int8_path: Path, calibration_images: np.ndarray) -> Path:
    """Static INT8 quantization (QDQ format, per-channel weights): activation ranges are
    measured once on calibration images, so integer kernels can be used for convolutions."""
    quantize_static(str(fp32_path), str(int8_path), ImageCalibrationReader(calibration_images),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    return int8_path


def file_size_mb(path: Path) -> float:
    """File size in megabytes (10^6 bytes)."""
    return os.path.getsize(path) / 1e6


def torch_size_mb(model: torch.nn.Module) -> float:
    """Size of the serialized PyTorch weights (state_dict) in megabytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1e6


def cpu_name() -> str:
    """Readable CPU name (Windows registry, Linux /proc/cpuinfo, else platform info)."""
    try:
        if platform.system() == "Windows":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for row in f:
                if row.startswith("model name"):
                    return row.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


# --------------------------------------------------------------------------- runners
def torch_runner(model: torch.nn.Module) -> Runner:
    """Run the PyTorch model on CPU without gradients."""
    model = model.cpu().eval()

    @torch.no_grad()
    def run(x: np.ndarray) -> np.ndarray:
        return model(torch.from_numpy(x)).numpy()

    return run


def onnx_runner(path: Path, threads: int) -> Runner:
    """Run an ONNX model with onnxruntime on CPU with a fixed number of threads."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return lambda x: session.run(None, {name: x})[0]


def predict_all(run: Runner, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Scaled outputs for uint8 images (N, H, W)."""
    x = images.astype(np.float32)[:, None]
    return np.concatenate([run(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def measure_latency(run: Runner, image_size: int, n_warmup: int = 50, n_runs: int = 300,
                    batch_size: int = 64) -> dict[str, float]:
    """Single-image latency (mean, p95) after warm-up, and batched time per image, in ms."""
    x1 = np.zeros((1, 1, image_size, image_size), dtype=np.float32)
    for _ in range(n_warmup):
        run(x1)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run(x1)
        times.append((time.perf_counter() - start) * 1000)
    xb = np.zeros((batch_size, 1, image_size, image_size), dtype=np.float32)
    run(xb)
    start = time.perf_counter()
    for _ in range(5):
        run(xb)
    batched = (time.perf_counter() - start) * 1000 / (5 * batch_size)
    return {"latency_mean_ms": float(np.mean(times)), "latency_p95_ms": float(np.percentile(times, 95)),
            "batched_ms_per_image": batched}


# --------------------------------------------------------------------------- benchmark
def benchmark(cfg: dict, thread_counts: tuple[int, ...] = (1, 4)) -> pd.DataFrame:
    """Export, quantize and benchmark all versions. Returns one row per (version, threads)."""
    from src.predictor import load_predictor

    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]
    model, scaler, targets = load_predictor(models_dir / "predictor.pt")
    size = cfg["dataset"]["image_size"]
    splits = load_splits(cfg, targets)
    test = splits["test"]
    calibration = splits["train"].images[:256]  # never test images

    fp32_path = export_onnx(model, models_dir / "predictor_fp32.onnx", size)
    pre_path = preprocess_for_quantization(fp32_path, models_dir / "predictor_fp32_pre.onnx")
    paths = {
        "onnx_fp32": fp32_path,
        "onnx_int8_dynamic": quantize_int8(pre_path, models_dir / "predictor_int8_dynamic.onnx"),
        "onnx_int8_dynamic_fc": quantize_int8(pre_path, models_dir / "predictor_int8_dynamic_fc.onnx",
                                              op_types=["MatMul", "Gemm"]),
        "onnx_int8_static": quantize_int8_static(pre_path, models_dir / "predictor_int8_static.onnx",
                                                 calibration),
    }
    sizes = {"pytorch_fp32": torch_size_mb(model), **{k: file_size_mb(p) for k, p in paths.items()}}

    reference = scaler.inverse(predict_all(torch_runner(model), test.images))
    device, rows = f"CPU: {cpu_name()}", []
    default_threads = torch.get_num_threads()
    for threads in thread_counts:
        torch.set_num_threads(threads)
        runners = {"pytorch_fp32": torch_runner(model),
                   **{k: onnx_runner(p, threads) for k, p in paths.items()}}
        for version, run in runners.items():
            pred = scaler.inverse(predict_all(run, test.images))
            metrics = metrics_table(test.targets, {version: pred}, targets).set_index("indicator")
            row = {"version": version, "threads": threads, "size_mb": sizes[version],
                   **measure_latency(run, size),
                   **{f"mae_{t}": metrics.loc[t, "mae"] for t in targets},
                   **{f"r2_{t}": metrics.loc[t, "r2"] for t in targets},
                   "mean_r2": float(metrics["r2"].mean()),
                   "max_abs_diff_vs_pytorch": float(np.abs(pred - reference).max()),
                   "device": device, "onnxruntime": ort.__version__, "torch": torch.__version__}
            rows.append(row)
            print(f"{version:21s} threads={threads}  size {row['size_mb']:.3f} MB  "
                  f"latency {row['latency_mean_ms']:.3f} ms (p95 {row['latency_p95_ms']:.3f})  "
                  f"mean R² {row['mean_r2']:.4f}")
    torch.set_num_threads(default_threads)
    return pd.DataFrame(rows)


def plot_tradeoffs(table: pd.DataFrame, path: Path) -> None:
    """Accuracy (mean test R²) vs single-image latency, and vs model size."""
    colours = {"pytorch_fp32": "tab:blue", "onnx_fp32": "tab:orange", "onnx_int8_dynamic": "tab:red",
               "onnx_int8_dynamic_fc": "tab:purple", "onnx_int8_static": "tab:green"}
    markers = {1: "o", 4: "s"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for _, r in table.iterrows():
        axes[0].scatter(r["latency_mean_ms"], r["mean_r2"], color=colours[r["version"]],
                        marker=markers.get(r["threads"], "^"), s=60,
                        label=f"{r['version']}, {r['threads']} thread(s)")
    one = table[table["threads"] == table["threads"].min()]
    for _, r in one.iterrows():
        axes[1].scatter(r["size_mb"], r["mean_r2"], color=colours[r["version"]], s=60, label=r["version"])
        below = r["version"] == "pytorch_fp32"  # avoids overlapping the onnx_fp32 label
        axes[1].annotate(f"{r['size_mb']:.2f} MB", (r["size_mb"], r["mean_r2"]), fontsize=8,
                         xytext=(-30, -14) if below else (5, 5), textcoords="offset points")
    axes[0].set_xlabel("single-image latency, mean (ms)")
    axes[1].set_xlabel("model file size (MB)")
    for ax in axes:
        ax.set_ylabel("mean test R² (4 indicators)")
        ax.legend(fontsize=7)
    fig.suptitle(f"CNN predictor on {table['device'].iloc[0]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--plot-only", action="store_true", help="redraw the chart from the saved CSV")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    if args.plot_only:
        plot_tradeoffs(pd.read_csv(results_dir / "edge_benchmark.csv"), results_dir / "edge_benchmark.png")
        return
    table = benchmark(cfg)
    table.to_csv(results_dir / "edge_benchmark.csv", index=False)
    plot_tradeoffs(table, results_dir / "edge_benchmark.png")
    cols = ["version", "threads", "size_mb", "latency_mean_ms", "latency_p95_ms", "batched_ms_per_image",
            "mean_r2", "r2_congestion_index", "max_abs_diff_vs_pytorch"]
    print("\n" + table[cols].round(4).to_string(index=False))
    print(table["device"].iloc[0])


if __name__ == "__main__":
    main()
