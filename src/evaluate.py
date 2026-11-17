"""Evaluation metrics.

Phase 3: regression metrics for the predictor (MAE, RMSE, R² per indicator).
Phase 5 will add the metrics for generated layouts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and R² for one indicator. R² is NaN if the true values are constant."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan"),
    }


def metrics_table(y_true: np.ndarray, predictions: dict[str, np.ndarray],
                  target_names: list[str]) -> pd.DataFrame:
    """One row per (model, indicator) with MAE, RMSE and R²."""
    rows = []
    for model_name, y_pred in predictions.items():
        for j, target in enumerate(target_names):
            rows.append({"model": model_name, "indicator": target,
                         **regression_metrics(y_true[:, j], y_pred[:, j])})
    return pd.DataFrame(rows)
