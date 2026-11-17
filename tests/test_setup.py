"""Phase 0 smoke tests: config loads and seeding is reproducible."""

import numpy as np

from src.config import load_config, set_seed


def test_config_loads() -> None:
    cfg = load_config()
    assert cfg["dataset"]["image_size"] == 64
    splits = cfg["dataset"]["splits"]
    assert abs(splits["train"] + splits["val"] + splits["test"] - 1.0) < 1e-9


def test_seed_is_reproducible() -> None:
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)
