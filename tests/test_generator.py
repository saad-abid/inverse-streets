"""Tests for the conditional VAE, its loss, sampling and post-processing (Phase 4)."""

import numpy as np
import pytest
import torch

from src.config import set_seed
from src.dataset import LayoutDataset, TargetScaler
from src.generator import (
    ConditionalVAE,
    beta_schedule,
    calibrate_threshold,
    generate_layouts,
    load_generator,
    postprocess,
    sample_probabilities,
    train_generator,
    vae_loss,
)

GEN_CFG = {"threshold": 0.5, "max_fragment_size": 10, "max_hole_size": 4}


def _small_model() -> ConditionalVAE:
    return ConditionalVAE(n_cond=2, latent_dim=4, channels=(8, 8, 8, 8))


def test_shapes() -> None:
    model = _small_model()
    logits, mu, logvar = model(torch.zeros(3, 1, 64, 64), torch.zeros(3, 2))
    assert logits.shape == (3, 1, 64, 64) and mu.shape == (3, 4) and logvar.shape == (3, 4)


def test_kl_is_zero_for_standard_normal_posterior() -> None:
    x = torch.zeros(2, 1, 8, 8)
    _, _, kl = vae_loss(torch.zeros(2, 1, 8, 8), x, torch.zeros(2, 4), torch.zeros(2, 4), beta=1.0)
    assert kl.item() == pytest.approx(0.0)


def test_loss_combines_terms() -> None:
    x = torch.ones(1, 1, 4, 4)
    loss, recon, kl = vae_loss(torch.zeros(1, 1, 4, 4), x, torch.ones(1, 2), torch.zeros(1, 2), beta=0.5)
    assert recon.item() == pytest.approx(16 * np.log(2), rel=1e-5)  # BCE of p=0.5 per pixel
    assert kl.item() == pytest.approx(1.0)                            # 0.5 * sum(mu^2)
    assert loss.item() == pytest.approx(recon.item() + 0.5 * kl.item())


def test_beta_schedule() -> None:
    assert [beta_schedule(e, 1.0, 4) for e in (1, 2, 4, 9)] == [0.25, 0.5, 1.0, 1.0]
    assert beta_schedule(1, 2.0, 0) == 2.0


def test_postprocess_removes_small_fragment_and_fills_hole() -> None:
    probs = np.zeros((20, 20))
    probs[5, 0:20] = 0.9                 # a real road (20 px)
    probs[15, 15] = probs[15, 16] = 0.9  # tiny fragment (2 px)
    probs[4:7, 8:11] = 0.9
    probs[5, 9] = 0.1                    # 1-pixel hole inside a road block
    out = postprocess(probs, threshold=0.5, max_fragment=10, max_hole=4)
    assert out.dtype == np.uint8 and set(np.unique(out)) <= {0, 1}
    assert out[5].sum() == 20 and out[15, 15] == 0 and out[5, 9] == 1


def test_postprocess_batch() -> None:
    assert postprocess(np.random.default_rng(0).random((3, 16, 16))).shape == (3, 16, 16)


def test_sampling_is_reproducible_and_binary() -> None:
    model = _small_model()
    scaler = TargetScaler(mean=np.zeros(2), std=np.ones(2))
    a = sample_probabilities(model, np.array([0.2, 10.0]), scaler, n=3, seed=7)
    b = sample_probabilities(model, np.array([0.2, 10.0]), scaler, n=3, seed=7)
    assert a.shape == (3, 64, 64) and np.array_equal(a, b) and ((a >= 0) & (a <= 1)).all()
    imgs = generate_layouts(model, np.array([0.2, 10.0]), scaler, 3, 7, GEN_CFG)
    assert set(np.unique(imgs)) <= {0, 1}


def test_checkpoint_roundtrip(tmp_path) -> None:
    model = _small_model()
    scaler = TargetScaler(mean=np.array([0.2, 20.0]), std=np.array([0.1, 10.0]))
    path = tmp_path / "gen.pt"
    torch.save({"state_dict": model.state_dict(), "scaler": scaler.to_dict(),
                "conditions": ["a", "b"], "latent_dim": 4, "channels": [8, 8, 8, 8],
                "image_size": 64}, path)
    loaded, loaded_scaler, conditions, threshold = load_generator(path)
    target = np.array([0.3, 25.0])
    assert conditions == ["a", "b"] and threshold == 0.5  # default when not calibrated
    assert np.array_equal(sample_probabilities(model, target, scaler, 2, 0),
                          sample_probabilities(loaded, target, loaded_scaler, 2, 0))


def test_training_reduces_reconstruction_loss() -> None:
    set_seed(0)
    images = np.zeros((32, 64, 64), dtype=np.uint8)
    for i, img in enumerate(images):
        img[:, (i % 8) * 8] = 1  # one vertical road at different positions
    cond = images.reshape(32, -1).mean(axis=1, keepdims=True).repeat(2, axis=1).astype(np.float32)
    ds = LayoutDataset(images, cond)
    cfg = {"batch_size": 8, "learning_rate": 3e-3, "epochs": 6, "beta": 1.0, "kl_warmup_epochs": 2}
    history = train_generator(_small_model(), ds, ds, cfg, "cpu", seed=0)
    assert history[-1]["train_recon"] < history[0]["train_recon"]


def test_calibrate_threshold_picks_density_match(monkeypatch) -> None:
    import pandas as pd

    import src.generator as gen
    from src.dataset import SplitData

    # Probabilities evenly spread over [0, 1): density at threshold t is exactly 1 - t.
    uniform = np.arange(64 * 64).reshape(1, 64, 64) / (64 * 64)
    monkeypatch.setattr(gen, "sample_probabilities", lambda *a, **k: uniform)
    targets = np.array([[0.6, 10.0]] * 5)
    val = SplitData(np.zeros((5, 64, 64), np.uint8), targets, pd.DataFrame({"id": range(5)}))
    best, table = calibrate_threshold(None, None, val, ["road_density", "n_intersections"],
                                      [0.2, 0.4, 0.6], seed=0)
    assert list(table["threshold"]) == [0.2, 0.4, 0.6]
    assert best == 0.4
    assert table["density_mae"].tolist() == pytest.approx([0.2, 0.0, 0.2], abs=1e-3)
