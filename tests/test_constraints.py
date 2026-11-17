"""Tests for the consistency rules (Phase 6), on a small hand-made label table."""

import numpy as np
import pandas as pd
import pytest

from src.constraints import ConstraintChecker, RuleResult, all_passed, format_results

CFG = {"density_min": 0.10, "density_max": 0.60, "min_connectivity": 0.95, "density_bin": 0.05,
       "intersections_percentiles": [0, 100], "support_radius": 0.25, "min_support": 3}


def _labels() -> pd.DataFrame:
    """Sparse layouts (density 0.10-0.14) have 2-6 intersections, dense ones (0.40-0.44)
    have 30-40; congestion is 4 for sparse and 2-3 for dense layouts."""
    rng = np.random.default_rng(0)
    sparse = pd.DataFrame({"road_density": rng.uniform(0.10, 0.14, 50),
                           "n_intersections": rng.integers(2, 7, 50),
                           "congestion_index": np.full(50, 4.0)})
    dense = pd.DataFrame({"road_density": rng.uniform(0.40, 0.44, 50),
                          "n_intersections": rng.integers(30, 41, 50),
                          "congestion_index": rng.uniform(2.0, 3.0, 50)})
    test_row = pd.DataFrame({"road_density": [0.9], "n_intersections": [99], "congestion_index": [1.0]})
    df = pd.concat([sparse.assign(split="train"), dense.assign(split="train"),
                    test_row.assign(split="test")], ignore_index=True)
    return df


@pytest.fixture
def checker() -> ConstraintChecker:
    return ConstraintChecker(_labels(), CFG)


def _failed(results: list[RuleResult]) -> set[str]:
    return {r.rule for r in results if not r.passed}


def test_consistent_targets_pass(checker: ConstraintChecker) -> None:
    results = checker.check_targets({"road_density": 0.42, "intersections": 35})
    assert all_passed(results), format_results(results)


def test_density_out_of_range(checker: ConstraintChecker) -> None:
    assert "density_in_range" in _failed(checker.check_targets({"road_density": 0.05, "intersections": 3}))
    assert "density_in_range" in _failed(checker.check_targets({"road_density": 0.70, "intersections": 3}))


def test_low_density_many_intersections_rejected(checker: ConstraintChecker) -> None:
    results = checker.check_targets({"road_density": 0.12, "intersections": 40})
    failed = _failed(results)
    assert {"intersections_vs_density", "combination_in_dataset"} <= failed
    msg = next(r.message for r in results if r.rule == "intersections_vs_density")
    assert "too many intersections" in msg


def test_density_with_no_data_is_rejected(checker: ConstraintChecker) -> None:
    results = checker.check_targets({"road_density": 0.27, "intersections": 15})
    assert {"intersections_vs_density", "combination_in_dataset"} <= _failed(results)


def test_test_split_is_not_used(checker: ConstraintChecker) -> None:
    # the only layout with density 0.9 is in the test split -> no data
    assert "intersections_vs_density" in _failed(checker.check_targets({"road_density": 0.9, "intersections": 99}))


def test_congestion_reachable_and_unreachable(checker: ConstraintChecker) -> None:
    ok = checker.check_targets({"road_density": 0.42, "intersections": 35, "max_congestion_index": 2.5})
    bad = checker.check_targets({"road_density": 0.42, "intersections": 35, "max_congestion_index": 1.5})
    assert all_passed(ok)
    assert _failed(bad) == {"congestion_reachable"}
    assert "lowest" in bad[-1].message


def test_no_congestion_limit_passes(checker: ConstraintChecker) -> None:
    assert checker.congestion_reachable({"road_density": 0.42, "intersections": 35}).passed


def test_layout_rules(checker: ConstraintChecker) -> None:
    good = {"road_density": 0.3, "connectivity": 1.0, "congestion_index": 2.0}
    broken = {"road_density": 0.3, "connectivity": 0.6, "congestion_index": 5.0}
    spec = {"max_congestion_index": 3.0}
    assert all_passed(checker.check_layout(good, spec))
    assert _failed(checker.check_layout(broken, spec)) == {"layout_connected", "layout_congestion_below_max"}
    assert len(checker.check_layout(good)) == 2  # no congestion rule without a limit


def test_layout_with_nan_congestion_fails(checker: ConstraintChecker) -> None:
    ind = {"road_density": 0.3, "connectivity": 1.0, "congestion_index": float("nan")}
    assert "layout_congestion_below_max" in _failed(checker.check_layout(ind, {"max_congestion_index": 3.0}))


def test_format_results() -> None:
    text = format_results([RuleResult("a", True, "fine"), RuleResult("b", False, "bad")])
    assert text == "[PASS] a: fine\n[FAIL] b: bad"
