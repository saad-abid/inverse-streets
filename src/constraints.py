"""Physical / urban consistency rules for generation targets and for generated layouts.

Except for the fixed density range, every limit is measured on the TRAINING layouts, so a
target is accepted only if the synthetic dataset contains similar layouts.

Rules on a target specification (checked BEFORE generation):
  1. density_in_range              road density between density_min and density_max
  2. intersections_vs_density      intersections inside the range seen for similar densities
  3. combination_in_dataset        enough training layouts close to (density, intersections)
  4. congestion_reachable          some similar training layouts have congestion <= the maximum
Rules on a generated layout (checked AFTER generation, with the graph method):
  5. layout_density_in_range, 6. layout_connected, 7. layout_congestion_below_max
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


@dataclass
class RuleResult:
    """Outcome of one rule: its name, pass/fail and a message for the user."""
    rule: str
    passed: bool
    message: str


def all_passed(results: list[RuleResult]) -> bool:
    """True if every rule passed."""
    return all(r.passed for r in results)


def format_results(results: list[RuleResult]) -> str:
    """One line per rule: PASS/FAIL, name, message."""
    return "\n".join(f"[{'PASS' if r.passed else 'FAIL'}] {r.rule}: {r.message}" for r in results)


class ConstraintChecker:
    """Checks targets and layouts against fixed limits and the training-data distribution."""

    def __init__(self, labels: pd.DataFrame, cfg: dict) -> None:
        train = labels[labels["split"] == "train"] if "split" in labels else labels
        self.density = train["road_density"].to_numpy(float)
        self.intersections = train["n_intersections"].to_numpy(float)
        self.congestion = train["congestion_index"].to_numpy(float)
        self.cfg = cfg
        self._scale = np.array([self.density.std(), self.intersections.std()])

    @classmethod
    def from_config(cls, cfg: dict) -> "ConstraintChecker":
        """Build from data/labels.csv and the `constraints` config section."""
        labels = pd.read_csv(Path(PROJECT_ROOT) / cfg["paths"]["data_dir"] / "labels.csv")
        return cls(labels, cfg["constraints"])

    # ----------------------------------------------------------------- helpers
    def _similar(self, density: float, intersections: float) -> np.ndarray:
        """Boolean mask of training layouts within the support radius (standardized units)."""
        diff = np.stack([self.density - density, self.intersections - intersections], axis=1)
        return np.linalg.norm(diff / self._scale, axis=1) <= self.cfg["support_radius"]

    # ----------------------------------------------------------------- target rules
    def density_in_range(self, spec: dict) -> RuleResult:
        d, lo, hi = spec["road_density"], self.cfg["density_min"], self.cfg["density_max"]
        ok = lo <= d <= hi
        msg = f"road density {d:.2f} is {'inside' if ok else 'outside'} the allowed range [{lo:.2f}, {hi:.2f}]"
        return RuleResult("density_in_range", ok, msg)

    def intersections_vs_density(self, spec: dict) -> RuleResult:
        d, n = spec["road_density"], spec["intersections"]
        near = np.abs(self.density - d) <= self.cfg["density_bin"] / 2
        if near.sum() < self.cfg["min_support"]:
            return RuleResult("intersections_vs_density", False,
                              f"the dataset has almost no layouts with road density near {d:.2f}, "
                              f"so no intersection count can be checked")
        p_lo, p_hi = self.cfg["intersections_percentiles"]
        lo, hi = np.percentile(self.intersections[near], [p_lo, p_hi])
        ok = lo <= n <= hi
        msg = (f"layouts with road density {d:.2f} +/- {self.cfg['density_bin'] / 2:.3f} have between "
               f"{lo:.0f} and {hi:.0f} intersections (percentiles p{p_lo}-p{p_hi}); {n} requested")
        if not ok:
            msg += " -> inconsistent: " + ("too many intersections for this density" if n > hi
                                            else "too few intersections for this density")
        return RuleResult("intersections_vs_density", ok, msg)

    def combination_in_dataset(self, spec: dict) -> RuleResult:
        count = int(self._similar(spec["road_density"], spec["intersections"]).sum())
        ok = count >= self.cfg["min_support"]
        msg = (f"{count} training layouts are close to (density {spec['road_density']:.2f}, "
               f"{spec['intersections']} intersections); at least {self.cfg['min_support']} needed")
        if not ok:
            msg += " -> this combination is (almost) never seen in the dataset"
        return RuleResult("combination_in_dataset", ok, msg)

    def congestion_reachable(self, spec: dict) -> RuleResult:
        limit = spec.get("max_congestion_index")
        if limit is None:
            return RuleResult("congestion_reachable", True, "no congestion limit requested")
        similar = self._similar(spec["road_density"], spec["intersections"])
        if not similar.any():
            return RuleResult("congestion_reachable", False,
                              "no similar layouts in the dataset, so the congestion limit cannot be checked")
        values = self.congestion[similar]
        n_ok = int((values <= limit).sum())
        msg = (f"{n_ok} of {similar.sum()} similar training layouts have congestion index <= {limit:.2f} "
               f"(lowest {values.min():.2f}, median {np.median(values):.2f})")
        if n_ok == 0:
            msg += " -> the requested congestion is lower than anything seen for this structure"
        return RuleResult("congestion_reachable", n_ok > 0, msg)

    def check_targets(self, spec: dict) -> list[RuleResult]:
        """All target rules, in order."""
        return [self.density_in_range(spec), self.intersections_vs_density(spec),
                self.combination_in_dataset(spec), self.congestion_reachable(spec)]

    # ----------------------------------------------------------------- layout rules
    def check_layout(self, indicators: dict, spec: dict | None = None) -> list[RuleResult]:
        """Rules for a generated layout, using indicators recomputed with the graph method."""
        d, lo, hi = indicators["road_density"], self.cfg["density_min"], self.cfg["density_max"]
        conn, min_conn = indicators["connectivity"], self.cfg["min_connectivity"]
        results = [
            RuleResult("layout_density_in_range", lo <= d <= hi,
                       f"road density {d:.2f}, allowed [{lo:.2f}, {hi:.2f}]"),
            RuleResult("layout_connected", bool(conn >= min_conn),
                       f"connectivity {conn:.2f} (share of nodes in the largest network), "
                       f"required >= {min_conn:.2f}"),
        ]
        limit = (spec or {}).get("max_congestion_index")
        if limit is not None:
            c = indicators["congestion_index"]
            results.append(RuleResult("layout_congestion_below_max", bool(np.isfinite(c) and c <= limit),
                                      f"congestion index {c:.2f}, maximum {limit:.2f}"))
        return results
