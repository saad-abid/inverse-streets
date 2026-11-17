"""Tests for text request -> JSON -> validation, with a mocked LLM (run offline)."""

import json

import numpy as np

from src.constraints import ConstraintChecker
from src.llm_spec import build_system_prompt, indicator_stats, interpret_request, parse_spec
from tests.test_constraints import CFG, _labels


def _mock_llm(answer: str):
    """Fake LLM that always returns `answer` and records what it was asked."""
    calls = []

    def llm(system: str, user: str) -> str:
        calls.append((system, user))
        return answer

    llm.calls = calls
    return llm


def _run(answer: str, request: str = "some request"):
    labels = _labels()
    return interpret_request(request, _mock_llm(answer), ConstraintChecker(labels, CFG), indicator_stats(labels))


def test_consistent_request_is_accepted() -> None:
    res = _run(json.dumps({"road_density": 0.42, "intersections": 35, "max_congestion_index": None}),
               "a dense neighbourhood with many intersections")
    assert res.accepted and res.spec["intersections"] == 35


def test_inconsistent_request_is_rejected_with_explanation() -> None:
    res = _run(json.dumps({"road_density": 0.12, "intersections": 40, "max_congestion_index": None}),
               "very low road density but 40 intersections")
    assert not res.accepted
    failed = {c.rule: c.message for c in res.checks if not c.passed}
    assert "intersections_vs_density" in failed and "too many intersections" in failed["intersections_vs_density"]


def test_unreachable_congestion_is_rejected() -> None:
    res = _run(json.dumps({"road_density": 0.42, "intersections": 35, "max_congestion_index": 1.2}))
    assert not res.accepted
    assert [c.rule for c in res.checks if not c.passed] == ["congestion_reachable"]


def test_invalid_json_is_rejected() -> None:
    res = _run("road density about 0.3")
    assert not res.accepted and res.spec is None and res.checks[0].rule == "valid_json"


def test_schema_violations_are_rejected() -> None:
    for bad in ({"road_density": 1.5, "intersections": 10, "max_congestion_index": None},
                {"road_density": 0.3, "intersections": 10.5, "max_congestion_index": None},
                {"road_density": 0.3, "intersections": 10},
                {"road_density": 0.3, "intersections": 10, "max_congestion_index": None, "style": "grid"}):
        res = _run(json.dumps(bad))
        assert not res.accepted and res.checks[0].rule == "json_schema", bad


def test_llm_failure_is_rejected_not_raised() -> None:
    def broken(system: str, user: str) -> str:
        raise ConnectionError("server down")

    labels = _labels()
    res = interpret_request("x", broken, ConstraintChecker(labels, CFG), indicator_stats(labels))
    assert not res.accepted and "server down" in res.checks[0].message


def test_parse_spec_accepts_null_congestion() -> None:
    spec, checks = parse_spec('{"road_density": 0.2, "intersections": 8, "max_congestion_index": null}')
    assert spec == {"road_density": 0.2, "intersections": 8, "max_congestion_index": None}
    assert checks[0].passed


def test_prompt_contains_dataset_ranges_and_no_repair_rule() -> None:
    labels = _labels()
    stats = indicator_stats(labels)
    prompt = build_system_prompt(stats)
    assert f"max {stats['road_density']['max']:.2f}" in prompt
    assert "Do NOT change values" in prompt


def test_indicator_stats_use_train_split_only() -> None:
    stats = indicator_stats(_labels())
    assert stats["road_density"]["max"] < 0.5  # the 0.9 layout is in the test split
    assert np.isclose(stats["congestion_index"]["max"], 4.0)


def test_request_is_sent_to_llm() -> None:
    labels = _labels()
    llm = _mock_llm('{"road_density": 0.42, "intersections": 35, "max_congestion_index": null}')
    interpret_request("dense please", llm, ConstraintChecker(labels, CFG), indicator_stats(labels))
    assert llm.calls[0][1] == "dense please"
