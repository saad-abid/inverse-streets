"""Text request -> validated JSON targets -> consistency rules -> (only then) generation.

A local LLM (through Ollama) turns a request such as "a dense neighbourhood with many
intersections and little congestion" into
    {"road_density": ..., "intersections": ..., "max_congestion_index": ... or null}.
The JSON is validated with a JSON schema, then checked with `constraints.py` BEFORE any
layout is generated. Inconsistent requests are rejected with an explanation. The LLM is
told NOT to repair inconsistencies: that is the job of the explicit rules.

The LLM is optional: everything takes an `llm(system, user) -> str` function, so tests use a
mock and the rest of the project runs without Ollama.

Run:  python -m src.llm_spec "a sparse suburb with few intersections"
      python -m src.llm_spec --examples        # fixed list of consistent / inconsistent requests
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import jsonschema
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.constraints import ConstraintChecker, RuleResult, all_passed, format_results

LLM = Callable[[str, str], str]

SPEC_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "road_density": {"type": "number", "minimum": 0, "maximum": 1},
        "intersections": {"type": "integer", "minimum": 0, "maximum": 500},
        "max_congestion_index": {"type": ["number", "null"], "minimum": 1},
    },
    "required": ["road_density", "intersections", "max_congestion_index"],
    "additionalProperties": False,
}

EXAMPLE_REQUESTS: list[tuple[str, str]] = [  # (request, expected decision)
    ("a dense neighbourhood with many intersections and little congestion", "accept"),
    ("a sparse suburb with few intersections", "accept"),
    ("a medium density grid with about 20 intersections", "accept"),
    ("road density 0.3 and 25 intersections", "accept"),
    ("very low road density but 40 intersections", "reject"),
    ("road density 0.05 with 3 intersections", "reject"),
    ("a very dense city with almost no intersections", "reject"),
    ("road density 0.4 with 35 intersections and a congestion index below 1.3", "reject"),
]


# --------------------------------------------------------------------------- prompt
def indicator_stats(labels: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Low / typical / high (10th / 50th / 90th percentile) and min / max of each target
    indicator on the training layouts, to ground words like 'dense' or 'many'."""
    train = labels[labels["split"] == "train"]
    out = {}
    for col in ("road_density", "n_intersections", "congestion_index"):
        v = train[col].to_numpy(float)
        out[col] = {"min": v.min(), "low": np.percentile(v, 10), "typical": np.percentile(v, 50),
                    "high": np.percentile(v, 90), "max": v.max()}
    return out


def build_system_prompt(stats: dict[str, dict[str, float]]) -> str:
    """Instructions for the LLM, with the value ranges of the dataset."""
    d, n, c = stats["road_density"], stats["n_intersections"], stats["congestion_index"]
    return (
        "You translate a request for a synthetic 64x64 street layout into JSON targets.\n"
        "Fields:\n"
        f"- road_density: share of road pixels. Dataset: min {d['min']:.2f}, low {d['low']:.2f}, "
        f"typical {d['typical']:.2f}, high {d['high']:.2f}, max {d['max']:.2f}.\n"
        f"- intersections: integer number of road intersections. Dataset: min {n['min']:.0f}, "
        f"low {n['low']:.0f}, typical {n['typical']:.0f}, high {n['high']:.0f}, max {n['max']:.0f}.\n"
        f"- max_congestion_index: upper limit for max/mean trips per road segment, or null if the "
        f"request says nothing about congestion or traffic. Dataset: min {c['min']:.2f}, "
        f"low {c['low']:.2f}, typical {c['typical']:.2f}, high {c['high']:.2f}.\n"
        "Rules:\n"
        "- Words like very low / low / medium / high / very high map to values around "
        "min / low / typical / high / max.\n"
        "- Copy every number given in the request exactly.\n"
        "- Do NOT change values to make the request consistent or realistic; another program "
        "checks consistency and must see the request as the user wrote it.\n"
        "- Answer with the JSON object only."
    )


# --------------------------------------------------------------------------- LLM client
def ollama_llm(cfg: dict) -> LLM:
    """LLM function that calls a local Ollama server with structured JSON output."""
    import requests

    def call(system: str, user: str) -> str:
        payload = {"model": cfg["model"], "stream": False, "think": False, "format": SPEC_SCHEMA,
                   "options": {"temperature": cfg["temperature"], "seed": 0,
                               "num_gpu": cfg.get("num_gpu", 0)},
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        response = requests.post(cfg["url"], json=payload, timeout=cfg["timeout_s"])
        response.raise_for_status()
        return response.json()["message"]["content"]

    return call


# --------------------------------------------------------------------------- parsing
def parse_spec(text: str) -> tuple[dict | None, list[RuleResult]]:
    """Decode the LLM answer and validate it with the JSON schema."""
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as err:
        return None, [RuleResult("valid_json", False, f"the LLM answer is not valid JSON ({err.msg})")]
    errors = sorted(jsonschema.Draft202012Validator(SPEC_SCHEMA).iter_errors(spec), key=str)
    if errors:
        msgs = "; ".join(f"{'/'.join(map(str, e.path)) or 'object'}: {e.message}" for e in errors)
        return None, [RuleResult("json_schema", False, msgs)]
    return spec, [RuleResult("json_schema", True, "the answer matches the JSON schema")]


@dataclass
class RequestResult:
    """Everything that happened to one text request."""
    request: str
    llm_answer: str
    spec: dict | None
    checks: list[RuleResult] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def accepted(self) -> bool:
        return self.spec is not None and all_passed(self.checks)


def interpret_request(request: str, llm: LLM, checker: ConstraintChecker,
                      stats: dict[str, dict[str, float]]) -> RequestResult:
    """Text -> LLM -> JSON schema -> consistency rules. Never generates anything."""
    start = time.perf_counter()
    try:
        answer = llm(build_system_prompt(stats), request)
    except Exception as err:  # network or server problem: reject, do not crash
        return RequestResult(request, "", None, [RuleResult("llm_call", False, f"LLM call failed: {err}")])
    spec, checks = parse_spec(answer)
    if spec is not None:
        checks += checker.check_targets(spec)
    return RequestResult(request, answer, spec, checks, time.perf_counter() - start)


# --------------------------------------------------------------------------- generation
def generate_for_spec(spec: dict, cfg: dict, checker: ConstraintChecker,
                      n: int, seed: int) -> tuple[np.ndarray, list[list[RuleResult]], pd.DataFrame]:
    """Generate n layouts for an ACCEPTED spec, recompute their indicators with the graph
    method and check every layout. Returns images, per-layout checks and indicators."""
    from src.build_labels import layout_indicators
    from src.generator import generate_layouts, load_generator

    model, scaler, conditions, threshold = load_generator(
        PROJECT_ROOT / cfg["paths"]["models_dir"] / "generator.pt")
    values = {"road_density": spec["road_density"], "n_intersections": spec["intersections"]}
    target = np.array([values[c] for c in conditions], dtype=float)
    images = generate_layouts(model, target, scaler, n, seed, {**cfg["generator"], "threshold": threshold})
    indicators = pd.DataFrame([layout_indicators(img, seed, i, cfg) for i, img in enumerate(images)])
    checks = [checker.check_layout(row, spec) for row in indicators.to_dict("records")]
    return images, checks, indicators


# --------------------------------------------------------------------------- reporting
def run_examples(cfg: dict, llm: LLM, checker: ConstraintChecker, stats: dict,
                 results_dir: Path) -> pd.DataFrame:
    """Run the example requests end to end and save a CSV and a figure."""
    rows, panels = [], []
    for k, (request, expected) in enumerate(EXAMPLE_REQUESTS):
        res = interpret_request(request, llm, checker, stats)
        row = {"request": request, "expected": expected,
               "decision": "accept" if res.accepted else "reject",
               "llm_json": res.llm_answer.strip(), "llm_seconds": round(res.seconds, 1),
               "failed_rules": "; ".join(c.rule for c in res.checks if not c.passed)}
        print(f"\n> {request}\n  LLM: {row['llm_json']}")
        print("  " + format_results(res.checks).replace("\n", "\n  "))
        if res.accepted:
            images, checks, ind = generate_for_spec(res.spec, cfg, checker, cfg["llm"]["n_generate"],
                                                    cfg["seed"] + 20_000 + k)
            valid = [all_passed(c) for c in checks]
            row.update({"n_generated": len(images), "n_valid_layouts": int(sum(valid)),
                        "mean_generated_density": ind["road_density"].mean(),
                        "mean_generated_intersections": ind["n_intersections"].mean()})
            print(f"  generated {len(images)} layouts, {sum(valid)} pass all layout rules")
            panels.append((request, images, ind, valid))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "llm_requests.csv", index=False)
    if panels:
        plot_request_layouts(panels, results_dir / "llm_request_layouts.png")
    return table


def plot_request_layouts(panels: list, path: Path, n_cols: int = 6) -> None:
    """For each accepted request: generated layouts, valid ones first (green title = valid)."""
    fig, axes = plt.subplots(len(panels), n_cols, figsize=(1.8 * n_cols, 2.2 * len(panels)), squeeze=False)
    for r, (request, images, ind, valid) in enumerate(panels):
        order = sorted(range(len(images)), key=lambda i: not valid[i])[:n_cols]
        for c, i in enumerate(order):
            ax = axes[r, c]
            ax.imshow(images[i], cmap="gray_r", interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"d={ind['road_density'][i]:.2f} int={ind['n_intersections'][i]}\n"
                         f"cong={ind['congestion_index'][i]:.2f}", fontsize=7,
                         color="tab:green" if valid[i] else "tab:red")
        axes[r, 0].set_ylabel("\n".join(request[j:j + 28] for j in range(0, len(request), 28)), fontsize=6)
    fig.suptitle("Accepted requests: generated layouts (green = passes all layout rules)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("request", nargs="?", help="text request")
    parser.add_argument("--examples", action="store_true", help="run the built-in example requests")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    labels = pd.read_csv(PROJECT_ROOT / cfg["paths"]["data_dir"] / "labels.csv")
    checker = ConstraintChecker(labels, cfg["constraints"])
    stats, llm = indicator_stats(labels), ollama_llm(cfg["llm"])
    if args.examples:
        table = run_examples(cfg, llm, checker, stats, PROJECT_ROOT / cfg["paths"]["results_dir"])
        agree = (table["decision"] == table["expected"]).mean()
        print(f"\nDecision matches the expected one for {agree:.0%} of the requests")
        return
    res = interpret_request(args.request or input("Request: "), llm, checker, stats)
    print(f"LLM JSON: {res.llm_answer}\n{format_results(res.checks)}")
    print("ACCEPTED: generation may start" if res.accepted else "REJECTED: nothing is generated")
    if res.accepted:
        _, checks, ind = generate_for_spec(res.spec, cfg, checker, cfg["llm"]["n_generate"], cfg["seed"])
        print(f"{sum(all_passed(c) for c in checks)} of {len(checks)} generated layouts pass all layout rules")


if __name__ == "__main__":
    main()
