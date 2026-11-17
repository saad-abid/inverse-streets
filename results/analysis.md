# Analysis of results

All layouts are **synthetic** (procedural 64x64 street patterns). Traffic indicators come from a
**simplified all-or-nothing shortest-path assignment** of random trips, not from a calibrated
traffic simulation. Every number below comes from running the code in this repository; the
source CSV is given for each.

## 1. Predicting indicators from an image (Phase 3)

Source: `predictor_metrics.csv`, `label_repeatability.csv`, `predictor_metrics_by_style.csv`.

The small CNN predicts road density (test R² 0.993), number of intersections (0.983) and
average shortest path (0.917) well, and the congestion index moderately (0.629). It beats both
baselines (training mean; linear model on road density) on every indicator except road
density, which the density-only baseline gets exactly by construction.

Two points limit how these numbers should be read:

- **Part of each traffic label is random.** Recomputing the test labels with other random trips
  and source nodes reproduces them only up to R² 0.958 (average shortest path) and 0.731
  (congestion index). These are ceilings for any predictor; the CNN is below but not far from them.
- **Much of the overall R² comes from recognising the style.** Within one style the indicators
  vary much less, and the CNN's R² drops (for example 0.103 for intersections in cul-de-sac
  layouts).

## 2. Generating layouts for target indicators (Phases 4 and 5)

Setup: 60 targets (density and intersections of 15 real test layouts per style), 10 generated
layouts per target (600 in total). Indicators of every generated layout are **recomputed with
the graph method**. As a reference, the same metrics are computed for the 10 real training
layouts whose indicators are closest to each target ("nearest real").
Sources: `generation_metrics.csv`, `generation_by_style.csv`, `generation_per_target.csv`,
`generation_cnn_crosscheck.csv`.

### Target agreement

| | Generated | Nearest real |
|---|---|---|
| road density, mean absolute error | 0.031 | 0.003 |
| road density, median relative error | 10.6 % | 1.0 % |
| intersections, mean absolute error | 7.42 | 0.28 |
| intersections, median relative error | 37.9 % | 0.0 % |

The generator follows the **density** target well over the whole range 0.06–0.60
(`generation_target_agreement.png`), slightly below the target for sparse layouts. It follows
the **intersections** target only roughly: the per-target mean error has a median of 6.1, and 9 of
the 60 targets (all regular or irregular grids) have an error above 10. Fine regular grids
with many thin streets are drawn as broken lines with far fewer intersections, and the
maximum target of 64 intersections is reached at only about 42–50.

### Plausibility

| | Generated | Nearest real |
|---|---|---|
| single connected road network | 38.3 % | 100 % |
| no small isolated fragments (< 10 % of the road pixels) | 45.0 % | 100 % |
| road width within the real range (1.0–3.0 px) | 99.0 % | 99.8 % |

Connectivity is the main weakness. By target style, the share of connected layouts is 65 %
(regular grid), 45 % (irregular grid), 32 % (radial) and 11 % (cul-de-sac): the sparser the
target, the more often thin roads break. For 8 of the 60 targets none of the 10 samples is
connected; for 3 targets all 10 are. The road-width check passes almost always, so it does not
separate good from bad samples here: broken roads still have a realistic median width.

### Indicator distributions

Generated layouts have many more **dead ends** than real ones (Wasserstein distance 7.30 vs 0.12
for the nearest-real reference), because broken roads create road ends. As a consequence they
also have a higher detour index (0.189 vs 0.012) and a higher congestion index (0.485 vs 0.055)
than real layouts (`generation_distributions.png`). Density and intersection distributions are
closer to the real ones.

### Diversity

The mean share of differing pixels between two layouts generated for the same target is 0.325,
close to the 0.339 between the 10 nearest real layouts. The generator does not collapse to one
output per target. This pixel measure is sensitive to small shifts, so it measures variety, not
the quality of that variety.

### Style is not controlled

Two numbers do not say which kind of layout is wanted. The generator never draws rings for
radial targets, and a dense cul-de-sac target (density 0.27, 6 intersections) produces
grid-like layouts with 9–32 intersections (`generated_grid.png`).

### The CNN predictor is not a reliable judge of generated layouts

On generated layouts, the CNN agrees with the graph method for density (R² 0.986) and roughly
for intersections (0.861), but not for average shortest path (0.187) or congestion index
(0.123), compared with 0.917 and 0.629 on real test layouts. Broken generated layouts are outside
the CNN's training distribution: it reads them like intact layouts. This is why generated
layouts must be checked with the graph method, not with the learned predictor.

## 3. What this means

- The pipeline works end to end: targets in, layouts out, independent check of the result.
- The cVAE is a **weak baseline**. It matches density, roughly matches intersections, keeps
  some variety, but produces a connected network for only 38 % of samples.
- In this dense synthetic dataset, simply **retrieving** the closest real layout beats the
  generator on every metric. A generator becomes useful for targets or combinations that the
  dataset does not contain, which Phase 6 has to check with explicit constraints.

## 4. Text requests, constraints and the LLM (Phase 6)

Source: `llm_requests.csv`. A local LLM (`qwen3.5:4b` through Ollama, CPU only) translated 8
hand-written requests into JSON; all 8 answers matched the JSON schema. The LLM copied every
explicit number and did not repair contradictions, so the explicit rules could reject them:
"very low road density but 40 intersections" and "a very dense city with almost no
intersections" were both rejected by three rules, each with a message saying which values the
dataset contains instead.

The decision matched the expected one for 6 of 8 requests. The two differences come from the
interaction between language and data, not from the code:

- "many intersections and little congestion" was mapped to the dataset extremes (64
  intersections, congestion index 1.22). Dense layouts with many intersections have a
  congestion index of at least 3.15 in this dataset, so the request was rejected. Whether
  "little congestion" should mean "the lowest ever seen" or "lower than usual" is a modelling
  choice that should be made explicit, not left to the LLM.
- "a sparse suburb" was mapped to density 0.09, below the fixed lower limit of 0.10 from the
  project plan. 12.4 % of the training layouts are below 0.10, so this fixed rule is stricter
  than the data.

For the two accepted requests, 5 of 20 and 13 of 20 generated layouts passed all layout rules
(density range, connectivity, congestion limit). Generation followed by filtering with the
graph method is what makes the final output trustworthy, since the generator itself cannot
guarantee connectivity or aim at a congestion level.

## 5. Making the predictor small and fast (Phase 7)

Source: `edge_benchmark.csv` (Intel Core i7-7820HQ, 1 and 4 threads, 748 test images).

Why it matters: an in-vehicle unit or a roadside device has little memory, a weak CPU, a power
budget, and often must answer within a fixed time. The p95 latency (the time 95 % of calls stay
under) matters more there than the mean, and a model that fits in a few hundred kilobytes can be
updated over a slow link.

What the measurements show:

- **The runtime matters as much as the model.** The same FP32 network answers in 0.32 ms with
  ONNX Runtime and in 2.19 ms with PyTorch (1 thread), with identical outputs. For a small model,
  framework overhead dominates.
- **Quantization is not automatically faster.** Dynamic INT8 on all layers makes the file 3.7x
  smaller (0.79 -> 0.21 MB) without losing accuracy, but runs 17x slower (5.42 ms), because
  quantized convolutions fall back to a slow generic kernel.
- **Choosing what to quantize matters.** Quantizing only the linear layers, which hold most of
  the weights, halves the size (0.40 MB) at the same speed and accuracy (mean R² 0.8805 vs
  0.8804). This is the best trade-off found.
- **Static INT8 trades accuracy for size.** It is the smallest version that stays fast (0.22 MB,
  0.48 ms), but the intersections R² drops from 0.983 to 0.951. Calibrating on more or more varied
  images, or keeping the first convolution in FP32, are the usual next steps.
- **The hardware decides.** This 2017 CPU has no 8-bit dot-product instructions, so INT8 gives
  no speed-up here; on recent x86 CPUs, ARM boards or an NVIDIA Jetson the result can differ and
  has to be measured on the target device.

## 6. Possible improvements (not done)

Longer training (the validation loss was still decreasing at epoch 30); a connectivity or
topology term in the loss; conditioning on more indicators (congestion) or on the style;
generating a graph instead of pixels; a conditional diffusion model for comparison.
