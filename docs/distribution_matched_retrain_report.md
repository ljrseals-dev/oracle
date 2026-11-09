# MagneticOracle v1 — Distribution-Matched Retrain

**Result: mixed. Output 4 recovers its positive centroid correlation, but the pre-registered composite acceptance gate fails because output 2 degrades beyond tolerance.**

## Experiment executed

- 512 training rows from 35 sequential fixed-control calibration trajectories (34 full 15-step sequences plus 2 rows).
- Model seed 1729; trajectory seeds 1729–1763; 200 epochs; Adam, learning rate 0.003.
- Calibration controls were held exactly at normalized output 0.5 using a zero-parameter network. The collector used the renderer ODE, impact crossing/jitter, resonator superposition, accumulated audio, and shared feature extraction.
- Fast trajectory settings: 4 seconds, 15 iterations, 3-event subset, `rtol=1e-3`, `atol=1e-5`. The network feature window is the first 4096 samples.
- The feature-window optimization was checked against a full fast render at seed 1730: maximum absolute difference across all 12 features and 15 iterations was exactly 0.0.

## Pre-registered acceptance results

| Output | Intended driver | Previous r | Retrained r | Delta | Gate |
|---|---|---:|---:|---:|---|
| 0 | rms | +0.557 | +0.492 | -0.065 | pass |
| 1 | rms | +0.632 | +0.495 | -0.137 | pass |
| 2 | spectral_flatness | -0.929 | -0.704 | +0.225 | fail |
| 3 | rms | -0.639 | -0.495 | +0.144 | pass |
| 4 | spectral_centroid | +0.139 | +0.581 | +0.442 | pass |
| 5 | spectral_flux | +0.833 | +0.975 | +0.142 | pass |

Output 4 reaches **r = +0.581**, above the +0.4 primary threshold. However output 2 changes by **+0.225**, exceeding the ±0.15 tolerance. Therefore the composite primary gate fails and the secondary-instability condition is triggered.

## Distribution match

| Feature | Training mean ± SD | Calibration reference mean ± SD | Retrained conditional-render mean ± SD |
|---|---:|---:|---:|
| rms | 8.6530 ± 5.3597 | 8.4263 ± 5.1560 | 6.1549 ± 3.4012 |
| spectral_centroid | 0.0758 ± 0.0208 | 0.0666 ± 0.0185 | 0.0998 ± 0.0318 |
| zero_crossing_rate | 0.0100 ± 0.0027 | 0.0101 ± 0.0027 | 0.0100 ± 0.0028 |
| spectral_flatness | 0.1405 ± 0.0980 | 0.1126 ± 0.1048 | 0.2370 ± 0.0960 |
| spectral_flux | 0.0365 ± 0.1256 | 0.0394 ± 0.1234 | 0.0562 ± 0.1217 |

The training rows approximate the fixed-control calibration marginals, especially RMS, ZCR, and flux. The conditional model's own render shifts away from those marginals, particularly in RMS, centroid, and flatness, supporting the feedback-loop concern.

## The important confound: joint-feature collinearity

In the training set, centroid correlates **+0.986** with ZCR and **-0.962** with flatness. Output 4 correlates with centroid at **+0.971**, but also with ZCR at **+0.984**, flatness at **-0.989**, and FFT bins around +0.987 to +0.989.

In the retrained render, output 4 correlates **+0.581** with centroid, **+0.908** with ZCR, and **-0.926** with flatness. Thus the centroid association recovers, but this does not prove a unique centroid-specific dependency. Correlated spectral features remain plausible proxies.

## Training convergence

- Aggregate MSE: 0.150055 → 0.000364.
- Output 4 MSE: 0.084745 → 0.00001406; minimum at epoch 200.
- Output 2 MSE: 0.196645 → 0.00009982; minimum at epoch 200.

## Verdict

**Hypothesis A receives meaningful support, but is not confirmed under the pre-registered acceptance rule.** Output 4’s centroid correlation rises from +0.139 to +0.581, and the training target correlation rises from +0.574 to +0.971. However, output 2’s correlation changes by +0.225, violating the other-output stability requirement. The secondary instability condition requires another controlled run before drawing a final conclusion.

The training feature matrix also reveals severe collinearity. Output 4 can correlate with centroid because centroid is nearly interchangeable with ZCR, flatness, and several FFT bins in this calibration-sequence dataset. This weakens any claim that the retrain isolated centroid as the causal driver.

## What remains untested

- Full 48-event × 60-iteration render.
- Musical quality.
- Cross-trajectory generalization and independent-sample validation.
- Whether a new multi-trajectory/decorrelated training design can preserve all six output correlations.
- Historical parity; recovered weights remain unavailable.

## Artifacts

- `magnetic_oracle_feature_conditional_retrain.pt` — weights-only-safe checkpoint with JSON metadata and per-output loss history.
- `retrain_fast.wav` and `retrain_fast.json` — retrained fast render and per-iteration diagnostics.
- `retrain_training_stdout.json` — training summary, feature statistics, correlation matrix, and output-target/feature correlations.
- `distribution_matched_retrain_acceptance.json` — machine-readable acceptance record.
