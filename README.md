# MagneticOracle v1

**Recovered magnetic-clavichord physics with a newly designed neural control layer.**

MagneticOracle v1 documents and tests a recovered PyTorch model and magnetic clavichord renderer. The work repairs the disconnected training-loss path, makes the feature/target contracts explicit, and evaluates proposed control mappings against documented criteria.

> **Research status:** the pipeline and controlled feature-space identifiability result are accepted. The six-output feature-conditional model is **not accepted as a production baseline**. Physical-render centroid isolation, full rendering, musical quality, cross-trajectory generalization, and historical parity remain open.

## What is recovered—and what is new

### Recovered from the supplied source (`file.txt`)

- Model topology: `Linear(12,128) → ReLU → Linear(128,128) → ReLU → Linear(128,6)`.
- Adam optimizer and learning rate `0.003`.
- The documented magnetic force / clapper ODE, resonator partials and decays, event and impact behavior, and renderer parameters as identified in source comments and experiment reports.

### Newly designed; not historical claims

- The 12-value feature contract: RMS, normalized spectral centroid, zero-crossing rate, spectral flatness, normalized spectral flux, mean, standard deviation, and five normalized FFT magnitudes.
- The feature-conditional target mappings and training objectives.
- Recovered historical weights: **unavailable**. New checkpoints are newly trained models, not reconstructed original weights.
- The decorrelation intervention and its synthetic, nonphysical feature tuples.

The original loss was detached from the model graph (`torch.tensor(loss, requires_grad=True)`), so it could not train the network parameters. This project does not claim to recover the original intended objective.

## Experiments and verdicts

### Distribution-matched retrain — mixed / composite gate failed

Output 4's centroid correlation increased from `+0.139` to `+0.581`, exceeding the preregistered `+0.4` threshold. However, output 2's flatness correlation moved from `−0.929` to `−0.704` (`Δ=+0.225`), outside its `±0.15` stability tolerance. Severe feature collinearity prevented attributing output 4's recovery to centroid specifically.

The target-range diagnostic found output 2 target SD increased from `0.05102` to `0.07843` (about `1.537×`); a narrower target range is not supported as the explanation for output 2's degradation.

### Feature-decorrelation experiment — controlled pass, deployment isolation open

The experiment independently permuted each feature column within training and holdout splits, preserving each feature's empirical marginal while breaking row-wise cross-feature pairing. This creates **nonphysical feature tuples** and is a model-identifiability intervention, not an audio-generation or spectral-warp method.

On the independent decorrelated holdout:

- Output 4 vs centroid: `r=+0.9519` — criterion passed.
- Output 4 vs ZCR: `r=+0.0024` — not a strong proxy association.
- Output 4 vs flatness: `r=−0.0264` — not a strong proxy association.
- Output 2 vs flatness: `r=−0.9985` — protection criterion passed.

On the physical fast-render trajectory, output 4 correlated `+0.9754` with centroid but also `+0.6930` with ZCR. Thus centroid is **not isolated from ZCR in the physical render**. This experiment supports centroid-specific dependence under independent feature inputs; it does not prove unique centroid causation in deployment audio.

Read the complete preregistered report and structured record:

- [`feature_decorrelation_report.md`](experiments/feature_decorrelation/reports/feature_decorrelation_report.md)
- [`feature_decorrelation_acceptance.json`](experiments/feature_decorrelation/results/feature_decorrelation_acceptance.json)
- [`distribution_matched_retrain_report.md`](experiments/distribution_matched_retrain/reports/distribution_matched_retrain_report.md)
- [`distribution_matched_retrain_acceptance.json`](experiments/distribution_matched_retrain/results/distribution_matched_retrain_acceptance.json)

## Requirements

- Python 3.10+ recommended
- NumPy
- PyTorch
- SciPy

Install dependencies in your environment, for example:

```bash
python -m pip install numpy torch scipy
```

No package manager lockfile or environment-specific pinning is included in this research snapshot.

## Run the renderer

From the project directory:

```bash
python src/magnetic_oracle/render_magnetic_clavecin.py --fast \
  --checkpoint artifacts/checkpoints/magnetic_oracle_feature_decorrelation.pt \
  --seed 1729 \
  --out render.wav \
  --diagnostics render.json
```

`--fast` is a pipeline-validation mode: 4 seconds, 15 iterations, first three events, with relaxed ODE tolerances. It is **not** the full render. Omit `--fast` for the regular renderer path; note that full rendering has not been acceptance-tested in this project.

Show CLI options:

```bash
python src/magnetic_oracle/render_magnetic_clavecin.py --help
python src/magnetic_oracle/train_magnetic_oracle.py --help
```

## Training examples

The training script supports explicit objectives and distributions. Objectives and feature mappings are newly designed; neither is historical.

```bash
python src/magnetic_oracle/train_magnetic_oracle.py \
  --objective feature-conditional \
  --distribution resonator \
  --samples 512 \
  --epochs 200 \
  --seed 1729 \
  --out new_checkpoint.pt
```

The specific preregistered feature-decorrelation experiment is reproducible from the experiment runner included in `experiments/feature_decorrelation/code/run_decorrelation_experiment.py`. Consult that runner and the report for its exact split, independent column permutation, training settings, and evaluation criteria.

## Checkpoint and validation notes

- Checkpoints use the `magnetic-oracle-v1` format and are designed for tensor-only (`weights_only=True`) loading with JSON metadata.
- The feature-decorrelation checkpoint passed the renderer's weights-only load and golden-vector validation in the recorded run.
- Renderer diagnostics are per-iteration data; interpret correlations as trajectory-level diagnostics, not generalization evidence.
- The recovered logistic force expression can emit an overflow warning at large distances. Its formula was retained rather than silently clamped or altered.

## Privacy and scope

This code is a local research renderer/trainer. No network-based workflow is required by the documented commands. The experiments described here do not establish any broader production, musical, or historical claims.

## Artifact SHA-256

Hashes below refer to the artifacts from the recorded feature-decorrelation experiment. Verify after downloading if integrity matters.

| Artifact | SHA-256 |
|---|---|
| Feature-decorrelation experiment ZIP | `8835c2fc04e5873011a5fa2302b9b908130d0956366bd20fc9eeb742c7bb7b8c` |
| Decorrelated checkpoint | `4c7de5e6a2935fbd32fae0ce098247670440c3dd7a916185858ac0bbe455c891` |
| Fast renderer WAV | `adfb39a7e34822e27a5b9c3e66b4121dbd6206722be961807091095e83b210a6` |
| Feature-decorrelation report | `2fe45aa2a2a571d35ea8e2241b824a7dd52ce20ba206c77e67fac90d9e49f9dc` |
| Feature-decorrelation acceptance JSON | `9e216b42e938ead98c1f288b876522d57c07a4bdd459c39d169d51025106287a` |

## Suggested next research step

Design a physically realizable intervention that changes centroid while quantifying its effects on ZCR and other spectral descriptors. Predefine tolerances, check achieved feature distributions, and evaluate multiple independent renderer trajectories before making a physical-deployment claim.
