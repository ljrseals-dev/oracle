# MagneticOracle v1 — Preregistered Feature-Decorrelation Experiment

**Verdict: centroid-specific dependence is learned in decorrelated feature space, but is not isolated on the physical renderer trajectory.**

## Design

- Base data: 512 measured feature rows from 35 fixed-control calibration trajectories, using the real renderer ODE, impact crossing/jitter, resonator accumulation, and shared feature extractor.
- Split: 384 training rows and 128 holdout rows from disjoint base rows.
- Intervention: independently permute each of the 12 feature columns within each split. This preserves each feature marginal exactly within each split while destroying row-wise cross-feature pairing.
- Model: recovered 12→128→128→6 topology; Adam 0.003; seed 1729; 1,500 epochs; target functions unchanged.
- Limitation: decorrelated feature tuples are intentionally nonphysical. This is a feature-space identifiability intervention, not a spectral warp or proof that audio can vary centroid independently of all FFT bins.

## Feature marginal check

Independent column permutations preserve each split’s empirical feature marginals. The following shows training-split marginals against the previously measured 512-row calibration reference; the full 12-feature mean/std/min/max and correlation matrices are in the JSON record.

| Feature | Calibration reference mean ± SD | Decorrelated training mean ± SD |
|---|---:|---:|
| rms | 8.426296 ± 5.155990 | 8.599412 ± 5.336772 |
| spectral_centroid | 0.066636 ± 0.018510 | 0.075652 ± 0.020679 |
| zero_crossing_rate | 0.010061 ± 0.002715 | 0.010025 ± 0.002731 |
| spectral_flatness | 0.112580 ± 0.104751 | 0.139476 ± 0.097843 |
| spectral_flux | 0.039435 ± 0.123381 | 0.036198 ± 0.125026 |
| mean | 0.070133 ± 0.043604 | 0.067670 ± 0.042148 |
| std | 8.426004 ± 5.155806 | 8.599145 ± 5.336606 |
| fft_bin_0 | 0.029647 ± 0.007933 | 0.028762 ± 0.007782 |
| fft_bin_1 | 0.050856 ± 0.013596 | 0.050659 ± 0.013655 |
| fft_bin_2 | 0.064393 ± 0.017217 | 0.064489 ± 0.017390 |
| fft_bin_3 | 0.047941 ± 0.012816 | 0.047750 ± 0.012870 |
| fft_bin_4 | 0.030185 ± 0.008077 | 0.029300 ± 0.007925 |

After permutation, centroid↔ZCR is r=+0.052 and centroid↔flatness is r=+0.036 in training; on the holdout they are r=+0.160 and r=-0.045. The remaining feature matrix is also recorded in full.


## Target dynamic-range diagnostic

| Output | Original target mean ± SD | Matched target mean ± SD | Matched/original SD |
|---|---:|---:|---:|
| 0 | 0.979748 ± 0.113629 | 0.945313 ± 0.201889 | 1.777× |
| 1 | 0.843898 ± 0.109538 | 0.805151 ± 0.177467 | 1.620× |
| 2 | 0.877802 ± 0.051024 | 0.837609 ± 0.078427 | 1.537× |
| 3 | 0.171553 ± 0.107622 | 0.197852 ± 0.176653 | 1.641× |
| 4 | 0.208301 ± 0.009966 | 0.206813 ± 0.015597 | 1.565× |
| 5 | 0.395941 ± 0.062709 | 0.082890 ± 0.113021 | 1.802× |

Output 2 target SD increased from **0.05102** to **0.07843** (1.537×). The proposed explanation that output 2 degraded because its target dynamic range narrowed is contradicted by the measured statistics.

## Independent decorrelated holdout — model identifiability

| Output 4 vs feature | Pearson r |
|---|---:|
| fft_bin_0 | -0.0721 |
| fft_bin_1 | -0.0907 |
| fft_bin_2 | +0.0921 |
| fft_bin_3 | -0.1043 |
| fft_bin_4 | +0.0025 |
| mean | +0.0724 |
| rms | -0.1089 |
| spectral_centroid | +0.9519 |
| spectral_flatness | -0.0264 |
| spectral_flux | +0.1182 |
| std | +0.0130 |
| zero_crossing_rate | +0.0024 |

- Centroid: **+0.9519** (≥ +0.4: PASS).
- ZCR: **+0.0024** (absolute value < 0.4: PASS).
- Flatness: **-0.0264** (absolute value < 0.4: PASS).
- Output 2 vs flatness: **-0.9985** (within ±0.15 of −0.929: PASS).

**Feature-space result: centroid-specific dependence supported.** The holdout uses disjoint base rows and independent feature-column permutations.

## Physical renderer fast diagnostic

Scope: 4 seconds, 15 iterations, 3 events per iteration, seed 1729; 274 impacts. Not the full 48-event × 60-iteration render.

| Output 4 vs feature | Pearson r |
|---|---:|
| rms | +0.8768 |
| spectral_centroid | +0.9754 |
| zero_crossing_rate | +0.6930 |
| spectral_flatness | +0.0609 |
| spectral_flux | -0.3954 |
| mean | +0.8803 |
| std | +0.8768 |
| fft_bin_0 | +0.9435 |
| fft_bin_1 | +0.4514 |
| fft_bin_2 | +0.2714 |
| fft_bin_3 | +0.7880 |
| fft_bin_4 | +0.7793 |

- Centroid: **+0.9754** (≥ +0.4: PASS).
- ZCR: **+0.6930** (absolute value < 0.4: FAIL).
- Flatness: **+0.0609** (absolute value < 0.4: PASS).
- Output 2 vs flatness: **-0.9901** (within ±0.15 of −0.929: PASS).

**Physical-render result: centroid is still not isolated from ZCR.** The model has learned a centroid-specific mapping under independent inputs, but the real renderer trajectory couples centroid and ZCR. The output-4 render correlation therefore cannot be interpreted as a unique centroid mechanism.

## Verdict

1. **Centroid-specific dependence in controlled feature space: PASS.** Output 4 tracks centroid, not ZCR/flatness, on the decorrelated holdout.
2. **Proxy-learning criterion: NOT MET.** Output 4 centroid correlation does not fall below +0.3 and neither ZCR nor flatness remains ≥ +0.5 in the decorrelated holdout.
3. **Output 2 protection: PASS.** Output-2/flatness remains within tolerance on both the decorrelated holdout and fast physical render.
4. **Deployment isolation: NOT MET.** On the physical renderer output 4 also correlates +0.693 with ZCR, so the strong claim does not transfer to this coupled trajectory.

The experiment resolves the model-level identifiability question under independent feature interventions, but not the physical-deployment question. A physically valid follow-up would need an audio-generation method with quantified independent control over centroid and ZCR/spectral descriptors; simply shifting spectral energy cannot hold every FFT-derived feature fixed.

## Limitations

- Decorrelated rows are synthetic feature tuples, not realizable audio windows.
- Renderer validation is one fast trajectory, not independent physical generalization.
- Full render, musical quality, and historical parity remain untested.
- Original checkpoint weights are unrecoverable; trained weights and target functions remain newly designed.
