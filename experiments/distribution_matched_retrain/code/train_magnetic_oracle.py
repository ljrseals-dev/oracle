#!/usr/bin/env python3
"""Train MagneticOracle v1 with explicit, newly designed objectives.

The 12->128->128->6 architecture and Adam learning rate are recovered from
file.txt. Neither objective nor the feature schema is historical.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ARCHITECTURE = {"input": 12, "hidden": [128, 128], "output": 6}
CHECKPOINT_FORMAT = "magnetic-oracle-v1"
OBJECTIVES = ("calibration", "feature-conditional")


class MagneticOracle(nn.Module):
    """Recovered topology: Linear(12,128)-ReLU-Linear(128,128)-ReLU-Linear(128,6)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract_features(audio: np.ndarray, previous_spectrum: np.ndarray | None = None):
    """New, documented 12-value feature contract shared with the renderer.

    [RMS, centroid/Nyquist, ZCR, flatness, flux, mean, std, first five
    normalized FFT magnitudes]. Flux is half the L2 distance between
    unit-normalized spectra (bounded [0, 1]), not a raw norm difference; this
    prevents silence-to-signal transitions from dominating the feature. This
    replaces the original random four inputs and raw-bin mix; architecture
    dimensions remain unchanged.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        audio = np.zeros(4096, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(audio))
    eps = 1e-12
    rms = float(np.sqrt(np.mean(audio * audio) + eps))
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / 44100.0)
    total = float(np.sum(spectrum)) + eps
    centroid = float(np.sum(freqs * spectrum) / total / 22050.0)
    zcr = float(np.mean(np.diff(np.signbit(audio)))) if audio.size > 1 else 0.0
    positive = spectrum[1:] + eps
    flatness = float(np.exp(np.mean(np.log(positive))) / (np.mean(positive) + eps))
    spectrum_unit = spectrum / (float(np.linalg.norm(spectrum)) + eps)
    if previous_spectrum is None or previous_spectrum.shape != spectrum.shape:
        flux = 0.0
    else:
        # Bounded normalized spectral flux in [0, 1], including silence-to-signal.
        flux = float(0.5 * np.linalg.norm(spectrum_unit - previous_spectrum))
    scale = float(np.max(spectrum)) + eps
    bins = (spectrum[:5] / scale).tolist()
    features = np.asarray([rms, centroid, zcr, flatness, flux,
                           float(np.mean(audio)), float(np.std(audio)), *bins], dtype=np.float32)
    return features, spectrum_unit


def feature_conditional_targets(features: np.ndarray) -> np.ndarray:
    """Auditable hand-designed targets; outputs are normalized to [0,1]."""
    rms, centroid, zcr, flatness, flux = np.clip(features[:5], 0.0, 1.0)
    return np.asarray([
        np.clip(0.20 + 1.10 * rms + 0.20 * centroid, 0.0, 1.0),  # field boost
        np.clip(0.15 + 0.70 * rms + 0.30 * zcr, 0.0, 1.0),       # snap
        np.clip(0.15 + 0.80 * (1.0 - flatness), 0.0, 1.0),      # decay
        np.clip(0.85 - 0.70 * rms, 0.0, 1.0),                   # impact threshold: louder -> lower threshold
        np.clip(0.15 + 0.75 * centroid, 0.0, 1.0),              # release
        np.clip(0.05 + 0.90 * flux, 0.0, 1.0),                   # pitch drift
    ], dtype=np.float32)


PHI = (1.0 + np.sqrt(5.0)) / 2.0
BASE_FREQ = 261.63
CMAJOR_SCALE = np.array([0, 2, 4, 5, 7, 9, 11, 12])
FREQ_SCALE = BASE_FREQ * (2 ** (CMAJOR_SCALE / 12.0)) * np.power(PHI, np.arange(8) / 12.0)


def sample_resonator_audio(rng: np.random.Generator, n: int = 4096) -> np.ndarray:
    """Render short impact-driven examples with the renderer's resonator recipe.

    This is a training-data sampler, not a replacement for the clapper ODE.
    Partial ratios, amplitudes, exponential decays, and pitch-drift conversion
    intentionally match render_magnetic_clavecin.py::resonator().
    """
    t = np.arange(n, dtype=np.float64) / 44100.0
    audio = np.zeros(n, dtype=np.float64)
    partials = np.array([1.0, 2.14, 3.0, 4.25, 5.6, 7.1]) * (1 + 0.002*np.arange(6)**2)
    base_decays = np.array([3.0, 2.4, 1.8, 1.3, 0.9, 0.6])
    amps = np.array([1.0, 0.7, 0.5, 0.35, 0.25, 0.18])
    # Renderer audio is a sum accumulated across many impact events. Vary the
    # count broadly (including silence) instead of training on isolated tones.
    for _ in range(int(rng.integers(0, 97))):
        velocity = float(rng.uniform(0.1, 1.0))
        freq = float(rng.choice(FREQ_SCALE))
        decay_scale = float(rng.uniform(0.5, 2.0))
        pitch_drift = float(rng.uniform(-18.0, 18.0))
        start = int(rng.integers(0, max(1, n // 3)))
        local_t = t[:n-start]
        drifted_freq = freq * (2.0 ** (pitch_drift / 1200.0))
        ring = np.zeros(n-start, dtype=np.float64)
        for j, partial in enumerate(partials):
            f = drifted_freq * partial
            envelope = velocity * amps[j] * np.exp(-local_t * base_decays[j] * decay_scale)
            ring += envelope * np.sin(2*np.pi*f*local_t + rng.random()*0.3)
        audio[start:] += ring
    return audio


def sample_synthetic_audio(rng: np.random.Generator, n: int = 4096) -> np.ndarray:
    """Optional legacy synthetic distribution; not representative of renderer output."""
    t = np.arange(n, dtype=np.float64) / 44100.0
    f0 = float(rng.choice(FREQ_SCALE))
    f1 = float(rng.uniform(2.0, 8.0) * f0)
    amp = float(rng.uniform(0.02, 0.8))
    phase = float(rng.uniform(-np.pi, np.pi))
    return amp * (0.72 * np.sin(2*np.pi*f0*t + phase)
                  + 0.20 * np.sin(2*np.pi*f1*t)
                  + 0.08 * rng.standard_normal(n))


def make_training_set(count: int, seed: int, distribution: str = "resonator"):
    """Create deterministic measured-feature examples from the selected audio distribution."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    previous = None
    sampler = sample_resonator_audio if distribution == "resonator" else sample_synthetic_audio
    for _ in range(count):
        audio = sampler(rng)
        features, spectrum = extract_features(audio, previous)
        xs.append(features)
        ys.append(feature_conditional_targets(features))
        previous = spectrum
    return np.stack(xs), np.stack(ys)


FEATURE_NAMES = ["rms", "spectral_centroid", "zero_crossing_rate",
                 "spectral_flatness", "spectral_flux", "mean", "std",
                 "fft_bin_0", "fft_bin_1", "fft_bin_2", "fft_bin_3", "fft_bin_4"]

def feature_distribution_summary(features: np.ndarray) -> dict:
    corr = np.corrcoef(features, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "feature_names": FEATURE_NAMES,
        "min": features.min(axis=0).astype(float).tolist(),
        "max": features.max(axis=0).astype(float).tolist(),
        "mean": features.mean(axis=0).astype(float).tolist(),
        "std": features.std(axis=0).astype(float).tolist(),
        "pearson_correlation_matrix": corr.astype(float).tolist(),
    }

def make_calibration_sequence_training_set(count: int, seed: int, sequence_runs: int):
    """Collect real sequential feature rows from fixed-control fast physics renders.

    Each trajectory uses the renderer's actual ODE, impact crossing/jitter,
    resonator superposition, audio accumulation, and shared feature extractor.
    A zero-parameter network gives sigmoid outputs exactly 0.5, making this a
    fixed-control calibration trajectory rather than a feature-conditional
    self-feedback trajectory. Rows remain ordered within each 15-step sequence.
    """
    import render_magnetic_clavecin as renderer

    renderer.configure_fast_mode()
    calibration_model = renderer.MagneticOracle()
    with torch.no_grad():
        for parameter in calibration_model.parameters():
            parameter.zero_()
    xs, sequence_ids, step_ids = [], [], []
    sequence_count = max(sequence_runs, int(np.ceil(count / 15)))
    with tempfile.TemporaryDirectory(prefix="magnetic_oracle_calibration_") as temp_dir:
        for sequence_index in range(sequence_count):
            wav_path = Path(temp_dir) / f"calibration_{sequence_index:03d}.wav"
            diagnostics = renderer.simulate(
                calibration_model, wav_path, diagnostics_path=None,
                seed=seed + sequence_index, iterations=15, rtol=1e-3, atol=1e-5,
                feature_window_only=True
            )
            for step_index, row in enumerate(diagnostics["input_features_by_iteration"]):
                if len(xs) >= count:
                    break
                xs.append([row[name] for name in FEATURE_NAMES])
                sequence_ids.append(sequence_index)
                step_ids.append(step_index)
            if len(xs) >= count:
                break
    if len(xs) != count:
        raise RuntimeError(f"Collected {len(xs)} of {count} requested calibration-sequence samples")
    features = np.asarray(xs, dtype=np.float32)
    targets = np.stack([feature_conditional_targets(row) for row in features])
    summary = feature_distribution_summary(features)
    summary["sequence_count_used"] = len(set(sequence_ids))
    summary["sequence_length_steps"] = 15
    summary["sequence_ids"] = sequence_ids
    summary["step_ids"] = step_ids
    summary["trajectory_seed_start"] = seed
    summary["trajectory_seed_end"] = seed + max(sequence_ids)
    summary["trajectory_mode"] = "renderer_fast_calibration_fixed_controls"
    summary["physics_source"] = "render_magnetic_clavecin.py::simulate; real ODE and accumulated resonator output"
    summary["reference_scope"] = "4-second fast mode, 15 iterations, first 3 events; not full 48-event render"
    return features, targets, summary

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True, timeout=2,
        ).strip()
    except Exception:
        return os.environ.get("GIT_COMMIT", "unknown")


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    model = MagneticOracle()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    if args.distribution == "calibration-sequence":
        features, conditional_targets, distribution_summary = make_calibration_sequence_training_set(
            args.samples, args.seed, args.sequence_runs
        )
    else:
        features, conditional_targets = make_training_set(args.samples, args.seed, args.distribution)
        distribution_summary = feature_distribution_summary(features)
    x = torch.from_numpy(features)
    if args.objective == "calibration":
        targets = torch.full((args.samples, 6), 0.5, dtype=torch.float32)
        objective_name = "calibration_only"
    else:
        targets = torch.from_numpy(conditional_targets)
        objective_name = "feature_conditional"

    history = []
    per_output_loss_history = []
    model.train()
    for epoch in range(args.epochs):
        # Full-batch updates keep this small baseline deterministic and simple.
        prediction = torch.sigmoid(model(x))
        squared_error = (prediction - targets) ** 2
        per_output_loss = torch.mean(squared_error, dim=0)
        # Keep the original aggregate reduction expression/order unchanged;
        # per-output diagnostics are observational and do not alter gradients.
        loss = torch.mean(squared_error)
        per_output_loss_history.append(per_output_loss.detach().cpu().tolist())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.sqrt(sum(
            torch.sum(p.grad.detach() ** 2) for p in model.parameters()
            if p.grad is not None
        )).item())
        if not np.isfinite(grad_norm) or grad_norm <= 0.0:
            raise RuntimeError("Gradient-path check failed: gradient norm is not positive and finite")
        optimizer.step()
        history.append(float(loss.detach().item()))

    model.eval()
    probe = torch.from_numpy(features[:1])
    with torch.no_grad():
        golden = torch.sigmoid(model(probe)).squeeze(0).cpu().tolist()

    metadata = {
        "name": "MagneticOracle v1",
        "checkpoint_format": CHECKPOINT_FORMAT,
        "architecture": ARCHITECTURE,
        "architecture_source": "recovered from file.txt verbatim",
        "objective": objective_name,
        "objective_flag_value": args.objective,
        "objective_provenance": "newly designed; not recovered from source",
        "outputs_2_5_wiring": "newly designed; documented in render_magnetic_clavecin.py",
        "release_time_range_seconds": [2.5, 3.5],
        "impact_threshold_mapping": "0.85 - 0.70 * rms; higher RMS lowers threshold",
        "training_epochs": args.epochs,
        "training_samples": args.samples,
        "optimizer": "Adam",
        "learning_rate": 0.003,
        "feature_schema": "new v1 measured-audio feature contract; see extract_features()",
        "training_distribution": args.distribution,
        "training_distribution_provenance": (
            "real sequential feature rows collected from fixed-control calibration renders using the renderer ODE, impact crossing/jitter, accumulated resonator audio, and shared feature extractor"
            if args.distribution == "calibration-sequence" else
            ("renderer resonator partials/decays/amps and pitch-drift recipe"
             if args.distribution == "resonator" else
             "optional synthetic two-tone/noise placeholder; distribution mismatch expected")
        ),
        "training_distribution_summary": distribution_summary,
        "target_mapping": "hand-designed, inspectable functions; see feature_conditional_targets()",
        "recovered_weights": False,
        "git_commit": git_commit(),
        "seed": args.seed,
        "final_loss": history[-1] if history else None,
        "final_per_output_mse": per_output_loss_history[-1] if per_output_loss_history else None,
        "per_output_loss_history_in_checkpoint": True,
        "training_feature_names": FEATURE_NAMES,
        "training_feature_min": features.min(axis=0).astype(float).tolist(),
        "training_feature_max": features.max(axis=0).astype(float).tolist(),
        "training_feature_mean": features.mean(axis=0).astype(float).tolist(),
        "training_feature_std": features.std(axis=0).astype(float).tolist(),
        "golden_input": probe.squeeze(0).tolist(),
        "golden_output_sigmoid": golden,
    }
    payload = {
        "format": CHECKPOINT_FORMAT,
        # Keep the checkpoint compatible with PyTorch's restricted weights-only
        # unpickler: metadata is JSON text and numeric history is a tensor.
        "metadata_json": json.dumps(metadata, sort_keys=True),
        "state_dict": model.state_dict(),
        "loss_history": torch.tensor(history, dtype=torch.float64),
        "per_output_loss_history": torch.tensor(per_output_loss_history, dtype=torch.float64),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        fitted = torch.sigmoid(model(x)).cpu().numpy()
    fitted_target_correlations = {}
    fitted_feature_correlations = {}
    for output_index in range(6):
        a, b = fitted[:, output_index], targets.cpu().numpy()[:, output_index]
        fitted_target_correlations[f"output_{output_index}"] = (
            float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else None
        )
        fitted_feature_correlations[f"output_{output_index}"] = {}
        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            feature_values = features[:, feature_index]
            fitted_feature_correlations[f"output_{output_index}"][feature_name] = (
                float(np.corrcoef(a, feature_values)[0, 1])
                if np.std(a) > 1e-12 and np.std(feature_values) > 1e-12 else None
            )
    metadata["training_output_target_pearson_r"] = fitted_target_correlations
    metadata["training_output_feature_pearson_r"] = fitted_feature_correlations
    payload["metadata_json"] = json.dumps(metadata, sort_keys=True)
    torch.save(payload, out)
    print(json.dumps({"checkpoint": str(out), "metadata": metadata,
                      "initial_loss": history[0] if history else None,
                      "final_loss": history[-1] if history else None,
                      "final_per_output_mse": per_output_loss_history[-1] if per_output_loss_history else None,
                      "training_output_target_pearson_r": fitted_target_correlations,
                      "training_output_feature_pearson_r": fitted_feature_correlations}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=OBJECTIVES, default="calibration")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--distribution", choices=("resonator", "synthetic", "calibration-sequence"), default="resonator",
                        help="Training data: resonator sampler, synthetic placeholder, or real fixed-control renderer sequences")
    parser.add_argument("--sequence-runs", type=int, default=35,
                        help="Maximum calibration trajectories for calibration-sequence mode (15 steps each)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 8 or args.sequence_runs < 1:
        parser.error("--epochs must be >= 1, --samples >= 8, and --sequence-runs >= 1")
    if args.out is None:
        suffix = "calibration" if args.objective == "calibration" else "feature_conditional"
        args.out = f"magnetic_oracle_{suffix}.pt"
    train(args)


if __name__ == "__main__":
    main()
