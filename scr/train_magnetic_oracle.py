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
    normalized FFT magnitudes]. This replaces the original random four inputs
    and raw-bin mix; architecture dimensions remain unchanged.
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
    if previous_spectrum is None or previous_spectrum.shape != spectrum.shape:
        flux = 0.0
    else:
        denom = float(np.linalg.norm(previous_spectrum)) + eps
        flux = float(np.linalg.norm(spectrum - previous_spectrum) / denom)
    scale = float(np.max(spectrum)) + eps
    bins = (spectrum[:5] / scale).tolist()
    features = np.asarray([rms, centroid, zcr, flatness, flux,
                           float(np.mean(audio)), float(np.std(audio)), *bins], dtype=np.float32)
    return features, spectrum


def feature_conditional_targets(features: np.ndarray) -> np.ndarray:
    """Auditable hand-designed targets; outputs are normalized to [0,1]."""
    rms, centroid, zcr, flatness, flux = np.clip(features[:5], 0.0, 1.0)
    return np.asarray([
        np.clip(0.20 + 1.10 * rms + 0.20 * centroid, 0.0, 1.0),  # field boost
        np.clip(0.15 + 0.70 * rms + 0.30 * zcr, 0.0, 1.0),       # snap
        np.clip(0.15 + 0.80 * (1.0 - flatness), 0.0, 1.0),      # decay
        np.clip(0.15 + 0.80 * rms, 0.0, 1.0),                   # impact threshold
        np.clip(0.15 + 0.75 * centroid, 0.0, 1.0),              # release
        np.clip(0.05 + 0.90 * flux, 0.0, 1.0),                   # pitch drift
    ], dtype=np.float32)


def make_training_set(count: int, seed: int):
    """Create deterministic synthetic audio examples and measured features."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    previous = None
    for i in range(count):
        n = 4096
        t = np.arange(n, dtype=np.float64) / 44100.0
        f0 = float(rng.uniform(70.0, 1800.0))
        f1 = float(rng.uniform(2.0, 8.0) * f0)
        amp = float(rng.uniform(0.02, 0.8))
        phase = float(rng.uniform(-np.pi, np.pi))
        audio = amp * (0.72 * np.sin(2*np.pi*f0*t + phase)
                       + 0.20 * np.sin(2*np.pi*f1*t)
                       + 0.08 * rng.standard_normal(n))
        # Vary transient density and spectral texture across examples.
        if i % 3 == 0:
            audio[:256] *= np.linspace(0.0, 1.0, 256)
        features, spectrum = extract_features(audio, previous)
        xs.append(features)
        ys.append(feature_conditional_targets(features))
        previous = spectrum
    return np.stack(xs), np.stack(ys)


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
    features, conditional_targets = make_training_set(args.samples, args.seed)
    x = torch.from_numpy(features)
    if args.objective == "calibration":
        targets = torch.full((args.samples, 6), 0.5, dtype=torch.float32)
        objective_name = "calibration_only"
    else:
        targets = torch.from_numpy(conditional_targets)
        objective_name = "feature_conditional"

    history = []
    model.train()
    for epoch in range(args.epochs):
        # Full-batch updates keep this small baseline deterministic and simple.
        prediction = torch.sigmoid(model(x))
        loss = torch.mean((prediction - targets) ** 2)
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
        "training_epochs": args.epochs,
        "training_samples": args.samples,
        "optimizer": "Adam",
        "learning_rate": 0.003,
        "feature_schema": "new v1 measured-audio feature contract; see extract_features()",
        "target_mapping": "hand-designed, inspectable functions; see feature_conditional_targets()",
        "recovered_weights": False,
        "git_commit": git_commit(),
        "seed": args.seed,
        "final_loss": history[-1] if history else None,
        "golden_input": probe.squeeze(0).tolist(),
        "golden_output_sigmoid": golden,
    }
    payload = {
        "format": CHECKPOINT_FORMAT,
        "metadata": metadata,
        "state_dict": model.state_dict(),
        "loss_history": history,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(json.dumps({"checkpoint": str(out), "metadata": metadata,
                      "initial_loss": history[0] if history else None,
                      "final_loss": history[-1] if history else None}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=OBJECTIVES, default="calibration")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 8:
        parser.error("--epochs must be >= 1 and --samples must be >= 8")
    if args.out is None:
        suffix = "calibration" if args.objective == "calibration" else "feature_conditional"
        args.out = f"magnetic_oracle_{suffix}.pt"
    train(args)


if __name__ == "__main__":
    main()
