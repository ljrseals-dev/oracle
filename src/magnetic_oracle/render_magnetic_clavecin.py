#!/usr/bin/env python3
"""Render the recovered magnetic clavichord physics using MagneticOracle v1.

Physics constants, magnetic_force(), clapper_ode(), event pattern, solve_ivp
settings, impact crossing/jitter, partial ratios, amplitudes, and WAV format
follow file.txt. The six neural control mappings are newly designed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.io.wavfile import write

PHI = (1 + np.sqrt(5)) / 2
FS = 44100
DUR = 18.0
DT = 1.0 / FS
T = np.arange(0, DUR, DT)
BASE_FREQ = 261.63
CMAJOR_SCALE = np.array([0, 2, 4, 5, 7, 9, 11, 12])
FREQ_SCALE = BASE_FREQ * (2 ** (CMAJOR_SCALE / 12.0)) * np.power(PHI, np.arange(8) / 12.0)
VOICES = 8
LUNAR_PHASE = 0.5 * (1 + np.sin(2 * np.pi * T / (DUR * 1.15)))
TIDAL_WAVE = 0.28 * np.sin(2 * np.pi * T / 4.1 + PHI * 1.2)
CHECKPOINT_FORMAT = "magnetic-oracle-v1"


class MagneticOracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        return self.net(x)


def magnetic_force(d, strength, phase, snap):
    d = np.abs(d) + 1e-6
    snap_curve = 1 / (1 + np.exp(-snap * (0.04 - d)))
    lunar = phase**PHI if phase > 0.5 else phase * 1.7
    tidal = 1 + 0.3 * np.sin(2 * np.pi * d * PHI)
    return strength * lunar * tidal * snap_curve / (d**2)


def clapper_ode(t, y, env, target, phase_arr, snap):
    x, v = y
    idx = min(int(t * FS), len(env) - 1)
    field = env[idx]
    phase = phase_arr[idx]
    F = magnetic_force(target - x, field, phase, snap)
    acc = (F - 0.9 * v - 40 * x) / 0.05
    return [v, acc]


def resonator(velocity, freq, decay_scale=1.0, pitch_drift=0.0):
    partials = np.array([1.0, 2.14, 3.0, 4.25, 5.6, 7.1]) * (1 + 0.002*np.arange(6)**2)
    decays = np.array([3.0, 2.4, 1.8, 1.3, 0.9, 0.6]) * decay_scale
    amps = np.array([1.0, 0.7, 0.5, 0.35, 0.25, 0.18])
    out = np.zeros_like(T)
    drifted_freq = freq * (2.0 ** (pitch_drift / 1200.0))
    for i in range(len(partials)):
        f = drifted_freq * partials[i]
        env = velocity * amps[i] * np.exp(-T * decays[i])
        out += env * np.sin(2*np.pi*f*T + np.random.rand()*0.3)
    return out


def generate_events():
    events = []
    t = 0.0
    for octave in range(3):
        for deg in [0, 4, 7, 12]:
            note = 60 + deg + octave*12
            events.append((t, note, 0.8 - octave*0.1))
            t += 0.11
    return events * 4


def extract_features(audio, previous_spectrum=None):
    """Must match train_magnetic_oracle.py's v1 feature contract."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        audio = np.zeros(4096, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(audio))
    eps = 1e-12
    rms = float(np.sqrt(np.mean(audio * audio) + eps))
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / FS)
    total = float(np.sum(spectrum)) + eps
    centroid = float(np.sum(freqs * spectrum) / total / (FS / 2.0))
    zcr = float(np.mean(np.diff(np.signbit(audio)))) if audio.size > 1 else 0.0
    positive = spectrum[1:] + eps
    flatness = float(np.exp(np.mean(np.log(positive))) / (np.mean(positive) + eps))
    if previous_spectrum is None or previous_spectrum.shape != spectrum.shape:
        flux = 0.0
    else:
        flux = float(np.linalg.norm(spectrum - previous_spectrum) /
                      (float(np.linalg.norm(previous_spectrum)) + eps))
    bins = (spectrum[:5] / (float(np.max(spectrum)) + eps)).tolist()
    features = np.asarray([rms, centroid, zcr, flatness, flux,
                           float(np.mean(audio)), float(np.std(audio)), *bins], dtype=np.float32)
    return features, spectrum


def load_checkpoint(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older supported PyTorch versions do not expose weights_only.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Not a {CHECKPOINT_FORMAT} checkpoint: {path}")
    metadata = payload.get("metadata", {})
    if metadata.get("name") != "MagneticOracle v1" or metadata.get("recovered_weights") is not False:
        raise ValueError("Checkpoint metadata is missing the MagneticOracle v1 provenance declaration")
    model = MagneticOracle()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        golden_input = torch.tensor(metadata["golden_input"], dtype=torch.float32).unsqueeze(0)
        actual = torch.sigmoid(model(golden_input)).squeeze(0).tolist()
    expected = metadata["golden_output_sigmoid"]
    if not np.allclose(actual, expected, rtol=1e-6, atol=1e-7):
        raise ValueError("Checkpoint golden-vector regression check failed")
    return model, metadata


def controls_for(model, features):
    with torch.no_grad():
        raw = model(torch.from_numpy(features).unsqueeze(0))
        c = torch.sigmoid(raw).squeeze(0).cpu().numpy()
    # Explicit physical mappings: every output is bounded and has documented units.
    return {
        "field_boost": 0.75 + 0.50 * float(c[0]),
        "snap": 0.80 + 0.80 * float(c[1]),
        "decay_scale": 0.50 + 1.50 * float(c[2]),
        "impact_thr": 0.50 + 1.50 * float(c[3]),
        "release_time": 1.00 + 4.00 * float(c[4]),  # seconds
        "pitch_drift": -18.0 + 36.0 * float(c[5]),  # cents
    }, c.tolist()


def simulate(model, out_path, diagnostics_path=None, seed=1729):
    np.random.seed(seed)
    events = generate_events()
    audio = np.zeros_like(T)
    all_envs = []
    previous_spectrum = None
    output_history = []
    impact_count = 0
    controls_history = []

    for iteration in range(60):
        audio_slice = audio[:4096]
        features, previous_spectrum = extract_features(audio_slice, previous_spectrum)
        controls, normalized_outputs = controls_for(model, features)
        output_history.append(normalized_outputs)
        controls_history.append(controls)

        field_boost = controls["field_boost"]
        snap = controls["snap"]
        global_field = np.zeros_like(T)
        for e in all_envs:
            global_field += e * 0.1

        for t0, note, vel in events:
            idx = int(t0 * FS)
            env = np.zeros_like(T)
            rise = int(0.12 * FS)
            env[idx:idx+rise] = np.linspace(0, vel*field_boost, rise)**(PHI*snap)
            env[idx+rise:] = vel * np.exp(-T[idx+rise:]/controls["release_time"])
            env *= (1 + 0.2*np.sin(2*np.pi*T*PHI))
            env += global_field
            target = 0.1 - 0.002*(note % 12)

            sol = solve_ivp(
                clapper_ode, [0, DUR], [0, 0], t_eval=T,
                args=(env, target, LUNAR_PHASE, snap), rtol=1e-5, atol=1e-7
            )
            x = sol.y[0]
            v = sol.y[1]
            impacts = np.where(np.diff(np.sign(x - 0.03 * controls["impact_thr"])) > 0)[0]
            impacts += np.random.randint(-2, 2, size=len(impacts))
            impacts = impacts[(impacts > 0) & (impacts < len(T))]
            impact_count += int(len(impacts))
            for imp in impacts:
                vel_imp = abs(v[imp])
                ring = resonator(vel_imp, FREQ_SCALE[note % len(FREQ_SCALE)],
                                 controls["decay_scale"], controls["pitch_drift"])
                length = min(len(ring), len(audio)-imp)
                audio[imp:imp+length] += ring[:length]
            if iteration == 59:
                all_envs.append(env)

    peak_before_normalize = float(np.max(np.abs(audio)))
    audio /= peak_before_normalize + 1e-9
    write(str(out_path), FS, (audio*32767).astype(np.int16))
    control_means = {k: float(np.mean([row[k] for row in controls_history]))
                     for k in controls_history[0]}
    diagnostics = {
        "checkpoint": str(getattr(model, "_checkpoint_path", "")),
        "sample_rate": FS, "duration_seconds": DUR, "iterations": 60,
        "event_count_per_iteration": len(events), "impact_count": impact_count,
        "peak_before_normalize": peak_before_normalize,
        "normalized_output_min": np.min(output_history, axis=0).tolist(),
        "normalized_output_max": np.max(output_history, axis=0).tolist(),
        "normalized_output_mean": np.mean(output_history, axis=0).tolist(),
        "control_mean": control_means,
        "wav": str(out_path), "seed": seed,
    }
    if diagnostics_path:
        Path(diagnostics_path).write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    return diagnostics


def compare(objective_a, objective_b, checkpoint_dir, seed):
    names = {"calibration": "magnetic_oracle_calibration.pt",
             "feature-conditional": "magnetic_oracle_feature_conditional.pt"}
    if objective_a not in names or objective_b not in names:
        raise ValueError("--compare values must be calibration and/or feature-conditional")
    reports = []
    for label in (objective_a, objective_b):
        path = Path(checkpoint_dir) / names[label]
        model, metadata = load_checkpoint(path)
        model._checkpoint_path = str(path)
        wav = Path(f"compare_{label.replace('-', '_')}.wav")
        diag = simulate(model, wav, None, seed)
        diag["objective"] = metadata["objective"]
        reports.append(diag)
    keys = ("impact_count", "peak_before_normalize")
    deltas = {key: reports[1][key] - reports[0][key] for key in keys}
    deltas["normalized_output_mean"] = (
        np.asarray(reports[1]["normalized_output_mean"]) -
        np.asarray(reports[0]["normalized_output_mean"])
    ).tolist()
    print(json.dumps({"comparison": [objective_a, objective_b],
                      "reports": reports, "deltas_b_minus_a": deltas}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="magnetic_oracle_calibration.pt")
    parser.add_argument("--out", default="clavecin_magnetique_FINAL.wav")
    parser.add_argument("--diagnostics", default="diagnostics.json")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--compare", nargs=2, metavar=("OBJECTIVE_A", "OBJECTIVE_B"),
                        help="Render calibration and feature-conditional checkpoints side by side")
    parser.add_argument("--checkpoint-dir", default=".")
    args = parser.parse_args()
    if args.compare:
        compare(args.compare[0], args.compare[1], args.checkpoint_dir, args.seed)
        return
    model, metadata = load_checkpoint(args.checkpoint)
    model._checkpoint_path = args.checkpoint
    diagnostics = simulate(model, args.out, args.diagnostics, args.seed)
    diagnostics["objective"] = metadata["objective"]
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
