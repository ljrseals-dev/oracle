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
FAST_MODE = False


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
    # Recovered logistic and inverse-square behavior are intentionally retained.
    # For positive snap, exp(snap * (d - 0.04)) can overflow at large |distance|;
    # the logistic then tends toward zero. This is the recovered force law, not
    # a newly introduced clamp. Do not silently change its numerical behavior.
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
    spectrum_unit = spectrum / (float(np.linalg.norm(spectrum)) + eps)
    if previous_spectrum is None or previous_spectrum.shape != spectrum.shape:
        flux = 0.0
    else:
        # Bounded normalized spectral-flux ratio (0..1), not a raw norm delta:
        # this prevents silence-to-signal transitions from dominating the feature.
        # The trainer uses this exact ratio definition.
        flux = float(0.5 * np.linalg.norm(spectrum_unit - previous_spectrum))
    bins = (spectrum[:5] / (float(np.max(spectrum)) + eps)).tolist()
    features = np.asarray([rms, centroid, zcr, flatness, flux,
                           float(np.mean(audio)), float(np.std(audio)), *bins], dtype=np.float32)
    return features, spectrum_unit


def load_checkpoint(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions predating weights_only (trusted local file only).
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Not a {CHECKPOINT_FORMAT} checkpoint: {path}")
    # Metadata is JSON text, avoiding custom pickle globals under weights_only=True.
    metadata = json.loads(payload.get("metadata_json", "{}"))
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
        "impact_threshold": 0.50 + 1.50 * float(c[3]),
        # Newly designed narrow range [2.5, 3.5] s; c=0.5 reproduces
        # the recovered source's 3.0 s envelope time constant exactly.
        "release_time": 2.50 + 1.00 * float(c[4]),  # seconds
        "pitch_drift": -18.0 + 36.0 * float(c[5]),  # cents
    }, c.tolist()


def simulate(model, out_path, diagnostics_path=None, seed=1729,
             iterations=60, rtol=1e-5, atol=1e-7, feature_window_only=False):
    np.random.seed(seed)
    events = generate_events()
    audio = np.zeros(4096 if feature_window_only else len(T), dtype=np.float64)
    all_envs = []
    previous_spectrum = None
    output_history = []
    impact_count = 0
    controls_history = []
    impact_counts_by_iteration = []
    rms_by_iteration = []
    feature_history = []
    active_events = [event for event in events if event[0] < DUR]
    if FAST_MODE:
        # generate_events() repeats the exact same 12-event motif four times.
        # A three-event prefix makes this a bounded pipeline check, not a
        # musical-quality render; the full mode retains all 48 events.
        active_events = active_events[:3]

    render_duration = DUR
    evaluation_times = T
    cached_event_velocities = None
    fixed_control_signature = None

    for iteration in range(iterations):
        iteration_impact_count = 0
        audio_slice = audio[:4096]
        features, previous_spectrum = extract_features(audio_slice, previous_spectrum)
        controls, normalized_outputs = controls_for(model, features)
        if feature_window_only:
            signature = np.asarray(list(controls.values()), dtype=np.float64)
            if fixed_control_signature is None:
                fixed_control_signature = signature
            elif not np.allclose(signature, fixed_control_signature, rtol=0.0, atol=1e-10):
                raise ValueError("feature_window_only requires fixed controls; use full simulate() for conditional models")
        output_history.append(normalized_outputs)
        controls_history.append(controls)
        rms_by_iteration.append(float(features[0]))
        feature_names_all = ["rms", "spectral_centroid", "zero_crossing_rate",
                             "spectral_flatness", "spectral_flux", "mean", "std",
                             "fft_bin_0", "fft_bin_1", "fft_bin_2", "fft_bin_3", "fft_bin_4"]
        feature_history.append({name: float(value)
                                for name, value in zip(feature_names_all, features)})

        field_boost = controls["field_boost"]
        snap = controls["snap"]
        global_field = np.zeros_like(T)
        for e in all_envs:
            global_field += e * 0.1

        for event_index, (t0, note, vel) in enumerate(active_events):
            idx = int(t0 * FS)
            env = np.zeros_like(T)
            rise = int(0.12 * FS)
            env[idx:idx+rise] = np.linspace(0, vel*field_boost, rise)**(PHI*snap)
            env[idx+rise:] = vel * np.exp(-T[idx+rise:]/controls["release_time"])
            env *= (1 + 0.2*np.sin(2*np.pi*T*PHI))
            env += global_field
            target = 0.1 - 0.002*(note % 12)

            if feature_window_only and iteration > 0:
                raw_impacts, velocity_state = cached_event_velocities[event_index]
            else:
                sol = solve_ivp(
                    clapper_ode, [0, render_duration], [0, 0], t_eval=evaluation_times,
                    args=(env, target, LUNAR_PHASE, snap), rtol=rtol, atol=atol
                )
                x = sol.y[0]
                velocity_state = sol.y[1]
                raw_impacts = np.where(
                    np.diff(np.sign(x - 0.03 * controls["impact_threshold"])) > 0
                )[0]
                if feature_window_only and iteration == 0:
                    if cached_event_velocities is None:
                        cached_event_velocities = []
                    cached_event_velocities.append((raw_impacts.copy(), velocity_state.copy()))

            impacts = raw_impacts + np.random.randint(-2, 2, size=len(raw_impacts))
            impacts = impacts[(impacts > 0) & (impacts < len(T))]
            impact_count += int(len(impacts))
            iteration_impact_count += int(len(impacts))
            for imp in impacts:
                vel_imp = abs(velocity_state[imp])
                if feature_window_only and imp >= len(audio):
                    # Later events cannot affect audio[:4096], but consume the
                    # same six phase draws as resonator() to preserve the RNG
                    # stream for the next iteration's first event.
                    for _ in range(6):
                        np.random.rand()
                    continue
                ring = resonator(vel_imp, FREQ_SCALE[note % len(FREQ_SCALE)],
                                 controls["decay_scale"], controls["pitch_drift"])
                length = min(len(ring), len(audio)-imp)
                audio[imp:imp+length] += ring[:length]
            if iteration == iterations - 1:
                all_envs.append(env)
        impact_counts_by_iteration.append(iteration_impact_count)

    peak_before_normalize = float(np.max(np.abs(audio)))
    audio /= peak_before_normalize + 1e-9
    write(str(out_path), FS, (audio*32767).astype(np.int16))
    control_means = {k: float(np.mean([row[k] for row in controls_history]))
                     for k in controls_history[0]}
    # Preserve every iteration's exact network outputs and physical controls so
    # responsiveness can be evaluated, not inferred from aggregate means.
    control_std = {k: float(np.std([row[k] for row in controls_history]))
                   for k in controls_history[0]}
    normalized_outputs_by_iteration = [list(map(float, row)) for row in output_history]
    controls_by_iteration = controls_history
    feature_names = ["rms", "spectral_centroid", "zero_crossing_rate",
                     "spectral_flatness", "spectral_flux", "mean", "std",
                     "fft_bin_0", "fft_bin_1", "fft_bin_2", "fft_bin_3", "fft_bin_4"]
    output_feature_correlations = {}
    for output_index in range(6):
        output_values = np.asarray([row[output_index] for row in output_history], dtype=float)
        output_feature_correlations[f"output_{output_index}"] = {}
        for feature_name in feature_names:
            feature_values = np.asarray([row[feature_name] for row in feature_history], dtype=float)
            if np.std(output_values) < 1e-12 or np.std(feature_values) < 1e-12:
                corr = None
            else:
                corr = float(np.corrcoef(output_values, feature_values)[0, 1])
            output_feature_correlations[f"output_{output_index}"][feature_name] = corr

    diagnostics = {
        "checkpoint": str(getattr(model, "_checkpoint_path", "")),
        "sample_rate": FS, "duration_seconds": DUR, "iterations": iterations,
        "render_mode": ("feature_window_cached_fixed_control_sequence" if feature_window_only else
                        ("fast_pipeline_subset" if FAST_MODE else "full_quality")),
        "event_count_per_iteration": len(active_events), "impact_count": impact_count,
        "impact_count_by_iteration": impact_counts_by_iteration,
        "input_rms_by_iteration": rms_by_iteration,
        "input_features_by_iteration": feature_history,
        "normalized_outputs_by_iteration": normalized_outputs_by_iteration,
        "controls_by_iteration": controls_by_iteration,
        "control_std": control_std,
        "output_feature_correlations": output_feature_correlations,
        "feature_output_correlation_matrix": {
            "feature_names": feature_names,
            "output_names": [f"output_{i}" for i in range(6)],
            "pearson_r_rows_by_feature": [
                [output_feature_correlations[f"output_{j}"][name] for j in range(6)]
                for name in feature_names
            ],
        },
        "correlation_note": "Pearson r across render iterations; null means one series is effectively constant. Sequential trajectory only, not independent-sample validation.",
        "impact_threshold_direction": "higher RMS target lowers threshold; intended to permit more impacts",
        "release_time_range_seconds": [2.5, 3.5],
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


def configure_fast_mode():
    """Set short pipeline-verification render settings; full mode stays unchanged."""
    global DUR, T, LUNAR_PHASE, TIDAL_WAVE, FAST_MODE
    FAST_MODE = True
    DUR = 4.0
    T = np.arange(0, DUR, 1.0 / FS)
    LUNAR_PHASE = 0.5 * (1 + np.sin(2 * np.pi * T / (DUR * 1.15)))
    TIDAL_WAVE = 0.28 * np.sin(2 * np.pi * T / 4.1 + PHI * 1.2)


def compare(objective_a, objective_b, checkpoint_dir, seed,
            render_iterations=60, render_rtol=1e-5, render_atol=1e-7):
    names = {"calibration": "magnetic_oracle_calibration.pt",
             "feature-conditional": "magnetic_oracle_feature_conditional.pt"}
    if objective_a not in names or objective_b not in names:
        raise ValueError("--compare values must be calibration and/or feature-conditional")
    reports = []
    for label in (objective_a, objective_b):
        path = Path(checkpoint_dir) / names[label]
        model, metadata = load_checkpoint(path)
        model._checkpoint_path = str(path)
        wav = Path(checkpoint_dir) / f"compare_{label.replace('-', '_')}.wav"
        diag = simulate(model, wav, None, seed, iterations=render_iterations,
                        rtol=render_rtol, atol=render_atol)
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
    parser.add_argument("--fast", action="store_true",
                        help="Pipeline verification: 4 s, 15 iterations, first 3 events, rtol=1e-3, atol=1e-5")
    args = parser.parse_args()
    if args.compare:
        if args.fast:
            configure_fast_mode()
            compare(args.compare[0], args.compare[1], args.checkpoint_dir, args.seed,
                    render_iterations=15, render_rtol=1e-3, render_atol=1e-5)
        else:
            compare(args.compare[0], args.compare[1], args.checkpoint_dir, args.seed)
        return
    model, metadata = load_checkpoint(args.checkpoint)
    model._checkpoint_path = args.checkpoint
    if args.fast:
        configure_fast_mode()
        iterations, rtol, atol = 15, 1e-3, 1e-5
    else:
        iterations, rtol, atol = 60, 1e-5, 1e-7
    diagnostics = simulate(model, args.out, args.diagnostics, args.seed,
                           iterations=iterations, rtol=rtol, atol=atol)
    diagnostics["objective"] = metadata["objective"]
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
