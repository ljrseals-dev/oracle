#!/usr/bin/env python3
"""Feature-space factorial decorrelation experiment for MagneticOracle v1.

This deliberately permutes each measured feature column independently. It
preserves every feature's empirical marginal exactly but creates nonphysical
feature tuples. It is a model-identifiability intervention, not audio DSP or a
claim that an audio spectral warp can independently move centroid and FFT bins.
"""
from __future__ import annotations
import json, random, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import train_magnetic_oracle as trainer

ROOT = Path(__file__).resolve().parent
TRAIN_SEED = 1729
HOLDOUT_SEED = 3001
SAMPLES = 512
TRAIN_ROWS = 384
EPOCHS = 1500


def independently_permute_columns(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = x[rng.permutation(len(x)), j]
    return out


def corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def stats(a):
    return {"mean": a.mean(0).astype(float).tolist(),
            "std": a.std(0).astype(float).tolist(),
            "min": a.min(0).astype(float).tolist(),
            "max": a.max(0).astype(float).tolist(),
            "correlation_matrix": np.nan_to_num(np.corrcoef(a, rowvar=False), nan=0.0).astype(float).tolist()}


def train_model(x, y):
    torch.manual_seed(TRAIN_SEED)
    torch.use_deterministic_algorithms(True)
    model = trainer.MagneticOracle()
    opt = optim.Adam(model.parameters(), lr=0.003)
    xt = torch.from_numpy(x.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    losses=[]; per=[]
    for _ in range(EPOCHS):
        pred=torch.sigmoid(model(xt)); se=(pred-yt)**2
        loss=se.mean(); per.append(se.mean(0).detach().numpy().astype(float).tolist())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    return model, losses, per


def evaluate(model, x, y):
    with torch.no_grad():
        pred=torch.sigmoid(model(torch.from_numpy(x.astype(np.float32)))).numpy()
    return {
      "prediction_target_correlations": [corr(pred[:,i],y[:,i]) for i in range(6)],
      "prediction_feature_correlations": {
        f"output_{i}": {name:corr(pred[:,i],x[:,j]) for j,name in enumerate(trainer.FEATURE_NAMES)}
        for i in range(6)},
      "prediction_mean": pred.mean(0).astype(float).tolist(),
      "prediction_std": pred.std(0).astype(float).tolist(),
      "target_mean": y.mean(0).astype(float).tolist(),
      "target_std": y.std(0).astype(float).tolist(),
      "target_min": y.min(0).astype(float).tolist(),
      "target_max": y.max(0).astype(float).tolist(),
      "per_output_mse": ((pred-y)**2).mean(0).astype(float).tolist(),
    }


def main():
    # Calibration feature rows are measured from real renderer trajectories.
    raw_all, _, raw_all_summary = trainer.make_calibration_sequence_training_set(SAMPLES, TRAIN_SEED, 35)
    # Split base rows first, then independently permute each column within each split.
    raw_train, raw_holdout = raw_all[:TRAIN_ROWS], raw_all[TRAIN_ROWS:]
    raw_train_summary = {"source": raw_all_summary, "rows": TRAIN_ROWS}
    raw_holdout_summary = {"source": raw_all_summary, "rows": SAMPLES-TRAIN_ROWS}
    x_train=independently_permute_columns(raw_train, TRAIN_SEED+11)
    x_test=independently_permute_columns(raw_holdout, HOLDOUT_SEED+11)
    y_train=np.stack([trainer.feature_conditional_targets(r) for r in x_train]).astype(np.float32)
    y_test=np.stack([trainer.feature_conditional_targets(r) for r in x_test]).astype(np.float32)
    model, losses, per_history=train_model(x_train,y_train)
    test_eval=evaluate(model,x_test,y_test)
    train_eval=evaluate(model,x_train,y_train)

    # Save a normal renderer-compatible checkpoint. Its use on actual audio is
    # intentionally a secondary out-of-design diagnostic, not the identifiability gate.
    probe=torch.from_numpy(x_train[:1])
    with torch.no_grad(): golden=torch.sigmoid(model(probe)).squeeze(0).tolist()
    metadata={
      "name":"MagneticOracle v1",
      "checkpoint_format":trainer.CHECKPOINT_FORMAT,
      "architecture":trainer.ARCHITECTURE,
      "architecture_source":"recovered topology; no historical weights",
      "objective":"feature_conditional",
      "objective_provenance":"newly designed; target functions unchanged from prior retrains",
      "training_distribution":"calibration-sequence-independent-column-permutation",
      "training_distribution_provenance":"measured calibration renderer rows; each of 12 feature columns independently permuted; empirical marginals exactly preserved; joint tuples intentionally nonphysical",
      "decorrelation_method":"independent row permutation per feature column, seeded, no spectral warp",
      "training_samples":SAMPLES,"training_epochs":EPOCHS,"seed":TRAIN_SEED,
      "feature_schema": "same 12-value measured-audio feature schema; tuples are factorial feature interventions, not realizable audio windows",
      "target_mapping":"unchanged hand-designed feature_conditional_targets; recomputed after column permutation",
      "recovered_weights":False,
      "training_feature_names":trainer.FEATURE_NAMES,
      "training_distribution_summary":stats(x_train),
      "training_raw_calibration_summary":stats(raw_train),
      "holdout_raw_calibration_summary":stats(raw_holdout),
      "holdout_decorrelated_summary":stats(x_test),
      "training_output_target_correlations":train_eval["prediction_target_correlations"],
      "training_output_feature_correlations":train_eval["prediction_feature_correlations"],
      "golden_input":probe.squeeze(0).tolist(),"golden_output_sigmoid":golden,
      "scope_limit":"feature-space identifiability test only; actual renderer inputs remain correlated and cannot establish unique dependence from render correlations alone"
    }
    ckpt={"format":trainer.CHECKPOINT_FORMAT,"metadata_json":json.dumps(metadata,sort_keys=True),
          "state_dict":model.state_dict(),"loss_history":torch.tensor(losses,dtype=torch.float64),
          "per_output_loss_history":torch.tensor(per_history,dtype=torch.float64)}
    checkpoint=ROOT/"magnetic_oracle_feature_decorrelation.pt"
    torch.save(ckpt,checkpoint)

    # Recreate prior original resonator training distribution exactly by seed,
    # and matched calibration sequence distribution for target-range comparison.
    original_x, original_y=trainer.make_training_set(SAMPLES,TRAIN_SEED,"resonator")
    # Matched target statistics below are computed from the exact seed-1729 target
    # arrays collected in the preceding run and recorded in the prior execution log.
    # Reconstruct target ranges analytically where no clipping is active; the
    # complete six-output values are embedded from the deterministic measured run.
    matched_y=np.stack([trainer.feature_conditional_targets(row) for row in raw_all]).astype(np.float32)
    matched_target_std=matched_y.std(0)
    matched_target_min=matched_y.min(0)
    matched_target_max=matched_y.max(0)
    target_compare={"output_names":[f"output_{i}" for i in range(6)],
      "original_resonator_seed_1729":{"mean":original_y.mean(0).tolist(),"std":original_y.std(0).tolist(),"min":original_y.min(0).tolist(),"max":original_y.max(0).tolist()},
      "matched_calibration_sequence_seed_1729":{"mean":matched_y.mean(0).tolist(),"std":matched_target_std.tolist(),"min":matched_target_min.tolist(),"max":matched_target_max.tolist()},
      "matched_to_original_std_ratio":(matched_target_std/original_y.std(0)).tolist()}

    report={
      "title":"MagneticOracle v1 — preregistered feature-decorrelation experiment",
      "design":{"train_seed":TRAIN_SEED,"holdout_seed":HOLDOUT_SEED,"samples_total":SAMPLES,"train_rows":TRAIN_ROWS,"holdout_rows":SAMPLES-TRAIN_ROWS,"epochs":EPOCHS,
        "method":"Independent permutation of each of the 12 measured feature columns, preserving each column's empirical marginal exactly while removing cross-feature row pairing.",
        "important_limitation":"Feature tuples are intentionally nonphysical; this isolates learned feature dependence but is not a spectral warp and does not preserve realizable audio constraints.",
        "training_trajectories":raw_train_summary,"holdout_trajectories":raw_holdout_summary},
      "training_distribution":stats(x_train),"holdout_distribution":stats(x_test),
      "target_dynamic_range_comparison":target_compare,
      "training":{"initial_loss":losses[0],"final_loss":losses[-1],"per_output_mse":train_eval["per_output_mse"],"evaluation":train_eval},
      "holdout_evaluation":test_eval,
      "criteria":{"centroid_specific": {"output4_centroid_r_ge_0_4":False,"output4_zcr_r_lt_0_4":False,"output4_flatness_abs_r_lt_0_4":False},
        "proxy_learning":{"output4_centroid_r_lt_0_3":False,"output4_zcr_or_flatness_r_ge_0_5":False},
        "output2_protection":{"flatness_correlation_within_0_15_of_minus_0_929":False}},
      "verdict":"pending_criteria_evaluation"
    }
    f4=test_eval["prediction_feature_correlations"]["output_4"]
    f2=test_eval["prediction_feature_correlations"]["output_2"]
    centroid_ok=(f4["spectral_centroid"] is not None and f4["spectral_centroid"]>=0.4 and abs(f4["zero_crossing_rate"])<0.4 and abs(f4["spectral_flatness"])<0.4)
    proxy_ok=(f4["spectral_centroid"] is not None and f4["spectral_centroid"]<0.3 and (abs(f4["zero_crossing_rate"])>=0.5 or abs(f4["spectral_flatness"])>=0.5))
    output2_ok=(f2["spectral_flatness"] is not None and abs(f2["spectral_flatness"]-(-0.929))<=0.15)
    report["criteria"]={"centroid_specific":{"output4_centroid_r_ge_0_4":bool(f4["spectral_centroid"]>=0.4),"output4_zcr_abs_r_lt_0_4":bool(abs(f4["zero_crossing_rate"])<0.4),"output4_flatness_abs_r_lt_0_4":bool(abs(f4["spectral_flatness"])<0.4),"pass":bool(centroid_ok)},
      "proxy_learning":{"output4_centroid_r_lt_0_3":bool(f4["spectral_centroid"]<0.3),"output4_zcr_or_flatness_abs_r_ge_0_5":bool(abs(f4["zero_crossing_rate"])>=0.5 or abs(f4["spectral_flatness"])>=0.5),"pass":bool(proxy_ok)},
      "output2_protection":{"output2_flatness_r":f2["spectral_flatness"],"within_0_15_of_minus_0_929":bool(output2_ok),"pass":bool(output2_ok)}}
    report["verdict"]="centroid_specific_dependence_supported_in_decorrelated_feature_space" if centroid_ok else ("proxy_learning_supported_in_decorrelated_feature_space" if proxy_ok else "ambiguous")
    report["training"]["per_output_loss_history"]=per_history
    out=ROOT/"feature_decorrelation_acceptance.json"
    out.write_text(json.dumps(report,indent=2,sort_keys=True))
    md=["# MagneticOracle v1 — Feature-Decorrelation Experiment","",f"**Verdict: {report['verdict']}**","",
      "## Design","",report['design']['method'],"", "Each feature column's marginal is preserved exactly by permutation. The resulting tuples are deliberately nonphysical; this is a controlled feature-space identifiability intervention, not a physically realizable spectral warp.","",
      "## Target dynamic-range diagnostic","","| Output | Original target mean ± SD | Matched target mean ± SD | Matched/original SD |","|---|---:|---:|---:|"]
    for i in range(6):
      a=target_compare['original_resonator_seed_1729'];b=target_compare['matched_calibration_sequence_seed_1729']
      md.append(f"| {i} | {a['mean'][i]:.6f} ± {a['std'][i]:.6f} | {b['mean'][i]:.6f} ± {b['std'][i]:.6f} | {target_compare['matched_to_original_std_ratio'][i]:.3f}× |")
    md += ["","## Independent decorrelated holdout","","| Output 4 correlation | r |","|---|---:|"]
    for k,v in f4.items(): md.append(f"| {k} | {v:+.4f} |")
    md += ["","## Output 2 protection","",f"Output 2 vs flatness: **{f2['spectral_flatness']:+.4f}**; baseline tolerance ±0.15 around −0.929: **{'PASS' if output2_ok else 'FAIL'}**.","","## Acceptance criteria","",json.dumps(report['criteria'],indent=2),"","## Limitations","","The feature-space holdout is the correct place to test learned dependence under independent inputs. A real renderer trajectory still couples spectral features; any output-to-feature correlations measured on that render are deployment associations, not causal identification. Full render, musical quality, and cross-trajectory physical generalization remain untested.",""]
    (ROOT/"feature_decorrelation_report.md").write_text("\n".join(md))
    print(json.dumps({"report":str(out),"checkpoint":str(checkpoint),"verdict":report['verdict'],"criteria":report['criteria'],"output4":f4,"output2_flatness_r":f2['spectral_flatness'],"target_dynamic_range_comparison":target_compare,"initial_loss":losses[0],"final_loss":losses[-1]},indent=2))

if __name__=='__main__': main()
