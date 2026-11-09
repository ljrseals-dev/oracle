# Repository layout and retention policy

This repository is an evidence-preserving research snapshot. Reorganization improves navigation without rewriting provenance or deleting tracked artifacts.

## Canonical paths

- `source/recovered/file.txt`: original supplied source, retained verbatim.
- `src/magnetic_oracle/`: canonical trainer and renderer entry points.
- `experiments/<name>/code/`: experiment-specific code snapshots.
- `experiments/<name>/reports/`: interpretation and experiment reports.
- `experiments/<name>/results/`: acceptance records, diagnostics, and training logs.
- `artifacts/checkpoints/`: research checkpoints.
- `artifacts/audio/`: rendered WAVs and comparisons.
- `artifacts/metrics/`: retained legacy metrics.
- `docs/audit/`: reconstruction audit, hashes, and weight-search evidence.
- `docs/research/`: supporting research notes.
- `archive/`: historical snapshots and redundant/legacy copies.

## Retention rules

1. Preserve the supplied source and all evidence artifacts.
2. Keep experiment-specific code near the results it generated.
3. Keep one canonical source entry point; retain historical copies for traceability.
4. Do not silently change result meanings or promote a research checkpoint to production.
5. Distinguish physical-render evidence from synthetic feature-space interventions.
6. Update links when paths change; do not rewrite Git history.

Some archived copies may be byte-identical. Their presence does not imply independent implementations.
