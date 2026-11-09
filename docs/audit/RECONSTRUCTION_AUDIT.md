# Amorph Neural Clavecin — Reconstruction Audit

Author attribution requested: Lewis James Seals (user-provided attribution; not independently verified).

## Scope
Reviewed the uploaded `previous-builds-neural-clavesin.zip` snapshot (20 entries; archive payload 4,066,205 bytes; ZIP size 2,077,497 bytes). A byte-preserving copy of the original archive is included, alongside extracted files.

## Findings
- The archive contains browser HTML/CSS implementations, README variants, a PDF report, and `file.txt` with neural/DSP critique and a Python/PyTorch prototype.
- README variant `README (3).md` describes an Oracle neural network as 8→16→2, with ReLU and Adam; intended outputs control `field_boost` and `snap`.
- The browser builds include JavaScript weight initialization and/or neural training code. These are model implementation clues, not evidence of a separately exported, trained weight checkpoint.
- The Python `MagneticOracle` prototype in `file.txt` defines a 12→128→128→6 network and creates it with default random initialization. No serialized PyTorch checkpoint (`.pt`/`.pth`), ONNX model, safetensors file, or separately exported neural weight blob was found in the archive listing. Thus this archive does not appear to contain trained binary weights; weights may have been runtime-generated or stored outside this snapshot.
- README (3) references CC0-1.0 license badge and an upstream patch library, while README (4) has a License section. The archive listing contains no standalone `LICENSE`/`COPYING` file. Badge/reference alone does not establish that every component in this archive is CC0 or MIT.
- Some README materials reference third-party Amorph Instrument, Cmajor patches, and an external patch/wiki library. Treat these as separate provenance/license checks before redistribution.

## Reverse-engineering reconstruction plan
1. Preserve this archive as the baseline snapshot; record hashes for source versions.
2. Select the authoritative browser build and recover the Oracle's exact initialization, forward pass, optimizer, data features, loss, and synthesis/MIDI mapping.
3. Separately reconstruct the Python magnetic-clapper model; fix the detached-loss issue documented in `file.txt` before treating its training as meaningful.
4. Add a deterministic seed, model export/import format, weight serialization tests, and inference/training regression tests.
5. Recover actual learned weights from any original runtime export, browser storage, external file, or training session. Without that, recreate weights by training/reinitializing—not exact recovery.
6. Add license only for components whose ownership and third-party terms are verified. Suggested attribution line if confirmed by author: `Copyright (c) 2026 Lewis James Seals`. Do not label third-party patches/assets MIT without permission or compatible terms.

## License status
No standalone license file was present in the archive. MIT is a valid choice for original code if the copyright holder elects to license it under MIT; this audit does not assert that all bundled material is solely authored or legally clear.
