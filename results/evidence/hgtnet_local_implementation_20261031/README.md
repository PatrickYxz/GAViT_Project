# HGTNet local implementation evidence — 2026-10-31

Scope: independent implementation and synthetic CPU engineering checks only. No HGTNet dataset result, target GPU run or formal training.

- `source_review/`: bounded primary-paper review and unresolved assumptions; exact author topology remains unverified.
- `attention/`, `graph/`, `pretrained/`, `checkpoint/`: module behavior tests and valid API/regression RED→GREEN evidence.
- `integration/`: root config/model/smoke RED→GREEN logs, full related-suite run, commands/exit codes and final audit.
- `full_cpu_01/summary.json`: actual full 74,619,858-parameter candidate initialized from verified Swin-B candidate, B1/B2×224, finite gradients across all trainable tensors, one optimizer step, exclusive checkpoint, then different-process reload with both errors0. `training_summary.json` intentionally retains completed=false because it predates reload; final status is in `summary.json`.
- `full_cpu_01/pretraining.json`: actual value load report, not the separate meta-schema audit. Source 353tensors, compatible318, intentionally-new155, intentionally-skipped35. Coverage counts state elements, including buffers.
- `review/`: independent code review and small real CPU integration probe. This probe is separate from full-model numerical evidence.
- `preservation/`: 52/52 existing metric/source/DFAGCN package hashes matched; newer DF analyses/advisor drafts have current snapshots only.

`full_cpu_01/model.pt` and `inference_fixture.pt` are preserved locally but excluded from Git by the scoped `.gitignore`; their complete file hashes remain in the JSON evidence. Official source weights are under the already ignored `checkpoints/`. No old checkpoint is overwritten.

The local model identity hashes six source modules. All package Python files are additionally inventoried in the final validation report. Torch/vision versions are runtime evidence, not part of the six-file hash; rerun numerical checks after a library/device change. The CPU smoke seed and optimizer are disposable engineering inputs, not a chosen remote-sensing training recipe.
