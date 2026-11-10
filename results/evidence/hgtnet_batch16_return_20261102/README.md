# Batch16 target run return evidence

State: **SERVER_REPORTED_BATCH16_PASS_STATUS_IDENTITIES_MATCH**. User-pasted actual run output is preserved in [reported_output.txt](reported_output.txt), followed by [reported status](status_reported.json) and [26-check status acceptance](status_validation.json). These are reported server observations, not direct filesystem retrieval; current source/package checks are local checks.

- [validate_return.py](validate_return.py) parses original CUDA/progress/COMPLETE lines; [validation.json](validation.json) has29/29 passed checks and scoped arithmetic.
- [Independent review](review/README.md):26/26 checks, no anomaly or repeat-GPU-run justification. Actual local delivered zipapp hash/current payload and normal completion path checked separately.
- Real4updates/64presentations, one warmup+three measured; all5gradient groups/451parameter tensors finite and all5representative group updates. No population-performance conclusion from loss on reused16train records.
- Median0.20135353s/update,79.46222746images/s computation only. Conditional scaled315/394steps per train/refit epoch63.43/79.33s;70900maximum updates3.96555h pure update computation. No full-data loader/augmentation/validation/test/save/startup/billing measurement or formal ETA.
- Reported CUDA25295323136B is total exposed capacity. Status supplies allocated4732054016B=4.40707GiB/reserved5215617024B=4.85742GiB, wall10.059117s,exit0,matchingconfig/manifest/weights/UUID. Accepted fixedbatch16 resourcecheck; remote source/runtime/summary raw bytes have not been retrieved.
- [Human report](../../../results/hgtnet_batch16_probe_3090_20261102.md) / [machine record](../../../results/hgtnet_batch16_probe_3090_20261102.json). Existing status has been received; do not rerun the accepted probe.

No source/config/recipe changes, new local numerical training, remote execution, new rental, formal table edits, Git commit/push or destructive cleanup in this receipt turn. Formal trainer/recipe/budget preparation remains separate.
