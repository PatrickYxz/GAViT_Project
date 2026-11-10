# Final review: root review after independent reviewer became unavailable

The final-review agent hit its usage limit before returning an audit. No independent final-review success is claimed. Root read the complete416-line initial runner plus subsequent changes, delivery additions, tests and the existing extraction/calibration/refit/test APIs; target CUDA/full NWPU was not executed locally.

## Fixed findings

1. Important: failure.json always retained formal_training_started=false when a later phase failed, even after real training. Root added a regression that first failed; valid CUDA batch events now preserve true, a failure at calibration before a first observed batch remains unknown, and malformed/incomplete events cannot establish training. Pre-training failures retain false. Three relevant RED→GREEN cases are recorded.
2. Important protocol traceability gap: phase receipts had timing/peaks but no saved-source/runtime identity link. The orchestration fixture failed on the missing identity; stage receipts now bind manifest, recipe, weights, physical UUID and SHA of saved source/runtime records. The source and runtime hash values are checked against actual written files.
3. Important integration gap caught before delivery: zipapp bootstrap did not forward status. A real zipapp status test failed first; the minimal whitelist change passes while run remains explicit.

No unresolved code findings were identified. No deferred polish changes.

## Reviewed behavior and evidence

- Stdlib-only inspect/status, exact physical10GiB RTX3080/UUID, precise manifest/weights/recipe/software gates, positive idle capacity inventory and40GiB free-disk gate.
- New pool6300 → actual copied5040/1260 split caches → calibration formal/no subset → fresh6300 refit → test25200 extraction/evaluation; selection uses unchanged pipeline, never test labels for hyperparameter choice.
- Actual small ZIP recovery, correct test recovery counts, path/hash/duplicate checks, no old data/cache writes; output mkdir is exclusive. No epoch resume is promised.
- Actual generated nohup command executes a harmless substitute executable and rejects an existing launch log. Real pipeline stdout is teed to new run.log. Completion comes after artifacts, summary and exit0; startup still requires actual server batch logs.
- Actual numerical small cache slice and core Torch/LIBLINEAR tests, plus a clearly labeled orchestration fixture substituting unavailable CUDA/full data. These fixtures do not count as real3080 formal execution.
- Full relevant139 methods passed. An earlier138-method run also passed; the later added shell case triggered local OpenMP initialization/fork failures in a complete run, preserved by name. Local test-only KMP_INIT_AT_FORK=FALSE yielded139/139; no core algorithm, old helper, server environment or recipe changed.
- Final pyz embedded bytes/bootstrap match current files, actual stdlib status works, local inspect verifies full official VGG SHA and correctly blocks missing server data/GPU/versions/binaries without creating runs. Twenty existing inputs including old bundles,19 metrics and12 legacy sources remain byte-identical.

## Deliberately not established

- Real full31500-image target execution, final OA/macro-F1/per-class results, total memory/runtime/cost: require the pending server formal run; the approved probe is not this evidence.
- Original paper numerical equivalence and multi-seed stability: outside this disclosed independent recipe; explicitly not claimed.
- Legacy full-repository regression: existing missing pytest/timm/torch_geometric remain documented from earlier runs; no whole-repo-green claim.
- Epoch checkpoint resume: existing core only saves model weights/history, not optimizer/RNG state; partial runs are preserved and rejected rather than silently resumed. Recovery can be planned from actual failure evidence.

There was no remote action, new training expense, Git commit/push or old artifact modification by root.
