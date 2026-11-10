# Independent AID smoke review

## Final verdict after the P2 correction

Specification compliance: PASS for the local preparation and delivery-source scope reviewed. Code quality: PASS, with no remaining actionable finding from this independent review. The target RTX3060 run and final delivery artifact remain outside this completed source review.

P2 is closed on `aid_entry.py` SHA `22acb01983492526e6a24bdb2acf303cd077f8023242b3125a2bf3f7708c39d7`. The data and engine frozen SHAs remain unchanged. The new shared run/status validator checks real-AID scope, frozen identities, exact counts/protocol/runtime/CUDA metadata, binding, history, actual first-step diagnostics,120 updates,60 ordered predictions and independently recomputed30-class metrics, checkpoint receipt/bytes and end-check success. The end-check receipt is generated only after the real end-of-run checks return successfully.

Independent narrow verification:4 targeted completion regressions passed, and all15 checks in `p2_delta_validation.json` passed. The original invalid fixture was copied with the current source identity and now returns `incomplete` while reporting `source_identity_matches=true`; this confirms rejection by the completion gate instead of only stale source metadata. The already generated actual full-model CPU artifacts match the engine-owned field schemas expected by the gate, while their CPU scope correctly remains ineligible for real-AID completion. Engine and entry calculations and the actual generated field shapes were inspected for compatibility; no real-AID/GPU completion is claimed.

The follow-up used standard-library checks with ML imports actively forbidden. It did not rerun the complete model, engine suite, combined suite or training. Current17-source SHA receipts and the exact targeted-test log are preserved in `p2_delta_validation.json` and `p2_delta_tests.log`.

## Initial review retained below

Scope: the new AID data, engine, entry, builder, configuration and four tests; read-only inspection of reused dependencies solely to establish the delivery closure. No reviewed source was edited. No GPU, installation, network, Git mutation, formal training or repeated test suite was used.

Initial specification verdict: one open P2 prevented approval of the completion/status contract. The data, engine and delivery paths otherwise conform to the reviewed design: frozen manifest selection, duplicate group preservation, no test-image reads, same hashed bytes for RGB600 decoding, separate30-class configuration, seeded shuffle, five-group first update diagnostics, durable progress, independent checkpoint reload, runtime gates and an aligned17-member source whitelist.

Initial code quality verdict: needed the same narrow P2 correction. No other actionable issue was established in this review. The final delivery bundle and target RTX3060 execution remain pending; preliminary package evidence is not treated as final-source or GPU evidence.

## P2: status accepts incomplete or incompatible completion evidence

Location: `external_baselines/hgtnet/aid_entry.py:409-414`; the current positive fixture at `tests/test_hgtnet_aid_entry.py:102-112` also endorses this behavior.

The completion gate accepts `input_scope=cpu_fixture`, CPU/float64 runtime, `training_started=false`, three invalid input/config/weight SHAs, no binding/runtime/selected-input/first-step/history/prediction files, and arbitrary non-Torch checkpoint bytes if the summary supplies their SHA. It reports `complete` solely from a small set of summary/progress fields plus current source and checkpoint byte hashes. This contradicts the explicit requirement that partial/malformed evidence cannot be complete and that CPU fixtures cannot represent the real AID/GPU run.

Bounded reproduction: `status_partial_repro.py` and its preserved `status_partial_repro.json`. This imports no ML package and performs no model operation. It reports `status_completed=true` and `status_state=complete` for the incompatible fixture.

Required correction: use a stdlib completion validator for frozen identities, real-AID scope and actual runtime metadata, strict integer counts/epoch/update fields, and the existence/consistency of binding, runtime, selected inputs, successful first-step diagnostics, complete history and60 ordered predictions with30-class metrics. Preserve current checkpoint-byte/source checks. No ML import, tensor deserialization or image reread is required from status. The incompatible/partial fixture must remain incomplete, while a complete consistent real-scope fixture must pass.

## Evidence scope

The current data SHA is `5fdc086cb0512086f93189955f8bda274fd6a9332a2c54afbda911881147d0f8`; engine SHA is `bedf4f592fcd3b968ea6641a0ae82353292172a09dbd5f3bff713cace17cc848`. These match the task1 frozen receipt. Entry/builder source lists agree exactly, all17 source/config paths were hashed, Python source AST parsing passed, and local project import dependencies are covered by the list. Task1's21 and task2's18 existing test results were read as prior evidence, not rerun or promoted to target-device evidence.

Initial status: root accepted P2 and returned the narrow change to the entry implementation agent. The initial reproduction is retained as pre-fix evidence; the final source identity and closure appear above.
