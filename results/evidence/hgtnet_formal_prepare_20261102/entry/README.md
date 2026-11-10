# Formal entry implementation evidence

Owned changes: external_baselines/hgtnet/formal_entry.py, tests/test_hgtnet_formal_entry.py and this entry evidence directory. Old numerical/data/runtime/builder files were not edited by this subtask.

## Behavior

- inspect/status use standard library only. Fixed real counts5040/1260/6300/25200, frozen formal recipe/model/config/weights/source binding, current idle3090 UUID, conservative6GiB work+2GiB scratch gates.
- Full pool6300 images verified/recovered before training; an archive is full-SHA verified at run start if supplied, so an invalid future-test source fails before learning. Without an archive, full existing pool and test image bytes must verify; test is never decoded/evaluated by inspection.
- Fresh run owns binding.json/run_identity.json plus scratch/.formal_owner.json. Resume checks exact binding/source/recipe/paths, full-SHA compatible completed-epoch snapshot receipts and scratch ownership using only stdlib. Tensor/optimizer validity remains the engine's actual load gate.
- Each invocation gets attempts/000N with immutable command/preflight/runtime/run.log and preserved prior summary/progress/exit/marker copies. Main status files are atomically replaced only inside the owned run; no prior attempts or foreign/user files are deleted. Scratch may be reconstructed on a new instance at the recorded path.
- Fresh candidate model_factory, deterministic epoch-seeded shuffle and separate train/eval datasets. A test loader request fails until the engine's before_test callback passes full source/input/pool checks. Final completion also requires actual test-root creation and full final input/image rechecks.
- Failed runtime gives no successful training claim; successful update receipts survive later failure. Resume stage=resumed uses the engine's actual recovered optimizer counter. Input mutation at before_test prevents test execution. Completion requires completed engine, valid final gates, summary, exit0 and exact marker.
- status returns compact OA/macro-F1 and best_val_metrics plus hashes/reload/phase metadata, avoiding25200 prediction and confusion arrays. Full predictions/metrics remain the engine's saved files.

## TDD and limitations

red.log:12 valid failures for missing formal-entry API. green.log:13 entry tests passed in27.153s before the final best_val_metrics compact-field adjustment. status_red.log:one expected missing test_metrics key error. status_best_red.log:one expected missing best_val_metrics key error. status_best_green.log:the updated short-status case passed in3.031s after the final source change. Root integration will run the complete current related suite; this evidence does not retrospectively label the older13-test source hash as the final one.

The fixture has a real31500-row,45-class manifest and31500 hashable byte files under temporary paths; these bytes intentionally are not JPEG images and no ML execution is claimed. GPU runtime and complete engine execution are mocked only at their explicit boundaries. Snapshot fixtures validate preflight SHA/ownership, not Tensor deserialization; numerical checkpoint/engine tests cover that separately. Independent child-process import guards confirm inspect/status do not import Torch/vision/NumPy/PIL. No GPU, server or formal metrics were produced.

Final py_compile and git diff --check passed. Source/test SHA values are in validation.json. Local pinned Python3.12.13 CPU environment was used; target Python3.11.8/CUDA is verified only by actual runtime entry execution.

## Fresh-review repair: interrupted pool recovery

The independent review found that an interrupted pool extraction in a replacement instance could leave scratch/pool without recovery.json while an epoch snapshot remained valid. A fixed scratch ownership identity then made repeated resume target that same incomplete directory. partial_pool_red.log reproduces this with an owned run, SHA-valid snapshot receipt and retained partial bytes; the original code passed pool/resume=True instead of the required fresh attempt destination and the test failed.

The entry now preserves incomplete pool and uses pool_attempt_000N for the new invocation, exactly as the deferred test recovery already does. Complete verified pool recovery is still reused normally; no partial/user file is overwritten. final_green.log verifies all14 entry tests against this final source,0failures/errors,32.707s. The existing source/test hashes were retained as initial_entry_sources, with current hashes and final14-test receipt added to validation.json. Root is rerunning the current whole integration/CPU pipeline after this source change.
