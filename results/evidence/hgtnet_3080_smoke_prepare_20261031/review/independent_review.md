# Independent HGTNet NWPU smoke review — 2026-10-31

Read-only production review by a fresh agent that authored none of the runner,
engine, resource/data adapters, package builder, or production tests. Reviewer
wrote only the small CPU probes and this evidence under `review/`.

**Verdict:** no reproduced Critical or Important issue in the reviewed scope.
This is local source/behavior review, not acceptance of an RTX3080 or real NWPU
run. No remote action, paid job, Git mutation, or formal trainer was exercised.

Reviewed the smoke design/plan, current research diary/result tables, the three
new HGTNet adapters/engine, runner, package builder, their tests, the reused
manifest/ZIP helper, and checkpoint/pretraining/gradient-report interactions.

## Evidence obtained

- Existing `test_hgtnet_nwpu*.py`: 33 tests passed, 1.322 seconds. This covers
  actual JPEG transforms/order/byte changes, fixed manifest shape/360+90 selection,
  train/val-only ZIP recovery, actual tiny CPU optimizer/validation/reload,
  SHA/metadata rejection, wrapper failure receipts and actual zipapp dispatch,
  immutable installation/source tampering refusal. Original tool output retains
  the complete run; the later root suite is the final consolidated log.
- `test_hgtnet_resources.py`: 15 tests passed, 0.044 seconds, in
  `resources_tests.log`. Includes fresh UUID selection, process-query failures,
  exact runtime settings, version metadata, import safety, disk and path checks.
- `failure_probes.py`: three independent small CPU probes passed, 0.082 seconds,
  in `failure_probes.log`. A second training-batch exception leaves the genuine
  first optimizer receipt and no checkpoint; a fresh reload with a deliberately
  wrong forward output is rejected; a wrapper exception after actual tiny CPU
  training/validation/reload preserves `training_started=true`, writes
  `completed=false`, `run.exit=1` and status `failed`. The wrapper runtime boundary
  is intentionally injected for this last probe; it is not CUDA evidence.

## Review conclusions and limits

- The CLI runs a fixed non-augmented RGB224 train360/val90/test0 epoch. Dataset
  access rechecks selected image digests; no test image decoder or extraction
  path is called. ZIP recovery verifies the full archive SHA and each selected
  image before use. Inspect's pending ZIP state does not claim this verification.
- Completion follows training, finite-value checks, ordered validation labels,
  actual checkpoint persistence and numeric reload, then source/config/manifest/
  weight and selected image rechecks. The wrapper suppresses completion on a
  raised engine or late-check exception and derives started state from the
  observed first optimizer receipt. Status requires summary completion and exit0.
- The real-image checkpoint schema binds config, selected validation identities,
  manifest, source and pretraining/runtime metadata. Real NWPU checkpoint inputs
  are distinct from the prior synthetic checkpoint schema; CPU fixtures remain
  explicitly marked. No existing synthetic checkpoint is used as training input.
- Runtime checks occur after imports and selected CUDA visibility, with fixed
  Torch/vision/CUDA/cuDNN/precision/thread/seed settings. Fresh inspect UUIDs are
  used rather than historical physical identifiers. Actual target execution is
  still required to establish the target's memory, runtime and CUDA behavior.
- Work/scratch budgets include a full smoke checkpoint and selected recovery,
  combine on one volume, and prohibit existing/containing destinations. Measured
  wall and allocator peaks have explicit scopes; no formal throughput/fee claim
  is justified by the local tests.
- Package installation verifies exact payload bytes and refuses foreign,
  modified, symlinked or extra importable source content. Existing model core,
  config and historical artifact preservation is independently handled by the
  root agent's final SHA audit, not claimed from Git's untracked-file diff.

One readiness note was sent immediately: an exact ZIP byte-size gate beside the
full SHA can unnecessarily block the fallback. Root traced the number to the
user's server transcript and removed the redundant equality gate. I inspected
the corrected nonempty-file/pending-SHA branch. It is a readiness usability
note, not a demonstrated integrity or false-completion flaw. I also inspected
root's change reducing first-step parameter clones to group representatives and
requiring recorded updates in every positive-gradient architectural group. The
root's final suite and full-model CPU fixture provide the final integrated
validation for these concurrent refinements.

No additional gate or approval loop is requested by this review. The bounded
GPU smoke remains pending until the user runs the delivered package and returns
its receipts; the formal trainer and formal accuracy result remain outside scope.

## Receipt regression addendum

Root identified a minor completion-report clarity issue: ZIP recovery completed,
but the final summary copied the preflight's pending `data_source`. At root's
request I added
`RunnerTests.test_completed_zip_recovery_receipt_replaces_pending_inspect_state`
to `tests/test_hgtnet_nwpu_runner.py`, preserving existing tests. The valid RED
in `data_receipt_red.log` runs eight tests with precisely one failure: the final
data-source assertion sees the old pending string after `run_smoke` returned0,
actual tiny CPU JPEG training/validation/save/reload completed and numerical
reload error was0. The regression also requires `archive_sha_verified=true`.

This fixture uses six real JPEGs and the real dataset/image hash checks, engine,
checkpoint operations and final source/config/file identity checks. The frozen
manifest protocol, ZIP staging, CUDA and pretrained-candidate boundaries are
explicit fixtures. The engine runs with `cpu_fixture` scope and train4/val2/test0;
it does not claim NWPU or CUDA execution. Root owns the production fix and final
GREEN run/package rebuild for this small receipt refinement.
