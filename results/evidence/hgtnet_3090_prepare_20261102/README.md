# HGTNet RTX3090 compatibility — 2026-11-02

User changed rental hardware because3080 was unavailable. Actual pasted NVIDIA inventory is in `user_gpu_inventory.txt/.json`: RTX3090, GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1, total24576MiB/free24123MiB/driver610.57.04. This is user-reported inventory, not direct remote CUDA/training evidence.

## Delivered change

New `output/hgtnet_nwpu_smoke_3090_20261102.pyz`: 52,719 bytes; SHA256 `33e6f0a9ab449b314ba62b6961389db2d7464039c598b7775f7fef881bda19ff`; bundle ID `81180fc3bd4903e5af338be1ff446c2ba762fca5ea19ad51c8a8495a66fdac51`. Instructions are in `output/hgtnet_nwpu_smoke_3090_README_20261102.md`.

Only three production files changed:

- `resources.py`: explicit target name→exact memory mapping for3080/10240 and3090/24576. Defaults remain3080; UUID, idle query, software/precision and capacity checks retain their previous behavior.
- Runner: explicit target profile, selected actual name/memory at CUDA boundary, profile in status/provenance, and separate target/package scratch namespace. Direct source use and zipapp use both derive paths from the selected profile.
- Builder/bootstrap: target_gpu manifest identity, build option and process-boundary propagation; an inherited3080 environment cannot override a3090 bundle. Existing immutable install/byte checks stay active.

The17-file payload retains identical model/data/engine/config/helper bytes except the resource module and runner. Fourteen locally present payloads are unchanged; generated weight provenance is identical. Seed42, batch2, one epoch/180steps, train360/val90/test0, Adam1e-4, RGB224 and no augmentation/AMP/TF32 are unchanged. Existing Swin-B bytes/SHA and old3080 package remain unchanged. No new numerical training was needed to change the hardware gate.

## Current verification

- `resources/`:9 new tests RED→GREEN plus15 unchanged resource tests.
- `integration/delivery_red.log`, `scratch_red.log`, `delivery_green.log`: missing target behavior and the direct-source scratch regression reproduced before fixes;5 target-delivery tests now pass.
- `integration/targeted_suite.json/.log`: **39/39** changed hardware/CLI/default3080/delivery tests pass, no errors/failures/skips;0.350s runner wall.
- `review/`: fresh independent review,39 targeted checks and actual current-source zipapp subprocess probes, with inherited target3080, ML imports forbidden and NVIDIA boundary populated from the user's inventory. Final target3090, exact24576MiB/current UUID and scratch were correct; status not_started/no runs. These are boundary fixtures, not real CUDA.
- `integration/bundle_inspect_guard.json`, `actual_bundle_status.json`: actual final production pyz is target3090, stdlib-only inspection, no ML imports or runs, not_started. Actual server resources are missing locally as expected.
- `integration/validation.json`: **110/110** integrity/preservation checks pass: new package/current sources, target metadata, original52 historical artifact hashes,14 unchanged local payloads, old3080 pyz and full Swin-B bytes.

## Numerical regression limitation

The attempt to run the entire HGTNet/DFAGCN suite did **not** pass: it collected148 executable/failed-import entries, with1 failure and26 errors. The prior `/private/tmp/gavit-dfagcn-cpu-20261027/bin/python` now resolves the base interpreter without Torch, NumPy, torchvision or Pillow; the project `.venv` also lacks them. The temporary wheels directories are empty. Production source files are present and unchanged at the numeric core. We did not install a different cached Torch version or claim299 numerical tests passed.

`integration/full_suite_attempt.log/.json` preserves the full failure, and `full_suite_dependency_failures.json` records all27 failure/error names. The wrapped CLI failure is `test_dfagcn_cli.CLITests.test_help_exposes_all_workflow_commands`; its underlying dependency import is unavailable. Failed imports and downstream errors include the DF cache/component/features/graph/fusion/linear/model/pipeline and HGT attention/checkpoint/graph/model/engine/pretraining modules. The exact named entries and tracebacks are retained.

The prior285-test/full CPU numerical evidence remains a historical2026-10-31 result, not a fresh pass. We validated the changed stdlib hardware/delivery paths now and preserved all numerical implementation bytes. Actual3090 CUDA compatibility, memory, time and the real360/90 smoke are pending the user's new-package inspect/run. No formal accuracy, GPU launch, new rental, source reset, checkpoint overwrite, Git commit or push was performed by the agent.
