# Task 2 entry and delivery implementation report

## Delivered files

- `external_baselines/hgtnet/aid_batch_probe.py`: new stdlib inspect/status entry and lazy numerical run boundary.
- `tools/package_hgtnet_aid_batch16_probe.py`: independent source-only safe zipapp builder/bootstrap.
- `tests/test_hgtnet_aid_batch_probe_entry.py`: 8 behavioral tests, including complete-artifact status, tampering, exact16 selection/reads, unique idle3060, lock/no-overwrite, success/failure wrapper receipts, and consistent450-gradient rejection.
- `tests/test_hgtnet_aid_batch_probe_delivery.py`: 6 actual temporary archive/install/subprocess tests.

The entry and builder have an identical18-source whitelist. It retains the safe AID entry/data/config/model dependencies, excludes the old AID numerical engine and old builder, and includes the new entry/engine/builder. Source/config18 + weight-provenance metadata1 + bootstrap1 + deliverymanifest1 =21 ZIP members. The independent delivery schema is `hgtnet-aid-batch16-delivery-v1`; entry schema is `hgtnet-aid-batch16-capacity-probe-v1`; bootstrap environment key is `HGTNET_AID_BATCH16_DELIVERY_ENTRY`.

## Tests and evidence

Command for RED and final GREEN:

```text
python3 -m unittest tests/test_hgtnet_aid_batch_probe_entry.py tests/test_hgtnet_aid_batch_probe_delivery.py
```

- `red.log`: initial11 failures because the new entry/builder were absent, before writing production files.
- `green_initial.log`: initial implementation exposed one expected fixture-path problem (`/var` is a symlink on macOS); fixtures now use their resolved temporary directory. The initial complete fixture was rejected for the same safe no-follow path reason. No production path guard was relaxed.
- `green.log`:11 passed after fixture-path correction and respecting actual `parameter_counts` extra per-group metadata.
- `green_extended.log`:13 passed after adding exact16 inspect byte-scope and successful wrapper receipts.
- `green_final.log`:14 passed, including consistent450 tensor receipt rejection. The final command ran in0.481 seconds.
- `validation.json`: command/result, closure equality18/21, and explicit execution boundaries.
- AST parsing of all4 new ownedPython files passed. Own source whitelist equality and exclusion checks passed.

Test counts count methods, not individual tamper subcases. Guarded real temporary zipapp subprocesses perform safe real installation and run inspect/status with Torch/vision/NumPy/PIL imports forbidden. They do not install dependencies. Their inspect honestly remains readyfalse in this environment.

## Behavior and self-review

Selection uses the existing manifest metadata validator, then takes exactly the lexicographically first frozen smoke train path for labels0 through15. It never reads val/test image bytes, recovers archives, recomputes a split, or deduplicates images. inspect validates source/config, the complete initialization weights, manifest and only those16 image bytes. run repeats source/manifest/config/weights/image checks at the end.

run auto-selects a unique idle RTX3060/12288MiB using the existing strict AID GPU helper; `_runtime_and_torch` binds the UUID before importing Torch and checks actual software/CUDA/FP32/threads/determinism/capacity. It refuses existing output, symlink paths and an existing exclusive run lock. Only run imports the numerical engine.

The actual HGTNet boundary checks74612163 total/trainable parameters and451 actual parameter tensors. Final artifact validation requires five precise gradient groups summing451 tensors, finite positive groupL2, no missing/nonfinite gradients and five true representative updates. It also verifies16unique/64processed/4updates/1+3, three finite positive measured times and their mean/median/throughput, complete actual3060 runtime and identities, empty accuracy/checkpoint=false, finite parameters, Adam settings, ending-check receipt, exact selection/provenance, all4 durable step files plus aggregate, firststep, optimizerprogress and complete progress. Exit0 alone or a skeletal summary cannot pass. A wholly consistent synthetic artifact fixture tests acceptance behavior; it is not evidence of a real GPU run.

Each callback records the actual updated step in an exclusive per-step file, aggregate list, optimizerprogress and firststep before updating general progress. Failure uses durable receipts/event count and exception-observed steps rather than resetting optimizersteps to zero. Post-update diagnostic failure fixture reports trainingstarted=true and1 actual update.

Timing/peak-reset semantics come from Task1 unchanged: model/input placement happens before the engine reset, so measured CUDA allocator peaks do not claim full initialization peak. Entry preserves the engine's compute/memory scope fields. Walltime after preflight additionally includes imports, decoding, initialization and all bounded updates.

The delivery algorithm intentionally follows the already-tested old AID safety implementation with its separate whitelist/schema/runner/date/environment key. It rejects duplicate/unsafe/colliding/nonregular archive paths, changed source bytes/manifests, foreign or altered installation trees, unexpected Python/code sources and symlinks. Only explicit inspect/status/run/help actions dispatch; unknown/default actions do not install or train. The builder refuses overwrite.

## Constraints and remaining unknowns

No old production/config/model/delivery files, result tables or diary were edited by this task. No final named output pyz was built: temporary test archives were removed with their fixtures. No staging/commit/push, network/dependency install/download, remote command, GPU access, original AID image/weight access or actual numerical training occurred. Only the2 owned newtest files were run, as explicitly requested; root owns old regressions, final build, protected-file snapshot and integration review.

The real3060 batch16 capacity, allocator peaks, numerical timing, success of full451-tensor gradients and4 updates remain unknown until the user runs the final bundle on the intended assets/device. The CPU-free entry fixture substitutes the numerical boundary only because executing that real dependency requires unavailable GPU/assets; assertions inspect the real wrapper's durable files and completion behavior. It makes no numerical/capacity/accuracy claim. Formal AID trainer/performance remain outside scope.
