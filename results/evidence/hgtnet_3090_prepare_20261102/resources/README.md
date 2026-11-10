# HGTNet RTX 3090 resource gate evidence — 2026-11-02

The user reported NVIDIA GeForce RTX 3090, UUID `GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1`, total 24576 MiB, free 24123 MiB, driver 610.57.04. These values are a local fixture copied from the user report; no live GPU query or GPU execution occurred. Runtime software values in tests remain synthetic fixtures.

`resources.py` now exposes `TARGETS` for exact full names and memory sizes: RTX 3080 / 10240 MiB and RTX 3090 / 24576 MiB. Both `select_gpu` and `check_runtime` accept keyword-only `target_name=TARGET_NAME`; the default remains the existing RTX 3080. Unsupported target names fail before inventory/runtime processing. UUID, exact actual name/memory, idle/process query, free memory, software versions, and precision gates remain required.

- `red.json` records the exact test command/environment and failing exit status; `red.log` shows 9 assertion failures for missing target support, not TypeError.
- `old_gate_user_fixture.json` shows the original 3080 gate rejected the user-reported 3090 inventory, as expected.
- `green.json` records exact commands/environment for the untouched original resource suite (15 passed) and the new target resource suite (9 passed); corresponding logs are retained.
- `source_audit.json` records source/package SHA256 and confirms every previous resource function except `select_gpu` and `check_runtime` has an identical AST to the delivered 3080 package. The added `_target_memory` validates the allowlist. `resources.diff` contains the exact source change.

Only CPU/stdlib resource gates were verified here. This evidence does not establish CUDA compatibility, GPU training success, speed, memory use, model accuracy, or paper results.
