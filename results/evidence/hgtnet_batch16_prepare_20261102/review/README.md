# Independent batch16 probe review

Reviewed new entry, numeric engine, registered builder routing and their three test files. No remaining actionable blocker found in the reviewed source hashes in `validation.json`.

- Fresh combined run in the restored pinned CPU runtime: 28 tests, 0 failures, 0 errors (13 entry + 7 actual CPU tensor engine + 8 delivery). `targeted_tests.log` records the unittest cases; expected negative-case tracebacks were printed to the command stream.
- The 16 old source/config payload files from the delivered, real3090-completed `hgtnet_nwpu_smoke_3090_20261102_fix1.pyz` are byte-identical to current counterparts. Builder bootstrap is intentionally updated for registered routing; old pyz was not modified.
- Engine is fixed to batch16/warmup1/measured3, FP32/all trainable and Adam lr1e-5/weight_decay5e-5. It checks real gradients/updates and preserves observed optimizer-step evidence on post-update diagnostic failure. All four updates reuse a single disposable model and batch.
- Wrapper initializes from the verified Swin-B candidate, does not load learned smoke weights, selects only 16 train rows and validates selected image bytes before/after. No validation/test decoding, accuracy or checkpoint saving path exists in this probe.
- Entry reuses exact current-UUID/physical3090/actual-CUDA/precision/version gates; output and scratch must be new. Status requires successful summary, exit0 and completion marker.
- Registered runner dispatch and rejection of missing/unregistered runner entries are exercised in genuine zipapps by the delivery tests. Production bundle identity/actual production `inspect` and `status` remain root integration responsibility.
- Initial timing-scope concern resolved: engine uses `compute_timing_scope` for synchronized compute intervals, separate from wrapper wall-clock `timing_scope`. Input provenance is assigned by the wrapper, not inferred from CUDA execution inside the generic tensor engine.

These are engineering source/CPU/stdlib delivery checks. They do not establish real3090 batch16 capacity, target throughput, a formal recipe, formal epoch ETA or benchmark accuracy. Actual target `inspect`→`run`→`status` result is still required.
