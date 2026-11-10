# Numeric engine TDD evidence

New bounded run_batch_probe engine, actual Torch2.2.2 CPU tensor operations with a small trainable45-class fixture; this verifies loop/diagnostic behavior, not full-HGT batch16/GPU capacity.

- hgtnet_batch16_engine_red.log:6 tests collected,15 expected missing-engine assertions including subtests; no implementation existed.
- hgtnet_batch16_engine_scope_red.log:7 tests,1 expected assertion failure because generic engine was assigning cpu_fixture from device; now engine explicitly returns caller_provided_tensors. Verified data wrapper binds real_nwpu only after actual record/file checks.
- hgtnet_batch16_engine_green.log:7tests passed/exit0, real4Adam updates/three positive timed batches, independent finite gradients/group change checks, immutable inputs, exactoptimizer defaults, boundedbudget/dtype/shape/frozen failures, no successful event beforefailedforward, and a completed update retained if post-update diagnostic fails.

Entire333-related-test suite and independent28-test review include these current7 tests. Timing/peak source scopes are explicit and GPU results remain pending. Core model and old smoke engine unchanged.
