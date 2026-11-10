# Batch16 stdlib entry evidence

- Owned implementation: `external_baselines/hgtnet/batch_probe.py` (new file).
- Owned regression tests: `tests/test_hgtnet_batch_probe_cli.py` (new file).
- `red.log`: eight tests failed because the new entry API did not exist, before implementation.
- `status_red.log`: measured throughput/status regression failed because the short status omitted the engine's median timing field; the status projection was then extended.
- `green.log`: current 13 tests pass under system Python3.14; `cpuenv_green.log`: the same 13 pass under the restored CPU environment at `/private/tmp/gavit-hgtnet-probe-cpu-20261102/bin/python`.

The tests construct and validate a complete31500-record manifest with real hashes and selected16 byte files, exercise real safe ZIP extraction, and verify no validation/test images are recovered. Only dependency/GPU inventory and the actual Torch execution boundary are replaced. They cover current-UUID command binding,3090-only profile, cache selection/fallback, full manifest digest rejection, no-ML inspect/status, new-path protection, runtime failure before training, first-update evidence retained after failure, and final image mutation rejecting an apparent engine success. Fixture bytes are deliberately not real images: these tests establish stdlib orchestration and receipts, not real decoding/CUDA/numerical training.

The entry reuses the existing runner's physical/CUDA-capacity window and actual runtime gates. It selects one first-sorted training record per label0..15 from the frozen smoke subset, reads only those16 image paths, and calls the separate bounded engine with batch16/one warmup/three measured updates. It never invokes the450-image smoke, validation/test loader, formal training, checkpoint creation, Git, network, or a remote GPU. Real GPU capacity/timing remains pending execution of the delivered probe. Core model, config, data adapter, old engine, resource helpers, and existing runner are outside this change's ownership and were not edited by this entry task.

Commands used:

```sh
python3 -m unittest discover -s tests -p test_hgtnet_batch_probe_cli.py -v
/private/tmp/gavit-hgtnet-probe-cpu-20261102/bin/python -m unittest discover -s tests -p test_hgtnet_batch_probe_cli.py -v
```

Root integration owns the complete related-suite run, immutable-source preservation audit, engine numerical tests, delivery build, and actual bundle verification. These13 orchestration tests alone are not a claim that the whole project passes.
