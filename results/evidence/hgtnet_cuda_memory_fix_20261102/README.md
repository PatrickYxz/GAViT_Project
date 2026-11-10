# CUDA visible-memory gate fix — 2026-11-02

Root cause is demonstrated by the actual supplied outputs and the old gate: NVIDIA inventory reports24576MiB physical total, while CUDA `_CudaDeviceProperties` displays24123MB. CUDA is available, count1 and nameRTX3090. The old equality comparison therefore stops before data/model training. The display does not supply exact bytes, and the user's command printed the torch.version module rather than a full version string. We do not infer a proven reservation/virtualization cause.

`user_reported_failure.json` and `user_cuda_diagnostic.txt` preserve a clearly reconstructed record from the supplied fields; they are not claimed to be downloaded server artifacts. The previous server output directory was already created before the failure, so the fix uses a separate package/output/scratch namespace.

## Change

Only `tools/run_hgtnet_nwpu_smoke.py` production source changed relative to the original3090 bundle:

- Separate CUDA availability/count/name/type/capacity diagnostics instead of a combined opaque error.
- Keep physical NVML target24576MiB and UUID/idle/software/precision checks strict.
- Accept positive integer CUDA bytes within `[NVML nominal bytes-1GiB, NVML nominal bytes]`; reject one byte outside either boundary. The1GiB allowance is an explicit project compatibility policy, not a measured model requirement or a claim about the cause of the difference.
- Print the raw device probe and retain separate CUDA/NVML observations and gap values in runtime provenance. The actual runtime still checks the full Torch/vision/CUDA/cuDNN/Python versions.

Model, data, engine, config, candidate weights and15 locally present other payloads remain byte-identical; builder and original3090 package remain unchanged. No protocol, batch2/seed42, augmentation/precision or optimizer change.

## Final delivery

`output/hgtnet_nwpu_smoke_3090_20261102_fix1.pyz`:53324 bytes, SHA256 `90c60dabcd185ffd0a1b575f8544e7363a469c4e3616528955cb3645c2ae7c42`, bundle ID `96d3451f4119c8a42f8507e819f534cca42e2b7235b21006984144d488158019`.

Instructions: `output/hgtnet_nwpu_smoke_3090_fix1_README_20261102.md`. Existing weights/manifest/ZIP are reused. Existing failed output is preserved; no server hot patch or deletion.

## Evidence and limits

- `integration/red.log`:6 new tests fail before implementation, including reproducing rejection of the representative reported capacity at the actual runtime-adapter boundary.
- `integration/green.log`:6/6 new unit tests pass, including exact/representative453MiB gap/1GiB boundary, positive integer capacity, wrong CUDA identity/count and one-byte violations. Representative raw bytes use24123×MiB as a synthetic fixture, not claimed as exact user-reported bytes.
- `integration/targeted_suite.json/.log`: **45/45** current hardware/CLI/default3080/delivery tests pass, no errors/failures/skips; runner wall0.3459s. The unavailable local full numeric suite is not rerun or claimed passing; the previous missing Torch/NumPy limitation remains documented.
- `review/`: independent before snapshot/probe reproduces the old rejection; final16 runtime boundary cases +9 information/identity checks pass, preserving wrong UUID/software/physical memory rejection. No unresolved review issue.
- `integration/bundle_inspect_guard.json`, `actual_bundle_status.json`: actual final fix1 zipapp inspected without ML imports, profile3090, independent fix1 paths, no runs, not_started.
- `integration/validation.json`: **112/112** file/identity/preservation checks pass: fix1 bytes and sources, old3090 package, builder,15 other local payloads,52 historical artifacts and full candidate weight bytes.
- `integration/commands_and_exits.json`: actual local executions and exit codes.

This is a gate correction and local stdlib boundary verification. Actual CUDA forward/backward, real360/90, memory peak, time and checkpoint outcome still require server execution of fix1. No new remote/GPU/training, library installation, Git commit/push or user artifact overwrite occurred here.
