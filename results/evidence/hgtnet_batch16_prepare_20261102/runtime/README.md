# HGTNet local CPU test runtime, 2026-11-02

Separate temporary environment: `/private/tmp/gavit-hgtnet-probe-cpu-20261102`.
Task cache: `/private/tmp/gavit-hgtnet-probe-cache-20261102`.
Existing project and old temporary environments were not modified.

The interpreter is CPython 3.12.13 on macOS arm64. The four intended ML/image dependencies are pinned exactly: torch 2.2.2, torchvision 0.17.2, numpy 1.26.4, Pillow 10.3.0. This local Python version differs from the target server's Python 3.11.8, so local checks cannot establish server runtime equivalence.

Creation command:

```sh
UV_CACHE_DIR=/private/tmp/gavit-hgtnet-probe-cache-20261102 /opt/homebrew/bin/uv venv --python /private/tmp/gavit-dfagcn-cpu-20261027/bin/python /private/tmp/gavit-hgtnet-probe-cpu-20261102
```

The first sandboxed installation failed because `pypi.org` could not be resolved. `install.log` preserves the error. One approved sandbox escalation used the official PyPI index with normal TLS verification:

```sh
UV_CACHE_DIR=/private/tmp/gavit-hgtnet-probe-cache-20261102 UV_HTTP_TIMEOUT=40 UV_HTTP_RETRIES=1 /opt/homebrew/bin/uv pip install --python /private/tmp/gavit-hgtnet-probe-cpu-20261102/bin/python --index-url https://pypi.org/simple torch==2.2.2 torchvision==0.17.2 numpy==1.26.4 pillow==10.3.0
```

`install_escalated.log` records resolution/install success: 12 packages prepared in 43.30 seconds and installed in 150 ms. The macOS arm64 torch wheel was 56.9 MiB; no CUDA/GPU wheel was installed. `requirements.freeze.txt` records all installed package versions.

Verification command:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE /private/tmp/gavit-hgtnet-probe-cpu-20261102/bin/python results/evidence/hgtnet_batch16_prepare_20261102/runtime/verify_runtime.py
```

The check imports all four pinned dependencies, asserts exact versions, and executes a real CPU forward, cross-entropy loss, backward, and optimizer update with finite gradients. `verify.log` and `runtime_verification.json` contain the actual result. This proves only the separate local CPU runtime; it is not a full HGTNet test, real NWPU evaluation, target GPU capacity result, or formal training outcome.
