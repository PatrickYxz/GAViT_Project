# DFAGCN independent v1: remaining three prespecified seeds

This package extends the existing independent seed42 experiment with seeds **43, 44, 45**, chosen consecutively before any new test. It preserves the existing positive and negative history. It does not reproduce unavailable official code or claim numerical equivalence to the published DFAGCN result.

`dfagcn_campaign_20261110_v1.pyz` SHA256: `c09b024cf03eb0becd6358abc156c8802c824b38509d34db7fc322758e68c28b` (162345 bytes). Unpacking is automatic on `status`/`inspect`; it verifies all immutable files on every invocation, refuses foreign/changed workspaces, and never installs Python packages. Keep the pyz in persistent `/home/featurize/work`.

The package's `campaign_freeze.json` records the frozen seeds, 17 source-file parity checks against the completed `dfagcn_nwpu_formal_3080_20261030_fix1` package, exact base recipe, data/archive/pretrained identities, and seed42 reference. The recipe bytes are SHA256 `86faee50060a74119250a538cd96df90e65864bf319ee4befb01dddd6effca92`; the effective recipe differs only in `seed`. Each new seed independently selects its best epoch and C using the same validation rule, freshly refits on all 6300 train+validation images, reloads, and evaluates all 25200 test images once. The 100 calibration epochs, Adam, batch100, summed three-branch CE, learning-rate schedule, nine-C grid, solver/bias and FP32/TF32 flags are unchanged.

## Native binaries and software

No Linux executables were recovered previously. Both local historical binaries are Mach-O arm64 and are deliberately excluded. The package includes the pinned official LIBLINEAR v250 source archive, copyright and provenance; on the server the explicit command below compiles with existing compilers and records the new binaries' hashes. It performs no global installation and no GPU work.

```sh
python /home/featurize/work/dfagcn_campaign_20261110_v1.pyz build-linear
```

Alternatively pass the existing, verified Linux executables explicitly if they remain in persistent work. Historical seed42 paths were `/home/featurize/work/dfagcn_nwpu_formal_3080_20261030/liblinear/bin/train` (SHA256 `39de6baa6e0db556aa08884ef8ed3c9746a7fdabfe599fb11efefafd09f3ec4b`) and sibling `predict` (SHA256 `fd8e3a88870145c21d13d163c2bbe724e12ef0c047018426495e65758b4977b0`). New native builds may have different byte hashes; source/compiler/build records must be retained. The builder supports `--linux-bin-dir` only with actual x86_64 ELF binaries and matching completed Linux `build.json` provenance.

The original software contract is Python3.11.8, torch2.2.2 (CUDA12.1), torchvision0.17.2, numpy1.26.4, Pillow10.3.0 and cuDNN8902. `inspect` reads distribution metadata without importing ML packages; `run` checks actual imported runtime. Use the matching isolated server interpreter; this package does not silently relax the software recipe.

## Inspect and launch one seed at a time

Replace `GPU-REPLACE-WITH-OBSERVED-UUID` with the actual selected physical UUID. Inspect binds any observed idle GPU with sufficient available memory, rather than requiring the historical 3080. CUDA is selected by UUID before initialization, device name/count are checked, and NVML versus CUDA total-memory gap must be <=1GiB; exact fixed memory equality is not used.

```sh
python /home/featurize/work/dfagcn_campaign_20261110_v1.pyz inspect --seed 43 --gpu-uuid GPU-REPLACE-WITH-OBSERVED-UUID --scratch-root /tmp/dfagcn_campaign_20261110_seed43_01 --cache /tmp/dfagcn_campaign_20261110_frozen_vgg_v1 --output /home/featurize/work/dfagcn_campaign_20261110_v1/runs/nwpu_seed43_01
```

Only after `ready=true`, use the exact reported `run_command`/`nohup_command`, or change `inspect` to `run` in the same command. The generated command preserves the selected interpreter and all resolved data/weight/binary paths. Use a unique `--background-log` if using generated nohup commands for multiple seeds. To run seeds44 and45, change the seed, scratch-root and output accordingly; retain the same cache root only after the previous seed has completed successfully and sealed all four frozen split caches.

The fixed defaults are the persistent archive `/home/featurize/work/NWPU-RESISC45_a826b44.zip`, manifest `/home/featurize/work/nwpu_protocol_20260927_02.json`, pretrained `/home/featurize/work/vgg16-397923af.pth` and existing data `/home/featurize/data/NWPU-RESISC45`. Full SHA256 checks and ordered manifest image hashes remain enforced. Missing images are recovered safely into instance scratch from the verified archive.

Feature arrays, recovered images and all large LIBLINEAR `.svm` text stay on instance-local scratch. Persistent output keeps histories, checkpoints, selection, learned linear models, CLI logs/metadata, all test predictions/metrics, cache-index snapshots and SHA/byte receipts for ephemeral text. Storage preflight derives actual filesystem budgets from frozen feature bytes, conservative SVM text bounds, actual ZIP uncompressed image bytes, and per-destination margins; it adds budgets when destinations share a filesystem. It requires 1GiB persistent-artifact headroom, not the previous 40GiB work requirement.

Cache reuse is allowed only for a complete `cache.ready.json` seal with exact ordered image hashes, split/class order, pretrained file, preprocessing, pooling/activation and feature-extractor source identity. Full numeric-file SHA256 checks run before reuse; the runtime additionally validates arrays through the original FeatureCache. Graph/fusion/classifier features and learned checkpoints are never shared across seeds. Failed/partial cache roots are refused and preserved.

## Monitor and complete

```sh
python /home/featurize/work/dfagcn_campaign_20261110_v1.pyz status --output /home/featurize/work/dfagcn_campaign_20261110_v1/runs/nwpu_seed43_01
```

A PID is insufficient. Confirm the same process occupies the selected GPU and `run.log` reports `DFAGCN_FORMAL_DEVICE`, real VGG batches, then `DFAGCN_TRAIN_BATCH` on CUDA with a finite loss. Inspect starts no training. This campaign wrapper does not resume epochs and never overwrites an existing output/scratch directory. A failed job keeps its logs, artifacts and exit1; any later restart must use new identities and preserve the failure.

`status` declares complete only when `run.exit=0`, the immutable completion receipt hashes all required artifacts/current source, sealed selection/refit/test records pass checks, the full 25200 ordered test predictions exist, all nine C evaluations were recorded, and fresh6300 refit/reload evidence is present. Archive the output records, package/runtime/input/source identities and selected GPU before releasing the instance. New metrics remain unverified until this remote run and independent returned-prediction audit actually complete.

Local evidence: 13 campaign tests pass, source parity and package bytes pass, extracted pyz status passes. Existing stdlib resource/delivery tests also pass. The attempted broader local suite cannot execute ML tests because the local `.venv` lacks torch (89 tests attempted; 14 errors and 1 CLI failure). No local fixtures constitute target-GPU evidence. A hardware change must remain visible when aggregating seed42 with the new results; do not combine speeds from different GPU models as same-device efficiency.
