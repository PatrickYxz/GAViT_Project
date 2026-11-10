# Independent resource-fix review — 2026-10-30

Verdict: no Critical, Important, or Minor actionable findings within the requested resource-fix scope. Reviewed runner SHA256: `24e6386c16d8022df8cb2b708dc5c456e6872be553418914d3982c6c5a7ec563`.

## Findings

- Critical: none.
- Important: none.
- Minor: none.

## Verified behavior

1. Compared the current runner against the actual runner embedded in `output/dfagcn_nwpu_formal_3080_20261030.pyz` (SHA256 `66cbb46bc8bc0fb8a79e7e95794534e6efb430ee1630be2258c91df16c98861b`). Of the embedded files that exist at the same current repository path, only the formal runner differs; 24 source/recipe/helper files remain byte-identical. Full runner diff is saved as `runner_vs_original_pyz.diff`.
2. Runner lines 92–106, 137–140, 165–178, 185–192, 262–263: allowed unsuffixed torch/vision metadata is narrow; mismatched builds and other versions remain rejected. Actual imported CUDA 12.1/cuDNN 8902/FP32 flags are still required. A unique eligible GPU is selected, the generated command binds its current UUID explicitly, and runtime comparison uses that same requested UUID rather than the old instance constant.
3. Lines 297–345, 402–407: pool/train/val/test arrays and recovered images all use the independent scratch root. Calibration/refit/test products continue to use persistent output. All four cache index files are copied to persistent `cache_manifests`; the indices contain ordered sample identities, array shapes/dtypes and numeric SHA256 values. Persistent storage metadata records scratch location and volatility.
4. Lines 82–89, 148–164, 215: inventories follow the closest existing destination ancestor and use its actual `st_dev`. Scratch retains the full 40 GiB minimum; persistent output receives a separate bound. Same-device paths require the sum of the two budgets. New output/scratch, equality and both overlap directions are rejected; test-image recovery checks the destination volume. No cleanup or deletion is introduced.
5. Lines 471–476: fallback to the previous formal package’s compiled train/predict files changes defaults only. The generated explicit paths still pass normal binary and capacity inspection. Independent probes demonstrated work and scratch capacity rejection after this fallback, and rejection of a fallback train symlink. SHA256 is recorded for the actual chosen files, and `_unchanged` rechecks those inputs after execution.

## Resource arithmetic

`pipeline.py` lines 287–290, 344, 352 and final test inference retain nine training plus nine validation SVM files, one 6300-row refit training file, one 6300-row refit prediction file, and one 25200-row test prediction file:

`9 × (5040 + 1260) + 6300 + 6300 + 25200 = 94500 rows`.

The actual writer (`linear.py` lines 81–90) uses 5632 dimensions and `.17g`. Finite float32 values require at most 23 characters including sign/exponent; index width is at most four. Conservatively charging 29 bytes per dimension and three for a two-digit label/separator/newline gives 163331 bytes/row. Thus the configured 15434779500-byte SVM upper bound is valid for the fixed FP32 pipeline. A separate stdlib-only formatter probe sampled 99638 finite float32 values and observed the expected maximum 23 characters; this sampling complements the formatting bound, rather than proving it alone.

The graph parameter count was independently recomputed from the three disjoint branch dimensions as 2957447. Four FP32 checkpoint tensor payloads require 47319152 bytes (~45.13 MiB). Ten 5633 × 45 LIBLINEAR models charged at 25 text bytes per double weight require 63371250 bytes (~60.44 MiB), excluding tiny headers. The 1 GiB artifact margin leaves 963051422 bytes (~918.44 MiB) after those payload bounds for serialization overhead, persistent cache indices, source, histories, metadata, logs, predictions and metrics. This is reasonable for the fixed recipe; it is not an assertion of measured full-run size or a filesystem reservation.

Persistent gate: 16508521324 bytes. User-reported work free: 17029922816 bytes, leaving 521401492 bytes (~0.486 GiB) beyond that gate, in addition to the artifact margin already included. Run reruns inspection; subsequent concurrent storage use can still change actual free capacity.

## Validation

- Independently reran the eight new stdlib resource tests: 8 passed in 1.070 seconds.
- Additional stdlib-only probes: identical paths and both overlap directions rejected; previous-binary fallback still rejects work/scratch capacity shortfall and symlink binary.
- Probe inputs, results, reviewed source/package hashes, and arithmetic are saved in `independent_review.json`.
- Review generated files only under the designated review directory. No source, tests, diary, helper, algorithm, previous package, remote state, or experiment data was modified by this reviewer. No ML imports, CUDA calls, training, or complete dataset checks were performed.

## Declined to judge

- Current remote GPU availability, actual installed/runtime imports, NFS quota behavior and live capacity: not directly connected to the server.
- Full GPU pipeline execution, accuracy, convergence, full-run storage/throughput/cost, and ephemeral cache survival: no training or full-data run was authorized for this review.
- New fix1 archive identity or README delivery consistency: archive build and final packaging belong to the parent’s completion verification, and were not yet supplied for this audit.
- Full existing DFAGCN suite: parent is running that suite; this reviewer ran only the eight resource regressions and the bounded probes described above.
