# NWPU soft-assignment GPU smoke

Release branch: `codex/soft-assignment-20260927`, based on `deb2b45`.
The implementation adds local prototype soft assignment and uses the same assignments for region pooling and graph feedback. It adds 99,840 parameters, retains 16 regions and the 48 spatial + 32 feature edges, and preserves the existing fixed grouping default.

The release candidate passed all 133 tests present in this branch on macOS/Python3.12/CPU (44 warnings, exit0, 47.52 seconds). See [test log](../results/evidence/soft_assignment_release_20260927/tests.log) and [source hashes](../results/evidence/soft_assignment_release_20260927/source_hashes.json). These are engineering checks, not GPU or accuracy results. The unrelated unpublished analysis tools and their tests are outside this branch.

## Frozen server inputs

User-reported on 2026-09-27:

- GPU: RTX2080 Ti, 22528 MiB total and 22009 MiB free at the time of querying.
- Existing checkout: `/home/featurize/work/GAViT_AID_20260920`, clean `deb2b45`.
- Images: `/home/featurize/data/NWPU-RESISC45`, 45 classes and 31,500 files.
- Manifest: `/home/featurize/work/nwpu_protocol_20260927_02.json`; SHA256 `ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`.
- Pretrained file: `/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors`; SHA256 `f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`.

The manifest uses split seeds42/4242 and grouped duplicates, with train5040/val1260/test25200. Eight groups of exact same-class duplicates remain entirely in one split; all files are retained, with 31,492 unique pixel images. All three compared models must share this manifest. Do not run prepare again or mix in the older NWPU results/splits.

## Independent checkout

Fetch the release branch from the existing repository, then create a detached worktree at the published commit reported by the PR. The target directory is `/home/featurize/work/GAViT_SOFT_20260927`; preserve the original checkout and historical runs. The published commit must be recorded before training. Stop if the directory already exists or Git reports an error; do not overwrite it.

## One-epoch smoke

Purpose: validate real pretrained loading, CUDA training, graph construction, checkpoint reconstruction and diagnostics, then measure resources. Use 8train/2val per class (360/90 total), seed42, batch32, workers2, FP32, no AMP, TF32 off. This phase runs one epoch, with the unchanged CE/AdamW3e-4/weight_decay1e-4/Cosine30 recipe. Validation images supply fixed assignment probes. No test inference or formal accuracy claim is authorized by this smoke command.

Run only after the independent checkout and hashes above match:

```bash
nohup python -u /home/featurize/work/GAViT_SOFT_20260927/run_scene.py train --model gavit --grouping soft_prototype --phase smoke --seed 42 --train-per-class 8 --val-per-class 2 --manifest /home/featurize/work/nwpu_protocol_20260927_02.json --data-root /home/featurize/data/NWPU-RESISC45 --output /home/featurize/work/nwpu_soft_smoke42_20260927_01 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --device cuda --batch-size 32 --workers 2 >> /home/featurize/work/nwpu_soft_smoke42_20260927_01.launch.log 2>&1 < /dev/null &
tail -f /home/featurize/work/nwpu_soft_smoke42_20260927_01.launch.log
```

The process first verifies all images against the manifest. The earlier prepare scan took47.6s; current validation or training duration is not yet measured. A PID alone does not demonstrate startup: require the CUDA model line, `TRAIN FIRST BATCH OK`, and a GPU process/memory allocation. In another terminal:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

After completion, Ctrl+C exits tail without stopping the nohup process. Read the final evidence:

```bash
cat /home/featurize/work/nwpu_soft_smoke42_20260927_01/run.exit /home/featurize/work/nwpu_soft_smoke42_20260927_01/run.json
tail -n 50 /home/featurize/work/nwpu_soft_smoke42_20260927_01.launch.log
```

Require `SCENE SMOKE COMPLETE`, `exit_code=0`, `status=complete`, counts360/90, actual48+32 graph edges, checkpoint hash and reload logit difference≤1e-5. Inspect the fixed probe's epoch0/1 diagnostics and epoch0/best/final tensors. A separate real-CUDA assignment-only gradient check remains necessary; falling classification loss alone does not verify that path.

## Limits and next steps

CUDA, Linux/Python3.11, multiworker real-data loading, and model gains remain unverified until returned evidence. The next research comparison is Swin/fixed4x4/soft grouping on one frozen NWPU protocol; formal30epoch calibration/refit/test and multiple seeds are separate runs. BigEarth96/64 real-image smoke is also still pending. Do not start full BigEarth training from this document.

Old scene checkpoints retain their recorded source-code identity and must be reconstructed with that source version. BigEarth's inherited resume mechanism lacks full RNG restoration and full split/source binding; this change preserves probe history and aliases but does not claim bitwise uninterrupted equivalence. Spatial edges represent anchor priors, not verified semantic adjacency.
