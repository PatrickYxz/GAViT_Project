# BigEarth soft45 launch implementation plan

**Goal:** Deliver the authorized single-model seed45 30-epoch training and best-checkpoint test as one pasteable background command, preserving the completed controls.

**Design authority:** `docs/run_bigearth_soft45_20260930.md` and the returned `BIGEARTH_SOFT45_PREFLIGHT_PASS`. This is bounded orchestration of frozen code, not a model change. Implement inline; user already approved the experiment and returned its required preflight, so no new approval loop.

**Architecture:** A new external Python runner imports the hash-pinned old runner's identity, argument-building and process helpers. It changes grouping to soft_prototype, invokes frozen train/test through isolated workers, and validates complete artifacts before advancing. The preflight runs in its own child so a CUDA context cannot contaminate the parent GPU-idle check. One compressed command writes only a fresh output directory, scripts and logs there, then detaches. No repository changes on the server.

**Constraints:** ed480e6; seed45; frozen full CSV/pretrain hashes; batch32/30 epochs/unchanged FP32 recipe; existing seed45 controls retained; no automatic retry, overwrite, smoke, refit, seed expansion or architecture tuning. Missing data or drift stops before training. Same actual GPU identity throughout this queue; historical UUID difference is reported.

**Files:** `tools/run_bigearth_soft45.py`, `tests/test_bigearth_soft45.py`, delivery/evidence under `results/evidence/bigearth_soft45_prepare_20260930`, execution guide and research diary/schedule/matrix.

## Tasks

- [x] Write and run failing tests for only-soft recipe/best-checkpoint routing, training history and metadata completeness, validated stage transition, nonzero exit and exit0-with-incomplete-result stopping, duplicate output rejection, test-array and probe integrity.
- [x] Implement only the missing external runner; reuse frozen helpers without editing them. Record resource scope, exact epoch metrics, probe artifacts, metadata/checkpoints, test metrics and predictions; all failure paths preserve output and write nonzero exits where the process can handle the failure.
- [x] Run targeted tests including CPU fixtures for actual runpy instrumentation and numerical artifact validation. These cannot substitute for real CUDA training.
- [x] Review final launcher and generate a hash-checked one-line payload. Validate Python/Bash/zsh syntax, payload roundtrip, sandboxed detached launch/failure/duplicate behavior and no model-source changes.
- [x] Archive returned preflight, update pending experiment records, deliver start/progress/completion commands. Server execution remains pending user action; PID alone is not success.

## Review focus

1. Parent CUDA initialization blocking its own workers: preflight must exit in a subprocess.
2. Exit0 without 30 epochs or wrong best weights: validate reports and artifacts before test.
3. RNG/probe changes contaminating optimization: hook only the checkpoint recorder, preserve return values, use released entry and fixed probe.
4. A second launch overwriting an interrupted run: exclusive directory and claim marker, never automatic fresh retry.
5. Lost outputs on worker/metadata failure: preserve raw logs and last checkpoint; no completion marker until all checks pass.

Validation: 12 targeted checks passed; command payload/CPU fixtures/detached fail-closed and duplicate-preservation checks passed. Reviewer found no additional actionable issues. GPU formal run remains pending user execution. No model-source changes or Git commit/push.
