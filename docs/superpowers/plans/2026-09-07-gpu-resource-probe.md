# GPU Resource Probe Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this approved first-week task inline. Steps use checkbox syntax for tracking.

**Goal:** Deliver portable 3080/3060 smoke profiling tools and frozen seed44/45 command plans.

**Architecture:** Keep the verified training checkout unchanged. A standard-library CLI validates inputs, creates an isolated smoke split and invokes the existing trainer through a resource-profiling wrapper. Formal commands are generated for review; they never auto-run after a smoke.

**Tech Stack:** Python standard library locally; existing PyTorch training environment, Linux /proc and nvidia-smi remotely.

**Spec:** `docs/thesis_schedule.md`, section GPU 资源实验设计.

## Global Constraints

- Preserve pre-existing dirty diary/results files; no commit, push, paid instance creation or logged-in browser access in this local preparation step.
- Training source must remain the clean `bb9f1cff463c996bc5f72695af538898c91490eb` checkout for runtime probes; install the tools outside that checkout.
- Both models, batch32, fixed input224, unchanged formal30 epochs, same-seed pairing.
- Smoke results are resource measurements only; missing resource fields and unverified CUDA paths are explicit, never replaced by zero.
- Refuse output collisions and preserve failed runs.

## Task 1: Portable resource preparation and command planning

**Files:** Create `tools/resource_probe.py`, `tests/test_resource_probe.py`.

**Interfaces:** `prepare_smoke_split(source, output, train_count=4096, val_count=1024)` returns a manifest; `training_args(repo, output, model, stage, seed)` returns exact trainer argv; `scan_dataset(source, expected_counts)` returns split identities and missing-band counts.

- [x] Write failing tests for deterministic train/val-only sampling, binary labels, empty/duplicate paths, output preservation, missing bands, formal seed validation and CLI plan having no side effects.
- [x] Run `python3 -m unittest tests.test_resource_probe -v` and observe missing implementation failures.
- [x] Implement CSV validation/sampling, full preflight, explicit smoke execution and formal command planning. Model arguments reuse the values in the spec; shell display uses `shlex.join`.
- [x] Run the tests and inspect actual CLI JSON output with `python3 tools/resource_probe.py plan --gpu 3080 --stage formal --seed 44 --output-dir /tmp/gavit-example-seed44`.

## Task 2: Measured wrapper around the real trainer

**Files:** Create `tools/profile_training.py`, `tests/test_training_profile.py`.

**Interfaces:** `timed_batches(iterable, stats, phase, clock, synchronize, warmup=10)` preserves batches and records measured work; `read_process_memory(root_pid, proc_root)` returns sampled process-family RSS/PSS and available host memory. Structured CLI `--repo --output-dir --gpu --hourly-rate --stage --model --seed [--smoke-dir]` reconstructs frozen arguments and invokes the existing training entrypoint; arbitrary trainer arguments are not accepted.

- [x] Write failing tests for warmup exclusion, timing including consumer work, unchanged batch order, exception propagation, process descendants, missing PSS, and non-overwriting reports.
- [x] Run `python3 -m unittest tests.test_training_profile -v` to observe failures.
- [x] Implement DataLoader iteration timing, best-effort periodic memory samples, CUDA peak reporting and successful artifact validation. Restore instrumentation in `finally`; failures retain diagnostics and nonzero exit codes.
- [x] Run both new test modules and existing `tests.test_experiment_identity`, `tests.test_sparse_hybrid_static_contract`.

## Task 3: Runbook and delivery

**Files:** Create `docs/gpu_resource_probe.md`; update `docs/thesis_schedule.md` and prepend `research_diary.md` entries without altering older results.

- [x] Document exact copy/extract, plan and smoke commands, external tools placement, output meanings, formal configuration and pending server verification.
- [x] Check CLI help, syntax, output-collision behavior, diff whitespace and untouched training source.
- [x] Package only the portable scripts and runbook in a tar.gz under a fresh temporary directory, inspect its file list, and link it to the user.
- [x] Mark local W1-03 complete only after checks pass; leave GPU measurements and formal seeds pending.

Verification on 2026-09-07: 37 local tests passed (15 new, 22 existing), py_compile and both CLI help commands passed. Extracted package files match the sources byte-for-byte. An independent reviewer could not run because of the service usage limit; the main agent reviewed the trainer/checkpoint integration. No independent-review or CUDA-pass claim is made.

Local scope does not require installing PyTorch on the Mac. Tensor-level/full GPU tests are mandatory on the destination instance before smoke; lack of local CUDA is not proof of a working GPU path.
