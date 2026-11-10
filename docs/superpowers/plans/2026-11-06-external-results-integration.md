# External Results Integration Implementation Plan

> **For agentic workers:** Use subagent-driven execution for independent analysis and thesis files; root handles synthesis, supervisor material and integration review.

**Goal:** Turn the completed three-cell external-baseline measurements into verified analysis, readable comparison material and a compiled thesis supplement.

**Architecture:** Derive AID diagnostics offline from immutable returned artifacts. Keep single-seed complete-method comparisons separate from core multi-seed/internal ablations. Use existing report and thesis structure.

**Tech Stack:** Python standard library and an existing Matplotlib environment; Markdown/plaintext/CSV/JSON; existing LaTeX compiler and PDF renderer.

**Spec:** `docs/superpowers/specs/2026-11-06-external-results-integration-design.md`.

## Global constraints

- No GPU model execution, new paid training, server source edits, changed splits or test replay.
- HGTNet AID: 5000 test rows, 30 classes, Accuracy96.92%, macro-F1 96.77286814931982%, selected96/final-calibration100/refit96, 55048 accepted steps.
- Attempt2 wall/peaks are not whole-experiment resources. Checkpoint binary hashes are server-verified; local receipt checks are not a local model reload.
- Preserve all seeds and known negative results; missing46/47 raw artifacts/F1 remain missing.
- No historical worktree edits, old core table rewrites or Git operations.

## Review focus

- Reject shuffled or changed test paths/labels rather than silently aligning inconsistent artifacts.
- Recompute all30 class metrics and confusion cells, including zero-count classes in fixtures.
- Keep best96 and final100 visible; validation selection never uses test.
- Distinguish raw paired predictions from aggregate count differences of1(AID) or4(NWPU).
- Figure labels must be legible; thesis compilation alone is insufficient visual QA.

## Task 1: Offline AID diagnostic analysis

- [x] Add a small analysis entry and meaningful tests for metric reconstruction, identity/order rejection and earliest validation winner.
- [x] Run relevant tests and the entry on the complete returned AID artifacts; match the existing two independent audits.
- [x] Export all predictions/errors/classes/confusions/history and an analysis report; save history/class/confusion PNG/SVG figures with deterministic ordering.

## Task 2: Comparison and supervisor material

- [x] Verify Swin/GAViT AID seed42 summary sources and source-level protocol/resource distinctions.
- [x] Expand the independent AID table to three rows; create a dated cross-dataset comparison with exact differences and explicit remaining requirements.
- [x] Update current Chinese/English report statuses and create a dated copyable update; preserve previous bytes and unchanged historical claims.

## Task 3: Thesis supplement

- [x] Preserve current thesis bytes; correct stale external-comparison status in introduction/setup.
- [x] Add a supplementary external-comparison chapter and include it in main.tex; retain partial-draft status and template formatting.
- [x] Include verified numerical tables and supported discussion; do not invent significance, causal superiority, missing paired counts or multi-seed external statistics.

## Task 4: Integrate and verify

- [x] Reconcile all report/table/LaTeX numbers with source JSON and verify raw artifact hashes unchanged.
- [x] Compile thesis, inspect changed PDF pages and analysis figures, and resolve layout/reference defects.
- [x] Refresh diary, evidence ledger and deliverable links; record remaining unified-protocol/multi-seed work as open.

## 2026-11-08 verification closeout

All four integration tasks completed.47-page partial thesis exported and checked:19 final pages directly reviewed and28 old-body pages pixel-identical to reviewed renders. Native and isolated cached-Tectonic compilation pass. Current result/copyable drafts and portableZIP are delivered; unified-protocol/multiseed external research and the rest of the thesis remain open. Evidence: `results/evidence/external_results_integration_20261106/final_validation_20261108.json`. Returned instance was not contacted.
