# HGTNet Local Implementation Plan

> **For agentic workers:** Execute authorized work continuously with test-driven-development and scoped parallel agents. No Git commits/pushes and no additional approval loop for the accepted independent local implementation.

**Goal:** Implement and numerically verify HGTNet-independent-v0 modules, complete model, candidate pretraining mapping and checkpoint roundtrip locally.

**Architecture:** New standalone external_baselines/hgtnet package, torchvision Swin primitives, explicit DA/graph operators, strict config/identity. No legacy model or training entry changes.

**Tech Stack:** Python3.12 local engineering runtime, Torch2.2.2/torchvision0.17.2 CPU, unittest; stdlib config/provenance. Target GPU execution separate.

**Spec:** docs/external_baselines_20261027/hgtnet_implementation_design_20261031.md

## Global Constraints

- Preserve dirty user files, DFAGCN/legacy sources/results and all existing checkpoints. Work in current checkout because related research sources are uncommitted; new package/files only.
- Explicit nonofficial independent candidate; unknowns must remain labeled. No GPU/formal training/remote mutation/model benchmark result/Git operation.
- Full candidate input224, channels128/256/512, attention2/2/18, stage spatial56→28→14→7; graph7→5/k20/group6,2; gap_sum.
- Pure module tests may use small channels. Full model smoke must execute the full actual configuration, not just meta/tiny substitutes.
- Local tests: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE /private/tmp/gavit-dfagcn-cpu-20261027/bin/python.

## Review Focus

- Wrong STB pair counting or layout permutations silently change architecture.
- Invalid k20/N16 must fail rather than clip; no cross-batch neighbors or reversed relative messages.
- Pretraining shape-only/partial loads must not masquerade as verified ImageNet or original HGTNet.
- Source/recipe/buffers/weights must bind checkpoint; fresh load must reproduce eval outputs.
- Full-model finite gradients must reach backbone, DA, graph and head; summaries must not turn synthetic CPU work into dataset/GPU scores.

### Task 1: DA and Swin stages

Own external_baselines/hgtnet/attention.py and tests/test_hgtnet_attention.py; tests/evidence path attention/.

- [x] Write behavior tests using hand gates and real torchvision blocks, verify expected missing-API RED.
- [x] Implement PatchEmbedding, DualAttention and TransformerStage with named parameters in spec.
- [x] Verify GREEN shapes/non-square/broadcast, pair shifts, gradients and no silent channel/downsample mismatch.

### Task 2: graph operators

Own external_baselines/hgtnet/graph.py and tests/test_hgtnet_graph.py; graph/ evidence.

- [x] Hand fixtures max vsmean, k/tie/self/batch, malformed inputs, valid finite gradients; RED.
- [x] Implement kNN/MRConv/Grapher/FFN/DWProjection; GREEN. Actual indices exposed only on request.

### Task 3: pretraining boundary

Own external_baselines/hgtnet/pretrained.py and tests/test_hgtnet_pretrained.py; pretrained/ evidence. Consumes exact target names from spec; independent of graph forward.

- [x] Test candidate mappings, missing/unexpected/shape/dtype/nonfinite and no partial mutation; RED.
- [x] Implement audit/apply/file loader; GREEN with actual copied CPU tensors and full official source schema on meta if needed.
- [x] Try bounded official candidate acquisition; save provenance/full SHA only on successful actual bytes. Otherwise explicitly pending, never fabricate success.

### Task 4: config/model/checkpoint/local smoke

Own config.py/model.py/checkpoint.py/__init__.py/__main__.py/smoke.py, configs/hgtnet_independent_v0.json and tests/test_hgtnet_model.py.

- [x] Config/model/checkpoint behavior tests RED before code.
- [x] Integrate Tasks1/2 under exact namespace; JSON-safe complete shape trace and parameter accounting.
- [x] Full B1/B2 CPU224 forward/backward, finite group gradients and one optimizer update; strict config/source bound save plus fresh-process reload (no overwrite).
- [x] Actual candidate load if bytes available, report matched/missing/intentionally-new coverage accurately. No online side effects in ordinary model constructors/loader.

### Task 5: review/verification/docs

- [x] Independent review current artifacts against spec; fix important findings via RED/GREEN.
- [x] Run full HGTNet/DFAGCN relevant tests once after integration; preserve known preexisting full-repo dependency failures without claiming all-repo green.
- [x] Check protected inputs/metrics/legacy sources; git diff --check, update diary/README with actual evidence and remaining GPU/formal gates.

## Progress / rulings

Interfaces pre-flight: stage names and layout consumed by model and source mapping fixed above; graph uses BNC and transitions BCHW; root owns config/init to avoid overlapping agent edits.

Ruling: user authorized independent implementation after prior audit/plan; settle explicit engineering candidate locally rather than repeat an approval loop. Cost: no claim of author exact architecture; formal comparison still needs pretraining/source/resource gates.

Ruling: replace earlier third-stage-no-downsample14→7 candidate with text-consistent stage3→7 and graph7→5. Exact Conv2/DWstride1pad0 remains an engineering choice, documented before training and not selected from outcomes.

Ruling: do not create a new Git worktree/branch that omits uncommitted research context; write only new named package/test files and preserve all user changes. No staging/committing/pushing.

Final local outcome: 87 HGTNet +148 DFAGCN tests passed; actual full CPU B1/B2, candidate load, all trainable finite gradients and Adam update, fresh-process reload max errors0. Independent review had no unresolved findings; 52/52 protected hashes matched. Full evidence: results/evidence/hgtnet_local_implementation_20261031/README.md. No GPU/dataset benchmark or Git publication.
