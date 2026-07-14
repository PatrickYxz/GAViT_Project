# GAViT Agent Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the lowercase platform notes with durable agent instructions, a detailed Featurize runbook, and a verified BigEarthNet Swin diary entry.

**Architecture:** Keep responsibilities separate: `AGENTS.md` defines stable project and experiment rules, `docs/featurize_runbook.md` owns operational commands and platform failures, and `research_diary.md` owns chronological results. Migrate useful content from `agent.md`, then remove the duplicate lowercase file only after the new runbook is verified.

**Tech Stack:** Markdown, shell verification commands, Git

## Global Constraints

- Do not modify model or training code.
- Preserve the existing uncommitted change in `models/bigearth_dataset.py`.
- Do not invent the exact best epoch or any BigEarthNet test metric.
- Treat `/home/featurize/work` as persistent and `/home/featurize/data` as instance-local.
- Reduce unnecessary training by requiring one clear paper question and one main changed variable per full run.
- Use one fixed-seed run for candidate screening; run three seeds only for the final Swin baseline and selected GAViT model.
- Keep chronological experiment results in `research_diary.md`, not `AGENTS.md`.

---

### Task 1: Create the Root Agent Instructions

**Files:**
- Create: `AGENTS.md`
- Read: `agent.md`
- Read: `research_diary.md`
- Read: `docs/superpowers/specs/2026-07-14-agents-documentation-design.md`

**Interfaces:**
- Consumes: Approved project objective and experiment policy from the design spec.
- Produces: Automatically discovered repository guidance and a link to `docs/featurize_runbook.md`.

- [ ] **Step 1: Create `AGENTS.md` with focused sections**

Use these exact section boundaries:

```markdown
# GAViT 项目协作规则

## 项目目标
## 开始工作前
## 节省训练成本
## 实验设计规则
## 训练前检查
## 论文实验记录
## 指标要求
## 代码与 Featurize 分工
## 完成标准
```

The content must state:

- The core claim compares Swin with GAViT and tests region grouping plus graph reasoning.
- Agents must read `research_diary.md`, recent logs, and existing result tables before proposing a run.
- Every full run names its hypothesis, single changed variable, paper destination, and cheaper alternative.
- Existing checkpoints and offline analysis take priority over retraining.
- Candidate runs use one fixed seed; only final Swin and best GAViT use three seeds.
- Negative results are retained.
- Data access, forward pass, local pretrained loading, and a small one-epoch smoke run precede a paid full run.
- Every formal experiment records identity, data, model, optimization, runtime, outputs, and interpretation.
- Local development owns code changes; Featurize owns GPU runs; emergency remote patches must be mirrored locally.
- Detailed commands live in `docs/featurize_runbook.md`.

- [ ] **Step 2: Verify instruction scope and required phrases**

Run:

```bash
rg -n "项目目标|节省训练成本|三次|负面结果|research_diary|featurize_runbook" AGENTS.md
wc -l AGENTS.md
```

Expected: every required topic is found; the file remains concise enough to scan in one sitting and contains no long BigEarthNet extraction transcript.

- [ ] **Step 3: Check Markdown formatting**

Run:

```bash
git diff --check -- AGENTS.md
```

Expected: no output and exit status 0.

- [ ] **Step 4: Commit the root instructions only**

```bash
git add -- AGENTS.md
git commit -m "docs: add GAViT agent experiment rules"
```

Expected: the commit contains only `AGENTS.md`.

---

### Task 2: Create the Featurize Runbook

**Files:**
- Create: `docs/featurize_runbook.md`
- Read: `agent.md`
- Read: `AGENTS.md`

**Interfaces:**
- Consumes: Stable rules from `AGENTS.md` and platform evidence from the old `agent.md`.
- Produces: Copy-pasteable Featurize recovery, data, smoke-test, training, monitoring, and evaluation procedures.

- [ ] **Step 1: Create the runbook with current operational sections**

Use these exact sections:

```markdown
# Featurize Runbook for GAViT

## Storage Model
## New Instance Checklist
## BigEarthNet Data
## Persistent Metadata and Splits
## Python Dependencies
## Local Swin Pretrained Weights
## Preflight Smoke Tests
## Full Training
## Monitoring and Recovery
## Evaluation
## Known Failure Modes
```

Include the verified paths and facts:

```text
/home/featurize/work/GAViT_Project
/home/featurize/data/BigEarthNet-S2
/home/featurize/work/GAViT_Project/bigearth_files/metadata.parquet
/home/featurize/work/GAViT_Project/bigearth_files/splits
/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors
```

Include the verified split counts:

```text
train: 237871
val: 122342
test: 119825
missing_patch: 0
missing_bands: 0
```

Document that Hugging Face TLS failed through both `httpx` and `requests`, so the persistent local `model.safetensors` is the preferred source. Include the verified `timm.create_model(..., pretrained_cfg_overlay={"file": path})` loading pattern.

Include commands for:

- Checking `nvidia-smi`, persistent files, and instance-local data.
- Adding the platform `BigEarthNet-S2.zip` dataset and waiting for extraction.
- Rebuilding splits only when the persistent CSV files are missing.
- Checking `self.labels = np.asarray(labels, dtype=np.float32)`.
- A small data/model/training smoke test.
- Protecting smoke checkpoints from full-run checkpoint names.
- Launching with `nohup python -u`, checking `ps` and `nvidia-smi`, and reading logs.
- Running `test_bigearth.py` against the persistent splits and checkpoint.

- [ ] **Step 2: Remove stale operational claims while preserving useful history**

Ensure the runbook does not claim:

- Installing SOCKS support reliably fixes Hugging Face access.
- Metadata or generated splits should normally live in `/home/featurize/data`.
- Featurize should use SLURM or `sbatch`.
- Closing the browser stops a `nohup` job.

Run:

```bash
rg -n "metadata.parquet|bigearth_files/splits|model.safetensors|HF|nohup|BigEarthNet-S2" docs/featurize_runbook.md
rg -n "sbatch|SLURM" docs/featurize_runbook.md
```

Expected: the first command finds the current procedures; any `SLURM` match only explains that Featurize does not use it.

- [ ] **Step 3: Check Markdown formatting**

Run:

```bash
git diff --check -- docs/featurize_runbook.md
```

Expected: no output and exit status 0.

- [ ] **Step 4: Commit the runbook only**

```bash
git add -- docs/featurize_runbook.md
git commit -m "docs: add Featurize training runbook"
```

Expected: the commit contains only `docs/featurize_runbook.md`.

---

### Task 3: Record the Verified BigEarthNet Swin Result

**Files:**
- Modify: `research_diary.md` immediately after the introductory horizontal rule

**Interfaces:**
- Consumes: User-provided completed training log and checkpoint listing.
- Produces: A chronological, paper-facing record that future agents can distinguish from the older interrupted SLURM runs.

- [ ] **Step 1: Add a new top diary entry**

Use this heading:

```markdown
## 2026-07-14 — Featurize BigEarthNet Swin baseline 完整训练
```

Record only these verified facts:

```text
Platform: Featurize
Epochs: 30/30
Train samples: 237871
Validation samples: 122342
Parameters: 27535501
Best validation mAP: 78.8%
Final validation mAP: 75.8%
Final validation F1: 71.2%
Checkpoint: checkpoints/best_bigearth_swin.pth
Checkpoint size: approximately 106 MB
```

Also record:

- BigEarthNet images were instance-local while metadata, splits, pretrained weights, logs, and checkpoints were persistent.
- The late mAP decline suggests possible overfitting.
- The exact best epoch is not yet extracted.
- Test mAP, macro-F1, micro-F1, and per-class AP remain pending.
- The next action is to reattach BigEarthNet on the new instance and evaluate the saved checkpoint before starting GAViT.

- [ ] **Step 2: Verify that no pending metric is presented as complete**

Run:

```bash
sed -n '1,90p' research_diary.md
rg -n "78.8%|75.8%|71.2%|测试.*待|best epoch.*待" research_diary.md
```

Expected: the new entry is first, verified values are present, and test metrics plus exact best epoch are explicitly pending.

- [ ] **Step 3: Check Markdown formatting**

Run:

```bash
git diff --check -- research_diary.md
```

Expected: no output and exit status 0.

- [ ] **Step 4: Commit the diary entry only**

```bash
git add -- research_diary.md
git commit -m "docs: record BigEarthNet Swin baseline"
```

Expected: the commit contains only `research_diary.md`.

---

### Task 4: Remove the Duplicate Lowercase Notes and Verify the Documentation Set

**Files:**
- Delete: `agent.md`
- Verify: `AGENTS.md`
- Verify: `docs/featurize_runbook.md`
- Verify: `research_diary.md`

**Interfaces:**
- Consumes: Completed replacement instructions and runbook.
- Produces: One automatically discovered instruction file with no duplicate lowercase notes.

- [ ] **Step 1: Confirm all old topics were migrated before deletion**

Run:

```bash
rg -n "Featurize directories|BigEarthNet data|Dependency gotchas|Smoke tests|Running full training" agent.md
rg -n "Storage Model|BigEarthNet Data|Python Dependencies|Preflight Smoke Tests|Full Training" docs/featurize_runbook.md
```

Expected: every old operational category has a corresponding current runbook section.

- [ ] **Step 2: Delete the untracked lowercase file**

Use `apply_patch` to delete `agent.md`. Do not use `rm`, and do not touch `models/bigearth_dataset.py`.

- [ ] **Step 3: Run cross-document verification**

Run:

```bash
test -f AGENTS.md
test -f docs/featurize_runbook.md
test ! -e agent.md
rg -n "docs/featurize_runbook.md|research_diary.md" AGENTS.md
git diff --check
git status --short
```

Expected:

- `AGENTS.md` and the runbook exist.
- `agent.md` does not exist.
- `AGENTS.md` links to the runbook and diary.
- `git diff --check` reports no whitespace errors.
- The pre-existing `models/bigearth_dataset.py` modification remains untouched.

- [ ] **Step 4: Commit the deletion only if Git tracks it**

The current `agent.md` is untracked, so deleting it normally produces no commit. If repository state differs at execution time, stage only `agent.md` and commit:

```bash
git add -- agent.md
git commit -m "docs: remove duplicate agent notes"
```

Expected: no unrelated files are staged.

