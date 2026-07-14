# GAViT Agent Documentation Design

Date: 2026-07-14

## Purpose

Create durable project guidance that helps future agents continue the GAViT paper
work without repeating failed setup work or spending GPU budget on experiments that
do not answer a clear research question.

The project objective remains unchanged: compare the Swin baseline with GAViT and
determine whether region grouping and graph reasoning provide a stable,
interpretable improvement for remote-sensing scene classification.

## Document Structure

### `AGENTS.md`

`AGENTS.md` is the concise, automatically discovered instruction file. It will be
written primarily in Chinese and contain:

- The paper objective and the role of Swin, region grouping, and graph reasoning.
- Required files to inspect before proposing a new experiment.
- Cost-aware experiment rules that reduce unnecessary full training runs.
- Preflight checks required before a paid run.
- The staged repeat policy: one fixed-seed screening run for candidates, followed
  by three runs only for the final Swin baseline and best GAViT configuration.
- Required experiment metadata and paper-facing evidence.
- Local-development and Featurize synchronization rules.
- A pointer to the detailed Featurize runbook.

It will not contain long platform commands, chronological results, or volatile
instance-specific state.

### `docs/featurize_runbook.md`

The runbook owns operational details:

- Persistent `/home/featurize/work` versus ephemeral `/home/featurize/data`.
- New-instance recovery and dependency checks.
- BigEarthNet-S2 layout, metadata source, split counts, and persistent split paths.
- Local Swin pretrained weights and the verified workaround for unreliable
  Hugging Face TLS access.
- Data, model, and one-epoch smoke tests.
- Background training, monitoring, checkpoint handling, evaluation, and recovery.
- Known failures: TLS errors, Git conflicts, incomplete browser uploads, missing
  data after changing instances, and smoke checkpoints overwriting real results.

Commands in this file must use the current persistent locations under
`/home/featurize/work/GAViT_Project/bigearth_files` where appropriate.

### `research_diary.md`

The diary remains the chronological source of experimental facts. A new entry will
record the completed BigEarthNet Swin baseline without claiming unverified values:

- 30 epochs completed.
- Best validation mAP: 78.8%.
- Final validation mAP: 75.8%.
- Final validation F1: 71.2%.
- Best checkpoint exists and is approximately 106 MB.
- Test metrics and the exact best epoch remain pending.
- The late validation decline is evidence of possible overfitting.

## Cost-Aware Experiment Policy

The documentation will not focus on whether an agent has authority to start paid
training. Its purpose is to reduce the number of unnecessary runs.

Before a full run, the experiment must identify:

1. The paper question or hypothesis it answers.
2. The single main variable changed from an existing experiment.
3. The table, figure, ablation, or claim that will use the result.
4. Whether existing logs, checkpoints, offline analysis, or a smoke test can answer
   the question without full training.

Candidates are screened with one fixed seed. Configurations that do not improve or
clarify the paper claim are not repeated. Only the final Swin baseline and selected
GAViT configuration receive three runs for mean and standard deviation.

Negative results must be recorded so future agents do not repeat them.

## Paper Evidence Requirements

Every formal experiment record must include:

- Identity: date, experiment name, Git commit, command, and random seed.
- Data: dataset version, split sizes, preprocessing, and augmentation.
- Model: architecture, parameter count, pretrained source, and all GAViT-specific
  grouping, graph, and integration settings.
- Optimization: epochs, batch size, optimizer, learning rate, weight decay,
  scheduler, loss, and classification threshold.
- Runtime: GPU model, peak memory, throughput, epoch time, wall time, and estimated
  or actual cost.
- Outputs: best and final epochs, train/validation/test metrics, log path,
  checkpoint path, and curve data.
- Interpretation: baseline delta, whether the hypothesis was supported, anomalies,
  and intended paper placement.

NWPU records use accuracy, macro-F1, per-class accuracy, and confusion matrices.
BigEarthNet records use mAP as the primary metric, with micro-F1, macro-F1,
per-class AP, and the classification threshold.

## Scope of This Change

Implementation will:

1. Replace the untracked lowercase `agent.md` with a concise root `AGENTS.md`.
2. Move and update operational content in `docs/featurize_runbook.md`.
3. Append the verified BigEarthNet Swin result to `research_diary.md`.

It will not modify model code, launch training, fabricate missing metrics, or touch
unrelated working-tree changes.

