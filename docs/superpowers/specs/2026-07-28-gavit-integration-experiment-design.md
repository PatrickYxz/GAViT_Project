# GAViT Integration and Controlled Experiment Design

**Date:** 2026-07-28
**Status:** Approved design, implementation not started

## 1. Purpose

The current GAViT v2 maps each GAT-refined region feature back to every
token assigned to that region, adds the feedback with unit strength, and
immediately mean-pools the updated tokens. For fixed assignments, this reduces
to the original global Swin descriptor plus a region-size-weighted global graph
correction:

```text
mean_i(token_i + feedback_assignment(i))
= mean_i(token_i) + sum_k(n_k / N) * feedback_k
```

This design follows the literal token-feedback sequence requested by the
supervisor, but it provides no token-level processing after feedback injection.
The experiment will determine whether controlled graph-feedback strength is
sufficient, and, only if necessary, whether token-specific retrieval from
graph-refined regions is more useful.

The work has two goals:

1. Restore a trustworthy control by validating and training the corrected kNN
   message direction.
2. Compare integration mechanisms without changing grouping, graph
   construction, GAT depth, backbone, data splits, or optimization settings.

## 2. Scope

### In scope

- Validate the corrected `selected_neighbor -> query` kNN direction.
- Preserve the current `token_feedback` implementation as the control.
- Add integration A: globally gated region feedback.
- Add integration B: lightweight graph-to-token cross-attention.
- Add focused configuration validation, architecture tests, and run metadata.
- Run engineering smoke tests, fixed proxy experiments, and evidence-gated full
  training.
- Select one final GAViT architecture before multi-seed evaluation.

### Out of scope

- Learnable semantic grouping, Slot Attention, or soft region assignments.
- Changing K=16 attentive spatial grouping.
- Changing kNN k=5 or adding new edge definitions.
- Changing the two-layer, four-head GAT.
- Adding a token Transformer or FFN after feedback.
- Changing the Swin backbone, augmentations, loss, optimizer, or primary metric.
- Claiming statistical significance from proxy or single-seed results.

## 3. Shared Architecture

All compared GAViT variants share this path:

```text
224x224 RGB image
  -> fully fine-tuned Swin-T
  -> 49 final-stage tokens, each 768-dimensional
  -> K=16 AttentiveSpatialGrouping
  -> 16 region nodes, each 768-dimensional
  -> corrected cosine kNN topology, k=5
  -> two GAT layers, four heads, hidden size 256 per head
  -> 16 refined region features, each 1024-dimensional
  -> integration-specific path
  -> mean pooling
  -> existing LayerNorm, dropout, and classifier
```

Cosine similarity determines kNN topology only. The existing GAT does not use
the returned cosine values as edge weights, so reports must describe the model
as cosine-kNN topology with learned GAT attention, not cosine-weighted message
passing.

## 4. Compared Integration Variants

### 4.1 Control: Current token feedback

The control retains the existing behavior:

```text
region_feedback = feedback_proj(refined_regions)
feedback_i = region_feedback[hard_assignment_i]
updated_token_i = token_i + feedback_i
```

It has no gate and is equivalent to feedback strength `alpha = 1`.

### 4.2 Integration A: Global gated feedback

Integration name:

```text
gated_feedback
```

The graph path and hard region-to-token assignment remain unchanged. A single
learnable scalar controls graph-feedback strength:

```text
alpha = sigmoid(gate_logit)
updated_token_i = token_i + alpha * feedback_i
```

Requirements:

- `alpha` is shared by all samples, regions, tokens, and channels.
- `alpha` is restricted to `[0, 1]`.
- Initialize `alpha` to `0.1`, equivalent to
  `gate_logit = log(0.1 / 0.9)`, approximately `-2.1972246`.
- Record `alpha` after every epoch.
- Do not add a per-token gate, per-channel gate, MLP, or new normalization.
- Retain the existing region feedback projection and classifier.

This variant tests one hypothesis only: unit-strength graph feedback may
destabilize or over-modify pretrained Swin representations.

### 4.3 Integration B: Lightweight graph-to-token cross-attention

Integration name:

```text
cross_attention_feedback
```

Each token retrieves its own mixture of all graph-refined region features:

```text
token_queries = Linear(768 -> 256)(tokens)
region_memory = Linear(1024 -> 256)(refined_regions)
token_context, attention = MultiheadAttention(
    query=token_queries,
    key=region_memory,
    value=region_memory,
    embed_dim=256,
    num_heads=4,
    batch_first=True,
)
projected_context = LayerNorm(Linear(256 -> 768)(token_context))
beta = sigmoid(gate_logit)
updated_tokens = tokens + beta * projected_context
```

Requirements:

- Initialize `beta` to `0.1` using the same logit initialization as A.
- All 49 tokens may attend to all 16 graph-refined regions.
- Use four cross-attention heads in the 256-dimensional bottleneck.
- Do not add a token self-attention block, Transformer encoder, or FFN.
- Preserve the existing classifier after mean pooling.
- Support optional diagnostic return of the `49 x 16` attention matrix without
  retaining it during ordinary training.
- Record `beta` after every epoch.

This variant tests whether token-specific graph retrieval is more useful than
copying one shared feedback vector to every token in a fixed region.

## 5. Configuration and Reproducibility Hardening

Before training:

- `GAViT` must reject unsupported `grouping`, `edge_type`, and `integration`
  strings instead of silently falling back to another architecture.
- BigEarthNet training and test CLIs must define matching explicit choices.
- `test_gavit.py` must not advertise `graph_only` unless the model implements
  that architecture. Historical graph-only checkpoints remain tied to their
  historical commit.
- Training and test entry points must construct identical architecture
  parameters.
- Existing state-dict checkpoint compatibility must be preserved.
- Each run must write a sidecar metadata JSON containing:
  - Git commit;
  - complete CLI arguments;
  - seed;
  - split paths and row counts;
  - model parameter count;
  - pretrained-weight path;
  - Python, PyTorch, timm, PyTorch Geometric, CUDA, and GPU versions;
  - checkpoint path;
  - log path.

The checkpoint filename and log filename must include the integration name,
experiment tier, seed, and corrected-kNN identity. Smoke, proxy, and formal
checkpoints must never share a path.

## 6. Test Design

### 6.1 Graph construction

- Directed 1-NN test confirms that each query receives messages from the
  neighbor it selected.
- Each query must have exactly `k` incoming kNN edges before any self-loops
  added internally by the installed graph library.
- Batch offsets must prevent cross-sample edges.

### 6.2 Model construction

- Current, A, and B produce `(batch_size, 19)` logits for BigEarthNet.
- Current, A, and B complete forward and backward passes.
- Backbone, graph reasoning, integration layers, and classifier receive finite
  gradients.
- Invalid configuration values fail with a clear error.
- Training and test construction from the same metadata produce compatible
  state dictionaries.

### 6.3 Gated feedback

- A initializes `alpha` to approximately `0.1`.
- `gate_logit` receives a finite, non-zero gradient on a non-degenerate batch.
- With `alpha` numerically near `1`, A matches current token feedback within
  floating-point tolerance in evaluation mode.
- With `alpha` numerically near `0`, graph feedback has negligible effect on
  logits within floating-point tolerance.

### 6.4 Cross-attention feedback

- B initializes `beta` to approximately `0.1`.
- Attention output has shape `(B, 49, 256)`.
- Diagnostic attention has shape `(B, 49, 16)` after averaging heads.
- Each token's attention weights sum to approximately one across 16 regions.
- Query, memory, attention, output projection, and gate parameters receive
  finite gradients.
- Diagnostic attention is not retained by the default training forward path.

### 6.5 Checkpoint round trip

For all three integrations:

- Save a state dict and metadata.
- Rebuild the model from metadata.
- Reload the state dict.
- Confirm identical evaluation logits for a fixed input within tolerance.

## 7. Experiment Funnel

### 7.1 Phase 0: Tests

No paid full training begins until all architecture tests pass in the Featurize
runtime, including PyTorch and PyTorch Geometric tests.

### 7.2 Phase 1: Engineering smoke

Use the existing fixed smoke data:

```text
256 train samples
128 validation samples
1 epoch
batch size 32
seed 42
```

Run current, A, and B with isolated logs and checkpoints. This phase validates
data access, local pretrained loading, CUDA forward/backward, validation,
checkpointing, throughput, and memory. Smoke metrics are not performance
evidence.

### 7.3 Phase 2: Fixed proxy experiments

Create and persist one proxy split containing 10% of the formal training and
validation rows:

```text
approximately 23,787 train samples
approximately 12,234 validation samples
5 epochs
batch size 32
seed 42
```

Proxy selection procedure:

1. Generate 100 uniform-random candidate subsets using seeds 42 through 141.
2. For each candidate, compute all 19 class prevalences.
3. Reject candidates missing a positive example for any class.
4. Score remaining candidates by their maximum absolute class-prevalence
   deviation from the corresponding full split.
5. Select the lowest-scoring candidate, with the smaller seed as the tie-break.
6. Save the chosen train and validation CSVs plus a prevalence report.
7. Reuse these exact CSVs for current, A, and B.

Record per epoch:

- validation macro mAP;
- validation macro-F1 at threshold 0.5;
- gate value for A or B;
- feedback/context norm divided by original-token norm, measured on the first
  validation batch;
- graph-branch gradient norm, measured on the first training batch;
- training throughput;
- peak GPU memory;
- wall time.

Also record the standard deviation of validation mAP over the last three
epochs. Proxy metrics select experiments only and must not enter the paper's
formal result tables.

### 7.4 Proxy rejection and promotion rules

Reject a candidate integration if any condition holds:

- NaN/Inf loss, metrics, activations, or gradients;
- failed checkpoint round trip;
- gate falls below `0.02` by the end of the proxy run;
- best validation mAP is more than `0.5` percentage points below corrected
  current-v2 proxy mAP;
- last-three-epoch mAP standard deviation exceeds current v2 by more than `0.5`
  percentage points and the candidate's mAP does not increase over those three
  epochs.

Promote A first when:

- its best proxy mAP is no more than `0.2` percentage points below current v2;
- its gate does not collapse toward zero;
- its last-three-epoch mAP standard deviation is no more than `0.5` percentage
  points above current v2;
- feedback-to-token norm remains finite. The norm ratio is diagnostic and does
  not independently promote or reject a finite candidate.

Promote B when:

- A is rejected; or
- B exceeds A by at least `0.3` proxy mAP percentage points and B's
  last-three-epoch mAP standard deviation is no more than `0.5` percentage
  points above A's.

The `0.3` and `0.5` thresholds are cost-control rules, not claims of statistical
significance.

## 8. Full Training and Evaluation

### 8.1 Architecture-selection runs

Use the formal fixed split and:

```text
30 epochs
batch size 32
AdamW
learning rate 3e-4
weight decay 1e-4
CosineAnnealingLR
seed 42
the same local Swin pretrained weights
```

Run order:

1. Corrected-kNN current v2: mandatory trustworthy control.
2. A: only if supported by proxy evidence.
3. B: only if A fails the formal criterion or B meets the explicit proxy
   promotion rule in Section 7.4.

Three full GAViT runs are an upper bound, not a commitment. Save the best
validation-mAP checkpoint and evaluate that checkpoint on the full test split.
Do not substitute final-epoch weights.

### 8.2 Metrics

Primary:

- test macro mAP.

Secondary:

- test macro-F1 at threshold 0.5;
- test micro-F1 at threshold 0.5.

Diagnostics:

- per-class AP;
- metrics for samples grouped by label cardinality: one label, two labels, and
  three or more labels;
- best and final validation metrics;
- parameters, throughput, peak memory, training wall time, and inference time.

Per-class improvements are descriptive and cannot independently determine model
selection.

### 8.3 Formal selection rule

The preferred result is a candidate whose test mAP exceeds both Swin and
corrected current v2.

If a candidate is within `0.3` test-mAP percentage points of Swin, it may still
be retained only when:

- macro-F1 and micro-F1 are each no more than `0.1` percentage points below
  Swin, allowing for printed-metric rounding;
- macro mAP on the predeclared three-or-more-positive-label group improves by
  at least `0.3` percentage points over both Swin and corrected current v2;
- its full-test macro mAP is no more than `0.1` percentage points below
  corrected current v2.

The `0.3` margin is an experiment decision boundary, not a confidence interval.
If no candidate meets the rule, stop expanding the architecture and report the
negative result and method limitations.

## 9. Final Multi-Seed Comparison

Architecture selection uses seed 42 only. After freezing one final GAViT:

```text
Swin: seed 42, 43, 44
selected GAViT: seed 42, 43, 44
```

Reuse the valid seed-42 Swin result and the selected model's formal seed-42
result. Run the remaining two seeds for each model and report mean and standard
deviation for primary and secondary metrics.

The architecture-selection budget and final multi-seed budget are separate.
After selecting the model, four additional full runs remain: Swin seeds 43 and
44, and selected-GAViT seeds 43 and 44.

## 10. Stop Conditions

Stop before a paid full run when:

- the Featurize graph-direction test fails;
- smoke cannot complete forward, backward, validation, and isolated checkpoint
  saving;
- proxy logs show numerical instability;
- output paths overlap an existing formal artifact;
- the current Git commit or data split differs from the recorded experiment;
- the candidate fails proxy promotion rules.

Stop architecture expansion when no A/B candidate meets the formal selection
rule. Do not add learnable grouping or another Transformer merely because the
result is negative.

## 11. Reporting

Every formal run must update `research_diary.md` and the corresponding result
tables with:

- exact commit, command, seed, split, preprocessing, and pretrained source;
- architecture and parameter count;
- optimizer, scheduler, loss, threshold, and epoch settings;
- peak memory, throughput, per-epoch time, wall time, and cost when available;
- best epoch, final epoch, validation and test metrics;
- checkpoint, log, metadata, and curve paths;
- interpretation against Swin and corrected current v2;
- whether the hypothesis was supported.

Negative and inconclusive results remain part of the experimental record.
