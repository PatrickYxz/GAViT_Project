# GAViT Sparse Hybrid Graph Experiment Design

**Date:** 2026-07-28

**Status:** Design approved; implementation and training not started

## 1. Purpose

This specification defines the next controlled GAViT experiments needed to
answer two separate research questions:

1. Does correcting the cosine-kNN message direction change the BigEarthNet-19
   result relative to the historical GAViT run?
2. Under the same edge-count budget, does a sparse graph combining spatial
   proximity and feature similarity outperform corrected pure kNN?

The second question directly follows the supervisor's instruction to construct
edges using both spatial proximity and feature similarity while keeping the
graph sparse. The corrected pure-kNN run must come first because the completed
formal GAViT result used the old message direction and therefore is not a valid
control for the new hybrid graph.

This document supersedes the pure-kNN topology and run-order assumptions in
`2026-07-28-gavit-integration-experiment-design.md`. It does not cancel the
proposed token-feedback A/B experiments in that document. Those experiments
remain paused and, if later resumed, must use the selected hybrid topology as
their fixed graph control. In this document, that fixed topology is
`sparse_hybrid_4n_top2`; changing it later requires a separate explicit design
decision.

## 2. Evidence and Decision Boundary

The formal BigEarthNet-19 results currently recorded are:

- Swin-T baseline: test mAP 70.9%.
- Historical GAViT v2: test mAP 70.4%.

The historical GAViT run used pure cosine kNN before the message direction was
corrected. Commit `649e7c1` changes the intended flow to
`selected_neighbor -> query`, but that change has not yet been validated in the
Featurize runtime or evaluated in a full controlled run.

The new hybrid graph is not expected to guarantee an accuracy gain. Its purpose
is to test the supervisor-aligned hypothesis that local spatial relations and
non-local feature relations provide more useful structure than feature
similarity alone.

## 3. Scope

### In scope

- Validate the corrected pure-kNN construction on Featurize.
- Run one corrected pure-kNN BigEarthNet-19 control with seed 42.
- Add a new sparse hybrid graph containing both spatial and feature edges.
- Give pure kNN and sparse hybrid the same pre-GAT edge-count budget.
- Run engineering tests, a short proxy, and one formal sparse-hybrid
  BigEarthNet-19 experiment with seed 42.
- Record overall, per-class, stability, and resource results.

### Out of scope

- Changing region grouping, region count, GAT width, GAT depth, token feedback,
  optimizer, augmentation, data split, loss, or classification thresholds.
- Implementing learnable soft assignment or Slot Attention.
- Implementing feedback integration A or B.
- Running NWPU-RESISC45 before the final graph and feedback architecture is
  selected.
- Running three seeds for intermediate candidates.
- Treating a five-epoch proxy metric as a paper result.

## 4. Shared Model Architecture

Both compared GAViT variants use the same path:

```text
image
  -> Swin-T backbone
  -> 49 final tokens, each 768-dimensional
  -> K=16 AttentiveSpatialGrouping on a fixed 4x4 grid
  -> 16 region nodes
  -> graph construction
  -> two-layer, four-head GAT
  -> existing hard-assignment token feedback
  -> immediate mean pooling
  -> BigEarthNet-19 classifier
```

Only graph construction changes between the corrected-kNN control and the
sparse-hybrid candidate. The existing token-feedback weakness remains a known
but deliberately fixed factor so that this experiment isolates edge topology.

## 5. Corrected Pure-kNN Control

For each of the 16 query nodes:

1. L2-normalize all region features.
2. Compute pairwise cosine similarity.
3. Exclude the query node itself.
4. Select the five most similar region nodes.
5. Create five directed edges in the form
   `selected_neighbor -> query`.

This creates exactly:

```text
16 queries x 5 incoming neighbors = 80 directed, non-self edges
```

Cosine similarity selects the topology only. Its numerical value is not passed
to GAT as an edge weight; message strength is learned by GAT attention.

## 6. Sparse Hybrid Graph

### 6.1 Spatial component

Treat the 16 regions as a 4x4 grid. Two regions are spatial neighbors only when
they share a horizontal or vertical boundary. Diagonal contact does not count.

The grid contains 24 undirected four-neighbor adjacencies:

```text
12 horizontal + 12 vertical = 24 undirected adjacencies
```

Materialize both directions for every adjacency:

```text
24 x 2 = 48 directed spatial edges
```

Every directed spatial edge follows `spatial_neighbor -> query`. Corner,
non-corner boundary, and interior nodes consequently receive two, three, and
four spatial edges respectively.

### 6.2 Feature component

For each query node:

1. Start from all other region nodes.
2. Exclude the query itself.
3. Exclude every four-neighbor spatial node already connected to the query.
4. Rank the remaining candidates by cosine similarity.
5. Select the top two candidates.
6. Create `feature_neighbor -> query` edges.

This creates exactly:

```text
16 queries x 2 feature neighbors = 32 directed feature edges
```

If cosine scores are exactly tied, lower region index is the deterministic
tie-breaker. Cosine scores select edges but are not passed to GAT as weights.

### 6.3 Combined budget and uniqueness

The sparse hybrid graph contains:

```text
48 spatial edges + 32 feature edges = 80 unique directed edges
```

The feature candidate exclusion makes spatial and feature sets disjoint by
construction. The final graph must still be checked for self-edges and duplicate
source-target pairs before it enters GAT.

The 80-edge value is the explicit sparse-edge budget. It makes the comparison
fair because corrected pure kNN also contains 80 directed edges. The budget
counts edges supplied by graph construction; any self-loops automatically added
inside `GATConv` are not part of this budget and must be configured identically
for both variants.

Each edge may carry a diagnostic label, `spatial` or `feature`, for counting and
visualization. This label is metadata only and must not change the GAT
computation in this experiment.

## 7. Configuration and Experiment Identity

The new topology must have a distinct explicit configuration value:

```text
edge_type=sparse_hybrid
```

The existing `edge_type=hybrid` means the historical direct concatenation of
kNN and spatial edges. It must not be silently redefined, aliased, or reported
as the new sparse hybrid. Unknown configuration values must raise an error
instead of falling back to another implementation.

Artifacts must distinguish the two formal runs:

- `corrected_knn`
- `sparse_hybrid_4n_top2`

Every log and checkpoint must record at least:

- Git commit;
- complete configuration and command;
- seed;
- dataset and split identity;
- pretrained-weight source;
- region count and grouping type;
- edge type and expected edge counts;
- GAT and token-feedback settings;
- dependency/runtime identity;
- log and checkpoint paths.

Smoke, proxy, and formal runs must use separate output paths so that no formal
checkpoint can be overwritten.

## 8. Validation and Error Handling

### 8.1 Corrected-kNN construction tests

For every graph in a batch, verify:

- exactly five incoming constructed edges per query;
- exactly 80 directed constructed edges;
- all edges follow `selected_neighbor -> query`;
- no self-edge;
- no duplicate source-target pair;
- no edge crossing from one image graph into another.

### 8.2 Sparse-hybrid construction tests

For every graph in a batch, verify:

- exactly 48 directed spatial edges;
- exactly 32 directed feature edges;
- exactly 80 unique directed constructed edges;
- two feature edges enter every query;
- two to four spatial edges enter each query according to grid location;
- no diagonal node is labeled as a spatial neighbor;
- every selected feature neighbor is among the cosine top two after excluding
  self and all spatial neighbors;
- all edges follow `neighbor -> query`;
- no self-edge, duplicate pair, or cross-image edge.

### 8.3 Model and checkpoint tests

For both graph types, verify:

- forward output has shape `(batch_size, 19)`;
- one backward pass produces finite gradients in graph parameters;
- graph outputs and loss contain no NaN or Inf;
- invalid edge-type values fail explicitly;
- a metadata-driven checkpoint reload rebuilds the same architecture and
  reproduces identical evaluation logits within numerical tolerance.

Any failed construction, model, or checkpoint test blocks smoke and formal
training. Runtime NaN/Inf, incorrect label dimensions, missing pretrained
weights, or output-path collisions must stop the run rather than trigger an
implicit fallback.

## 9. Experiment Sequence

### Stage 0: Featurize validation

1. Run the kNN direction unit test in the actual Featurize environment.
2. Confirm the corrected-kNN edge counts and direction.
3. Run a small 256-train/128-validation, one-epoch smoke test.
4. Confirm forward, backward, validation, and checkpoint saving.
5. Record warmed throughput, peak GPU memory, first-epoch time, and projected
   formal-run time.

Do not start formal training if any Stage 0 check fails.

### Stage 1: Corrected pure-kNN formal control

Run BigEarthNet-19 for 30 epochs with seed 42 using:

- K=16 attentive spatial grouping;
- corrected cosine kNN with k=5;
- the current two-layer, four-head GAT;
- the current hard token-feedback integration;
- the same data split, pretrained weights, optimizer, learning rate, loss,
  scheduler, augmentations, and evaluation settings as the recorded baseline.

This run answers whether the message-direction correction changes the historical
GAViT result. It does not by itself satisfy the supervisor's combined-edge
request.

### Stage 2: Sparse-hybrid engineering gate

After implementing the sparse hybrid:

1. Run all graph, model, and checkpoint tests.
2. Run the same one-epoch smoke protocol.
3. Run one fixed five-epoch proxy with seed 42.

The proxy is used only to detect engineering failure, severe instability, or
obviously broken learning. Its metric is not used as a paper result and is not
directly compared with the 30-epoch formal numbers.

### Stage 3: Sparse-hybrid formal comparison

If Stage 2 is stable, run the sparse hybrid on BigEarthNet-19 for 30 epochs with
seed 42 and all non-topology settings identical to Stage 1.

The formal sparse-hybrid run remains necessary even when its five-epoch proxy
does not exceed the corrected-kNN proxy, because it directly tests the
supervisor's requested combined relation design. It may be cancelled only for
a documented engineering failure, invalid comparison, or training instability
that makes the result uninterpretable.

### Stage 4: Later architecture work

After comparing corrected kNN and sparse hybrid:

- use `sparse_hybrid_4n_top2` as the fixed graph for token-feedback experiments;
- evaluate simple gated feedback A before cross-attention feedback B;
- run three seeds only for Swin and the ultimately selected GAViT;
- evaluate NWPU-RESISC45 only after the final GAViT architecture is selected.

## 10. Metrics and Interpretation

For each formal BigEarthNet-19 run, save:

- best-validation and final-epoch mAP;
- test mAP from the selected checkpoint;
- macro-F1 and micro-F1;
- per-class AP;
- classification threshold and label-cardinality statistics;
- loss and metric curves;
- parameter count;
- peak GPU memory;
- warmed throughput;
- time per epoch and total training time.

The primary sparse-hybrid comparison is against corrected pure kNN, not only
against the historical GAViT result. Swin test mAP 70.9% remains the external
baseline.

Interpret the outcomes as follows:

- **Strong positive:** sparse hybrid exceeds corrected kNN in test mAP and also
  exceeds the Swin baseline.
- **Limited positive:** sparse hybrid exceeds corrected kNN, but not Swin.
- **Potentially useful trade-off:** sparse hybrid is within 0.3 percentage
  points of Swin mAP, macro-F1 and micro-F1 do not materially decline, and at
  least three coherent relationship-sensitive classes improve.
- **Negative:** sparse hybrid does not exceed corrected kNN and adds no coherent
  per-class or interpretability benefit.

Small numerical differences from a single seed are not evidence of stable
superiority. They determine only whether a candidate is worth final multi-seed
evaluation.

## 11. Required Diagnostics and Reporting

For corrected kNN and sparse hybrid, save representative BigEarthNet
visualizations containing mixed water, vegetation, agricultural, and built-up
components. The visualization must distinguish spatial and feature edges and
use the corrected message direction. Airport, bridge, church, stadium, harbor,
and forest remain required examples when the later NWPU evaluation is run; they
are not BigEarthNet-19 class requirements.

For the sparse hybrid, also report:

- spatial and feature edge counts;
- feature-neighbor cosine score distribution;
- reciprocal-feature-edge rate;
- whether feature edges connect non-local but plausibly related regions;
- GAT attention statistics separated by diagnostic edge type where available;
- training stability relative to corrected kNN.

After each formal run, immediately update `research_diary.md` and the relevant
result tables with the command, commit, configuration, checkpoint, metrics,
resources, and interpretation. Negative results must be retained.

## 12. Completion Criteria

This experiment cycle is complete only when:

1. corrected-kNN construction is validated on Featurize;
2. the corrected pure-kNN formal control has complete artifacts;
3. the sparse-hybrid construction passes all specified tests;
4. the sparse-hybrid formal run has complete artifacts, or a documented
   engineering/stability failure justifies cancellation;
5. both formal results are compared under the same evaluation procedure;
6. the diary and result tables record the outcome without mixing historical,
   proxy, validation, and test metrics.

No model code or training is authorized by this design document alone. A
separate reviewed implementation plan is required before execution.
