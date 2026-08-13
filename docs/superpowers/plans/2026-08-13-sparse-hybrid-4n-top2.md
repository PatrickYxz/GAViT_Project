# Sparse Hybrid 4N Top-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the supervisor-aligned `sparse_hybrid` graph topology: 48 directed four-neighbor spatial edges plus 32 directed non-spatial cosine top-2 edges for each K=16 region graph.

**Architecture:** Keep the existing Swin, attentive spatial grouping, GAT, and token-feedback path unchanged. Add one isolated graph-construction function, select it through the explicit `edge_type=sparse_hybrid` configuration, and preserve the historical `hybrid` implementation under its existing name. The builder fails closed unless it receives exactly 16 region nodes and returns the existing `(edge_index, edge_weight, batch)` interface.

**Tech Stack:** Python 3, PyTorch, PyTorch Geometric, `unittest`, argparse.

## Global Constraints

- One major experimental variable changes: graph topology only.
- `sparse_hybrid` means K=16, four-neighbor spatial adjacency, and two non-spatial feature neighbors per query.
- Edges follow PyG message direction `neighbor -> query`.
- Every graph has exactly 48 spatial plus 32 feature edges: 80 unique directed non-self edges.
- Feature candidates exclude the query and all four-neighbor spatial nodes.
- Exact cosine ties select lower region indices first.
- Cosine values select topology only; `GraphReasoning` continues to learn message strength through GAT attention.
- The historical `edge_type=hybrid` behavior is not changed or aliased.
- No grouping, region count, GAT, integration, optimizer, augmentation, loss, split, or threshold change is in scope.
- Do not start a proxy or formal training run in this implementation cycle.

---

### Task 1: Sparse hybrid graph construction

**Files:**
- Modify: `models/graph_construction.py`
- Modify: `tests/test_graph_construction.py`

**Interfaces:**
- Consumes: `features: torch.Tensor` with shape `(B, 16, D)`.
- Produces: `build_sparse_hybrid_graph(features, feature_k=2) -> tuple[edge_index, edge_weight, batch]`.
- `edge_index` has shape `(2, B * 80)`, `edge_weight` has shape `(B * 80,)`, and `batch` has shape `(B * 16,)`.

- [ ] **Step 1: Add a failing topology and batching test**

Extend `tests/test_graph_construction.py` to import `build_sparse_hybrid_graph`. Use two batched 16-node identity-feature graphs and assert:

```python
edge_index, edge_weight, batch = build_sparse_hybrid_graph(features)
self.assertEqual(edge_index.shape, (2, 160))
self.assertEqual(edge_weight.shape, (160,))
self.assertEqual(batch.tolist(), [0] * 16 + [1] * 16)
```

For each batch item, classify the first 48 local edges as spatial and the remaining 32 as feature edges. Assert 80 unique pairs, no self-edge, no cross-image edge, incoming spatial degrees `[2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 2, 3, 3, 2]`, and two feature edges per target.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
python -m unittest tests.test_graph_construction -v
```

Expected: import failure because `build_sparse_hybrid_graph` does not exist.

- [ ] **Step 3: Add a failing exclusion and deterministic-tie test**

For identity features, all valid off-diagonal cosine scores tie at zero. Assert query 0 receives feature edges `(2, 0)` and `(3, 0)`, because nodes 1 and 4 are excluded spatial neighbors. Assert query 5 receives `(0, 5)` and `(2, 5)`, because nodes 1, 4, 6, and 9 are excluded spatial neighbors.

- [ ] **Step 4: Implement the minimal builder**

In `models/graph_construction.py`:

```python
def build_sparse_hybrid_graph(
    features: torch.Tensor,
    feature_k: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

Validate `features.ndim == 3`, `features.shape[1] == 16`, and `feature_k == 2`. Enumerate four-neighbor sources for every query in row-major 4x4 order, calculate cosine similarity, mask self and spatial candidates, and use stable descending sort so equal scores retain ascending candidate index. Emit all spatial edges first and all feature edges second for each batch graph. Offset both source and target indices by `b * 16`.

- [ ] **Step 5: Run graph tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_graph_construction -v
```

Expected: all kNN and sparse-hybrid construction tests pass.

- [ ] **Step 6: Add and pass fail-closed input tests**

Assert non-3D input, K other than 16, and `feature_k` other than 2 raise `ValueError` with the rejected value in the message. Rerun the graph test module.

- [ ] **Step 7: Commit graph construction**

```bash
git add models/graph_construction.py tests/test_graph_construction.py
git commit -m "feat: add sparse hybrid graph construction"
```

---

### Task 2: GAViT and CLI configuration wiring

**Files:**
- Modify: `models/gavit.py`
- Modify: `train_bigearth.py`
- Modify: `test_bigearth.py`
- Modify: `train_gavit.py`
- Modify: `test_gavit.py`
- Create: `tests/test_sparse_hybrid_model.py`

**Interfaces:**
- Consumes: `edge_type="sparse_hybrid"` with `num_regions=16`.
- Produces: the unchanged `GAViT.forward(x) -> logits` interface and metadata containing `architecture.edge_type == "sparse_hybrid"`.

- [ ] **Step 1: Add a failing model configuration test**

Create `tests/test_sparse_hybrid_model.py`. Patch `models.gavit.SwinBackbone` with this small module returning `(B, 49, 8)` tokens:

```python
class TinyBackbone(nn.Module):
    hidden_dim = 8

    def __init__(self, **kwargs):
        super().__init__()
        self.proj = nn.Linear(3, self.hidden_dim)

    def forward(self, x):
        pooled_rgb = x.mean(dim=(-2, -1))
        token = self.proj(pooled_rgb).unsqueeze(1)
        return token.expand(-1, 49, -1)
```

Under `patch("models.gavit.SwinBackbone", TinyBackbone)`, construct:

```python
model = GAViT(
    num_classes=19,
    num_regions=16,
    knn_k=5,
    gat_hidden=4,
    gat_heads=2,
    gat_layers=2,
    grouping="attentive_spatial",
    edge_type="sparse_hybrid",
    integration="token_feedback",
    pretrained=False,
)
```

Do not catch the exception. The desired test expects construction to succeed, so the current edge-type validation must make the test fail with `ValueError`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python -m unittest tests.test_sparse_hybrid_model -v
```

Expected: failure at edge-type validation.

- [ ] **Step 3: Wire the builder into GAViT**

Import `build_sparse_hybrid_graph`, append `"sparse_hybrid"` to `VALID_EDGE_TYPES`, document the new option, and add a forward branch:

```python
elif self.edge_type == "sparse_hybrid":
    edge_index, edge_weight, batch = build_sparse_hybrid_graph(region_features)
```

Do not change the existing `hybrid`, `knn`, or `spatial` branches.

- [ ] **Step 4: Verify forward, backward, and finite outputs**

Change the model test to expect construction success. Run a two-image forward pass, assert logits shape `(2, 19)` and finite values, calculate a scalar BCE-with-logits loss, call `backward()`, and assert every present graph-parameter gradient is finite.

- [ ] **Step 5: Add the explicit CLI choice everywhere**

Add `"sparse_hybrid"` to `--edge_type` choices in all four train/test entry points. Keep defaults unchanged. Because `train_bigearth.py` already stores `vars(args)`-derived architecture fields and its checkpoint name contains `edge_type`, no new metadata field or naming scheme is introduced.

- [ ] **Step 6: Run focused tests and syntax checks**

Run:

```bash
python -m unittest tests.test_graph_construction tests.test_sparse_hybrid_model tests.test_checkpoint_roundtrip -v
python -m py_compile models/graph_construction.py models/gavit.py train_bigearth.py test_bigearth.py train_gavit.py test_gavit.py
```

Expected: all tests and syntax checks pass in the Featurize environment.

- [ ] **Step 7: Commit model and CLI wiring**

```bash
git add models/gavit.py train_bigearth.py test_bigearth.py train_gavit.py test_gavit.py tests/test_sparse_hybrid_model.py
git commit -m "feat: wire sparse hybrid topology into GAViT"
```

---

### Task 3: Engineering gate and experiment handoff

**Files:**
- Modify: `docs/featurize_runbook.md`
- Modify: `research_diary.md`

**Interfaces:**
- Consumes: committed `edge_type=sparse_hybrid` implementation.
- Produces: reproducible Featurize validation and smoke commands with isolated artifacts; no formal training command is executed.

- [ ] **Step 1: Run all locally available tests**

Run:

```bash
python -m unittest discover -s tests -v
git diff --check
```

If local PyTorch dependencies are unavailable, record that as an environment boundary and run the torch-free suites locally. Do not report torch-dependent tests as passed until Featurize supplies fresh evidence.

- [ ] **Step 2: Document the Featurize validation gate**

Add commands to `docs/featurize_runbook.md` for:

```bash
python -m unittest discover -s tests -v
python train_bigearth.py \
  --model gavit \
  --data_dir /home/featurize/data/BigEarthNet-RGB_split_smoke \
  --epochs 1 --batch_size 32 --lr 3e-4 \
  --num_regions 16 --knn_k 5 \
  --gat_hidden 256 --gat_heads 4 --gat_layers 2 \
  --grouping attentive_spatial \
  --edge_type sparse_hybrid \
  --integration token_feedback --dropout 0.1 --seed 42 \
  --run_stage smoke \
  --run_tag sparse_hybrid_4n_top2_<commit>_smoke \
  --pretrained_path bigearth_files/model.safetensors
```

The runbook must require checking the first training batch, validation completion, isolated checkpoint/metadata/last-state files, peak memory, throughput, and elapsed time. It must explicitly block the five-epoch proxy and 30-epoch formal run until this gate passes.

- [ ] **Step 3: Record the implementation state in the diary**

Add a dated entry containing the hypothesis, exact changed variable, expected 80-edge budget, Git commit, tests actually run, unresolved Featurize checks, and the statement “未启动 proxy 或正式训练”. Do not add performance metrics.

- [ ] **Step 4: Commit the engineering handoff**

```bash
git add docs/featurize_runbook.md research_diary.md
git commit -m "docs: add sparse hybrid engineering gate"
```

- [ ] **Step 5: Final verification**

Run:

```bash
git status --short --branch
git log --oneline -4
git diff --check HEAD~3..HEAD
python -m unittest discover -s tests -v
```

Report separately: locally verified facts, Featurize-recorded facts, and remaining smoke/proxy/formal work.
