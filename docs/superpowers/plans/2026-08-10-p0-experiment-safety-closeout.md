# P0 Experiment Safety Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining experiment-identity, artifact-isolation, and BigEarthNet resume risks before any corrected-kNN or sparse-hybrid training.

**Architecture:** Keep model behavior unchanged. Put torch-free experiment identity and path rules in a focused `experiment_identity.py` module, keep tensor checkpoint operations in `utils.py`, make training and evaluation scripts validate checkpoint identity explicitly, store best model and last training state separately, and make every new run use a deliberate stage and run tag with overwrite protection.

**Tech Stack:** Python, PyTorch, `unittest`, JSON sidecars, Git, argparse.

## Global Constraints

- Do not modify grouping, graph construction, GAT, token feedback, data augmentation, loss, or evaluation metrics.
- Do not implement `sparse_hybrid` in this plan.
- Do not start smoke, proxy, or formal training while executing this plan.
- Preserve historical `edge_type=hybrid` semantics.
- Preserve legacy checkpoints without metadata for evaluation only; new training and resume paths must use complete metadata.
- A requested resume with missing or inconsistent artifacts must fail instead of training from scratch.
- Best-model checkpoint and last-epoch training state must have separate paths and purposes.
- Fresh runs must refuse to overwrite an existing checkpoint, metadata sidecar, or last-state file.
- Local Python has no PyTorch. Pure identity/path tests run locally; torch-dependent RED/GREEN tests run in the Featurize environment before their production changes are accepted.
- Every formal run must use a clean Git commit, `run_stage=formal`, and a unique `run_tag`.

## File Structure

- Create `experiment_identity.py`: torch-free metadata paths, validation, hashing, Git state, run-stage, run-tag, and overwrite rules.
- Modify `utils.py`: tensor checkpoint I/O, runtime identity, and last-state save/load helpers.
- Modify `train_bigearth.py`: strict run identity, strict resume, last-state saving every epoch, complete metadata.
- Modify `test_bigearth.py`: model/dataset/class-count validation before model construction.
- Modify `train_gavit.py`: run tag, overwrite protection, and complete metadata for NWPU.
- Modify `test_gavit.py`: model/dataset/class-count validation before model construction.
- Modify `baselines/swin_baseline/train_swin_baseline.py`: run tag, overwrite protection, and accurate optimizer metadata.
- Modify `baselines/swin_baseline/test_swin.py`: checkpoint identity validation.
- Modify `tests/test_checkpoint_roundtrip.py`: metadata identity and last-state regression tests.
- Create `tests/test_experiment_identity.py`: pure tests for path naming, overwrite protection, Git dirty state, and metadata validation.
- Modify `docs/featurize_runbook.md`: commands and artifact paths matching the new interfaces.
- Modify `research_diary.md`: record only verified results after tests finish.

---

### Task 1: Experiment Identity and Metadata Validation

**Files:**
- Create: `tests/test_experiment_identity.py`
- Create: `experiment_identity.py`
- Modify: `utils.py`
- Modify: `test_gavit.py`
- Modify: `test_bigearth.py`
- Modify: `baselines/swin_baseline/test_swin.py`

**Interfaces:**
- Produces: `get_git_state(cwd: str | None = None) -> dict`
- Produces: `metadata_path_for(ckpt_path: str) -> str`
- Produces: `write_checkpoint_metadata(ckpt_path: str, metadata: dict) -> None`
- Produces: `load_checkpoint_metadata(ckpt_path: str) -> dict | None`
- Produces: `validate_checkpoint_identity(metadata: dict, *, expected_model: str, expected_dataset: str, expected_num_classes: int) -> None`
- Produces: `validate_complete_architecture(metadata: dict, required_keys: tuple[str, ...]) -> None`
- Consumes: existing metadata behavior currently located in `utils.py`.

- [ ] **Step 1: Write failing pure tests**

Add tests covering these exact behaviors:

```python
def test_checkpoint_identity_rejects_wrong_dataset(self):
    meta = {"model": "gavit", "dataset": "BigEarthNet-19",
            "architecture": {"num_classes": 19}}
    with self.assertRaisesRegex(ValueError, "dataset"):
        validate_checkpoint_identity(
            meta, expected_model="gavit",
            expected_dataset="NWPU-RESISC45", expected_num_classes=45,
        )

def test_checkpoint_identity_rejects_wrong_model(self):
    meta = {"model": "swin", "dataset": "NWPU-RESISC45",
            "architecture": {"num_classes": 45}}
    with self.assertRaisesRegex(ValueError, "model"):
        validate_checkpoint_identity(
            meta, expected_model="gavit",
            expected_dataset="NWPU-RESISC45", expected_num_classes=45,
        )

def test_checkpoint_identity_rejects_wrong_class_count(self):
    meta = {"model": "gavit", "dataset": "NWPU-RESISC45",
            "architecture": {"num_classes": 19}}
    with self.assertRaisesRegex(ValueError, "num_classes"):
        validate_checkpoint_identity(
            meta, expected_model="gavit",
            expected_dataset="NWPU-RESISC45", expected_num_classes=45,
        )

def test_metadata_architecture_must_be_complete(self):
    meta = {"architecture": {"num_regions": 16}}
    with self.assertRaisesRegex(ValueError, "knn_k"):
        validate_complete_architecture(meta, ("num_regions", "knn_k"))
```

Use a temporary Git repository to assert `get_git_state()` returns the commit and changes `dirty` from `False` to `True` after modifying a tracked file.

- [ ] **Step 2: Run pure tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_experiment_identity -v
```

Expected: import or attribute failures for the three not-yet-defined helpers.

- [ ] **Step 3: Implement minimal identity helpers**

Implement the declared functions in the new torch-free `experiment_identity.py`. `validate_checkpoint_identity()` must accumulate model, dataset, and class-count mismatches and raise one `ValueError`. `validate_complete_architecture()` must reject a metadata sidecar that exists but omits a required architecture field. `get_git_state()` must return both `commit` and `dirty`; it must not describe a dirty runtime as clean. Move the existing JSON/path helpers from `utils.py` into this module and import them back into `utils.py` so existing callers keep working during the transition.

`write_checkpoint_metadata()` must be callable before the first epoch, so a fresh run has identity metadata even if interrupted before producing a best checkpoint. Update `save_checkpoint()` to store:

```python
meta["git"] = get_git_state()
```

Do not retain a second ambiguous top-level `git_commit` field.

- [ ] **Step 4: Integrate identity checks into evaluators**

Immediately after loading a metadata sidecar:

```python
validate_checkpoint_identity(
    metadata,
    expected_model=args.model,
    expected_dataset="BigEarthNet-19",
    expected_num_classes=NUM_CLASSES,
)
```

Use the equivalent fixed values in NWPU GAViT and Swin evaluators. Legacy checkpoints without metadata may use explicit CLI architecture values, but must continue printing the existing warning.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
python3 -m unittest tests.test_experiment_identity -v
python3 -m py_compile utils.py test_gavit.py test_bigearth.py \
  baselines/swin_baseline/test_swin.py
```

Expected: all pure tests pass and compilation exits 0.

---

### Task 2: Unique Artifact Paths and Overwrite Protection

**Files:**
- Modify: `tests/test_experiment_identity.py`
- Modify: `experiment_identity.py`
- Modify: `utils.py`
- Modify: `train_bigearth.py`
- Modify: `train_gavit.py`
- Modify: `baselines/swin_baseline/train_swin_baseline.py`

**Interfaces:**
- Produces: `validate_run_tag(run_tag: str) -> str`
- Produces: `validate_run_stage(run_stage: str) -> str`
- Produces: `assert_fresh_output_paths(paths: list[str]) -> None`
- Produces: `training_state_path_for(ckpt_path: str) -> str`
- Consumes: Task 1's `metadata_path_for()`.

- [ ] **Step 1: Write failing tests**

Add tests proving:

```python
def test_run_tag_rejects_path_characters(self):
    for value in ("../formal", "a/b", "", "contains space"):
        with self.subTest(value=value):
            with self.assertRaises(ValueError):
                validate_run_tag(value)

def test_fresh_output_rejects_existing_artifact(self):
    with tempfile.TemporaryDirectory() as tmp:
        existing = os.path.join(tmp, "model.pth")
        Path(existing).touch()
        with self.assertRaises(FileExistsError):
            assert_fresh_output_paths([existing])
```

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_experiment_identity -v` and confirm failure because the helpers do not exist.

- [ ] **Step 3: Implement path helpers**

`validate_run_tag()` accepts only `[A-Za-z0-9][A-Za-z0-9._-]*`. `validate_run_stage()` accepts only `smoke`, `proxy`, or `formal`. `training_state_path_for()` returns a separate `<checkpoint-stem>.last.train_state.pth` path. `assert_fresh_output_paths()` resolves each explicit path and raises with the complete collision list if any exists.

- [ ] **Step 4: Require and apply run tags**

Add this argument to all three training entry points:

```python
parser.add_argument("--run_stage", required=True,
                    choices=["smoke", "proxy", "formal"])
parser.add_argument("--run_tag", required=True,
                    help="Unique artifact identity, e.g. corrected_knn_smoke")
```

Validate both values and append them to generated checkpoint names. Before a fresh run, check the checkpoint, metadata, and last-state paths together. A resume skips the fresh-output check but must pass Task 3's strict resume checks. When `run_stage=formal`, reject a dirty Git state before loading data or pretrained weights.

Include `run_stage` and `run_tag` in metadata. For a fresh run, write the initial sidecar with `write_checkpoint_metadata()` before entering the training loop. Keep `--checkpoint_path` supported in BigEarthNet, but apply the same collision checks to explicit paths.

- [ ] **Step 5: Verify GREEN**

Run the pure tests and `py_compile` for all three training scripts. On Featurize, confirm parser-level dry invocations without `--run_stage` or `--run_tag` exit with argparse errors before data loading.

---

### Task 3: Exact BigEarthNet Last-State Resume

**Files:**
- Modify: `tests/test_checkpoint_roundtrip.py`
- Modify: `utils.py`
- Modify: `train_bigearth.py`

**Interfaces:**
- Produces: `save_training_state(model, optimizer, scheduler, path: str, *, epoch: int, best_metric: float) -> None`
- Produces: `load_training_state(model, optimizer, scheduler, path: str, *, map_location: str) -> tuple[int, float]`
- Consumes: Task 1 metadata validation and Task 2's `training_state_path_for()`.

- [ ] **Step 1: Write failing torch-dependent tests**

Using `nn.Linear`, `AdamW`, and `CosineAnnealingLR`, write tests that:

1. perform one optimizer and scheduler step;
2. save epoch 3 and best metric 71.2;
3. mutate model parameters and construct fresh optimizer/scheduler objects;
4. load the state;
5. assert model parameters, optimizer state, scheduler `last_epoch`, returned `start_epoch == 4`, and returned `best_metric == 71.2` are restored;
6. assert a missing last-state path raises `FileNotFoundError`.

- [ ] **Step 2: Verify RED on Featurize**

Run:

```bash
python -m unittest tests.test_checkpoint_roundtrip -v
```

Expected: failures for the missing training-state helpers. An import error is not an acceptable RED result.

- [ ] **Step 3: Implement last-state helpers**

Save one dictionary containing:

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch": epoch,
    "best_metric": best_metric,
}
```

`load_training_state()` restores all three states and returns `epoch + 1` and `best_metric`. It must use `weights_only=True` in the verified Featurize PyTorch version.

- [ ] **Step 4: Replace BigEarthNet resume flow**

Remove `--start_epoch`. When `--resume` is supplied:

- require checkpoint metadata and last-state file;
- validate model, dataset, class count, and every architecture value against CLI;
- restore the last model, optimizer, scheduler, epoch, and best mAP from the last-state file;
- never load the best checkpoint as training state;
- never fall back to fresh training.

Save the best model only when validation mAP improves. Save the last-state file after every completed epoch, regardless of improvement.

- [ ] **Step 5: Verify GREEN on Featurize**

Run:

```bash
python -m unittest tests.test_checkpoint_roundtrip -v
python -m unittest discover -s tests -v
```

Expected: all tests pass with zero errors.

---

### Task 4: Complete Training Metadata

**Files:**
- Modify: `utils.py`
- Modify: `experiment_identity.py`
- Modify: `train_bigearth.py`
- Modify: `train_gavit.py`
- Modify: `baselines/swin_baseline/train_swin_baseline.py`
- Modify: `tests/test_experiment_identity.py`

**Interfaces:**
- Produces: `sha256_file(path: str) -> str`
- Consumes: Tasks 1-3 metadata and run identity helpers.

- [ ] **Step 1: Write failing SHA-256 test**

Create a temporary file containing `abc` and assert:

```python
self.assertEqual(
    sha256_file(path),
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
)
```

- [ ] **Step 2: Verify RED and implement `sha256_file()`**

Run the pure test, confirm the missing-symbol failure, implement chunked hashing in `experiment_identity.py`, and rerun to GREEN.

- [ ] **Step 3: Record complete BigEarthNet identity**

Add these sidecar sections:

```python
"data": {
    "root": os.path.abspath(args.data_dir),
    "train_count": len(train_set),
    "val_count": len(val_set),
    "train_csv_sha256": sha256_file(train_csv_path),
    "val_csv_sha256": sha256_file(val_csv_path),
},
"pretrained": {
    "path": os.path.abspath(args.pretrained_path) if args.pretrained_path else None,
    "sha256": sha256_file(args.pretrained_path) if args.pretrained_path else None,
},
"execution": {
    "command": sys.argv,
    "run_stage": args.run_stage,
    "run_tag": args.run_tag,
},
```

Runtime metadata written by `save_checkpoint()` must include Python, PyTorch, timm, torch-geometric, CUDA, GPU name, Git commit, and Git dirty state.

- [ ] **Step 4: Record accurate NWPU identity**

Store absolute data root, train/validation counts, class mapping, pretrained source, command, and run tag. In the Swin baseline, define and record the actual optimizer weight decay instead of relying on an undocumented AdamW default.

- [ ] **Step 5: Verify metadata serialization**

Use a temporary directory and a small model on Featurize to save/load sidecar JSON. Assert required top-level keys are exactly present and JSON serialization succeeds.

---

### Task 5: Runbook, Full Verification, and Research Record

**Files:**
- Modify: `docs/featurize_runbook.md`
- Modify: `research_diary.md`

**Interfaces:**
- Consumes: final CLI and artifact formats from Tasks 1-4.
- Produces: copyable Featurize commands for tests and future corrected-kNN smoke/formal runs.

- [ ] **Step 1: Update the runbook**

Replace the temporary `train_bigearth_local.py` instructions with the tracked `--pretrained_path bigearth_files/model.safetensors` interface. Every smoke/formal command must include a distinct `--run_tag`, log path, and checkpoint expectation. Remove old `best_bigearth_swin.pth` move commands and the obsolete statement that resume restores model weights only.

- [ ] **Step 2: Run local verification**

Run:

```bash
git diff --check
python3 -m py_compile models/gavit.py utils.py train_bigearth.py \
  test_bigearth.py train_gavit.py test_gavit.py \
  baselines/swin_baseline/train_swin_baseline.py \
  baselines/swin_baseline/test_swin.py \
  tests/test_checkpoint_roundtrip.py tests/test_experiment_identity.py
python3 -m unittest tests.test_experiment_identity -v
```

Expected: zero failures for locally runnable checks.

- [ ] **Step 3: Run Featurize verification**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: all tests pass with zero failures and zero errors. Do not start training in this task.

- [ ] **Step 4: Review the final diff**

Confirm no graph topology, grouping, GAT, token-feedback, augmentation, loss, or metric code changed. Confirm old `hybrid` remains unchanged and `sparse_hybrid` remains unimplemented.

- [ ] **Step 5: Update the research diary**

Record the final commit, exact test commands, local and Featurize results, and explicitly state that P0 engineering protection does not constitute a model-performance experiment.

- [ ] **Step 6: Commit the closeout**

Stage only the reviewed P0 files and commit with:

```bash
git commit -m "fix: harden experiment checkpoint lifecycle"
```

Do not include unrelated result CSV files, `.superpowers/`, `.workbuddy/`, or supervisor correspondence unless the user separately requests them.
