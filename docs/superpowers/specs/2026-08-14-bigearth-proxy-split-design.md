# BigEarthNet Fixed 10% Proxy Split Design

## 1. Purpose

Create one reproducible BigEarthNet-19 proxy split for low-cost five-epoch
engineering runs. The proxy sits between the 256/128 one-epoch smoke and the
full 30-epoch formal experiment. It checks that a candidate architecture learns
stably for several epochs without spending the full formal-training budget.

The proxy metric is not a paper result and cannot replace the full-data formal
comparison. A stable sparse-hybrid proxy is still followed by one 30-epoch,
full-split, seed-42 formal run.

## 2. Fixed Protocol

- Source data: the existing formal `train.csv` and `val.csv` only.
- Proxy size: floor of 10% of each source split.
- Expected current sizes: 23,787 train rows and 12,234 validation rows.
- Candidate seeds: every integer from 42 through 141 inclusive.
- Training protocol: 5 epochs, batch size 32, training seed 42.
- The formal test split is never sampled, copied, or evaluated during proxy
  selection or proxy training.
- The generated proxy split is reused unchanged for later proxy comparisons.

## 3. Alternatives Considered

### 3.1 Selected: deterministic candidate search

Generate 100 uniform-random candidates and select the candidate whose class
prevalences are closest to the formal train and validation splits. This follows
the approved experiment funnel, avoids an additional dependency, and makes the
selection criterion auditable.

### 3.2 Rejected: multilabel iterative stratification

Iterative stratification can match marginal distributions more directly, but it
adds a dependency and does not follow the already declared seeds 42--141
candidate procedure.

### 3.3 Rejected: one seed-42 random sample

A single random sample is simpler, but it can distort rare-class prevalence and
provides no evidence that a better candidate was not readily available.

## 4. Component Boundary

Add a standalone script:

```text
baselines/bigearth/prepare_proxy_split.py
```

The script reads split CSVs, validates them, chooses the proxy rows, and writes
the proxy artifacts. It does not import model or training modules and does not
modify `train_bigearth.py`. The implementation uses only the Python standard
library so split generation can be tested without PyTorch, NumPy, pandas, or
PyTorch Geometric.

Add focused tests in:

```text
tests/test_prepare_proxy_split.py
```

Update `docs/featurize_runbook.md` with the exact generation, verification, and
five-epoch sparse-hybrid proxy commands.

## 5. Input Contract

The command consumes:

```text
<input-dir>/train.csv
<input-dir>/val.csv
```

Each CSV must have exactly this ordered schema:

```text
patch_path,label_0,label_1,...,label_18
```

Requirements:

- every `patch_path` is non-empty;
- every label is finite and exactly binary (`0`, `1`, `0.0`, or `1.0`);
- neither split is empty;
- the requested fraction produces at least one row in each proxy split;
- the input file SHA256 hashes are computed before selection and stored in the
  report.

The script validates CSV identity and labels but does not traverse millions of
image files. Image existence remains a separate Featurize preflight check.

## 6. Candidate Selection Algorithm

For each candidate seed from 42 through 141:

1. Create an independent `random.Random(seed)` instance for train and another
   independent `random.Random(seed)` instance for validation.
2. Uniformly sample without replacement `floor(0.1 * row_count)` indices from
   each split.
3. Compute all 19 proxy prevalences separately for train and validation.
4. Reject the candidate if any class has zero positive rows in either proxy
   split.
5. For every class in both splits, compute the absolute difference between
   proxy prevalence and the corresponding full-split prevalence.
6. Score the candidate by the maximum of those 38 absolute differences.

Select the candidate with the lexicographically smallest `(score, seed)` pair.
The lower seed is therefore the explicit tie-break. Sort selected row indices
before writing so output row order follows the formal CSV order.

If all 100 candidates are rejected, exit nonzero without writing any partial
artifact.

## 7. CLI and Outputs

The Featurize command will be:

```bash
python baselines/bigearth/prepare_proxy_split.py \
  --input_dir bigearth_files/splits \
  --output_dir bigearth_files/proxy_10pct_seed42 \
  --fraction 0.1 \
  --seed_start 42 \
  --candidate_count 100
```

The output directory contains exactly:

```text
train.csv
val.csv
prevalence.json
```

`prevalence.json` records:

- algorithm name and schema version;
- fraction, seed range, selected seed, and selected score;
- source and proxy row counts;
- input and output CSV SHA256 hashes;
- for every split and label: full positive count, proxy positive count, full
  prevalence, proxy prevalence, and absolute deviation;
- maximum deviation for train, validation, and both splits combined.

The CSV artifacts contain Featurize absolute image paths and stay in the
persistent `bigearth_files` directory. The script, tests, design, and runbook
are committed to Git; generated CSVs and JSON are not committed.

## 8. Fail-Closed Behavior

- Refuse unknown, missing, duplicated, or reordered CSV columns.
- Refuse non-binary labels, empty paths, empty inputs, invalid fractions,
  and non-positive candidate counts.
- Refuse to overwrite any existing target artifact.
- Write all three outputs through temporary sibling files and publish them only
  after selection, report construction, and hashing succeed.
- On failure, remove temporary files and any final artifact created by the
  current invocation. Because the command refuses pre-existing targets before
  writing, rollback cannot delete an older proxy artifact. Never delete or
  modify source split CSVs.

## 9. Test Strategy

Use `unittest` with small synthetic CSVs to verify:

- exact floor-10% output sizes and preserved header/row order;
- deterministic output and selected seed for identical inputs;
- selection of a better prevalence candidate over a worse candidate;
- lower-seed tie-break;
- rejection of candidates missing a positive class;
- rejection when every candidate is invalid;
- validation of schema, paths, labels, fraction, and candidate count;
- refusal to overwrite existing artifacts;
- report counts, deviations, and SHA256 identities;
- no partially published artifacts after a controlled write failure.

The focused tests must fail before implementation, pass after implementation,
and the full test suite must pass before the branch is pushed.

## 10. Proxy Training Gate

Before starting the five-epoch run on Featurize:

1. Pull the committed generator and runbook on the
   `codex/sparse-hybrid-4n-top2` branch.
2. Run the full unit-test suite.
3. Generate the proxy artifacts once.
4. Verify the report, row counts, hashes, and absence of missing positive
   classes.
5. Launch one foreground sparse-hybrid proxy with an isolated `run_stage=proxy`
   run tag, log, checkpoint, metadata, and last training state.

Record all five epoch metrics, peak CUDA memory, throughput, wall time, selected
proxy seed, report hash, Git commit, and artifact paths in `research_diary.md`.
Do not add proxy metrics to `results/bigearth_comparison.csv` and do not start
formal training if the proxy has NaN/Inf, OOM, failed validation/checkpointing,
or otherwise uninterpretable learning behavior.

## 11. Non-Goals

- Changing the formal BigEarthNet split.
- Selecting a model based on proxy test performance.
- Adding early stopping or changing the optimizer, scheduler, augmentation,
  model topology, or classification threshold.
- Comparing proxy metrics directly with 30-epoch formal metrics.
- Generating proxy splits separately for each candidate architecture.
