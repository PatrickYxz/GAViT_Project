# PaperPilot-Reranker Design Spec

Date: 2026-05-18
Status: Draft for user review
Target repository: `PaperPilot-Reranker`
Reference repositories:
- Local design/workflow reference: `C:\Users\Administrator\PycharmProjects\GAViT_Project`
- PaperPilot data/runtime reference: `C:\Users\Administrator\PycharmProjects\PaperPilot`

## 1. Purpose

`PaperPilot-Reranker` is an independent GPU-training project for improving PaperPilot evidence retrieval. It builds a supervised `question, chunk -> relevance` dataset from QASPER and PaperPilot ColBERT retrieval traces, fine-tunes a cross-encoder evidence reranker, reranks ColBERT top-k chunks, and evaluates both retrieval quality and downstream QA quality.

The project is intentionally separate from PaperPilot. PaperPilot remains the agent/runtime system; PaperPilot-Reranker owns data export, label construction, model training, reranking evaluation, and experiment reporting.

## 2. External Basis

This design relies on three current public interfaces:

- QASPER is a scientific-paper QA dataset with questions, full paper text, answers, and supporting evidence. Its Hugging Face dataset card lists both question answering and evidence selection as supported tasks.
- Sentence-Transformers supports cross-encoder reranker training, and current docs recommend the trainer-based API (`CrossEncoderTrainer`) instead of older `fit`-style training.
- `BAAI/bge-reranker-base` is published as a BGE reranker model; the model card describes BGE rerankers as cross-encoder models that trade speed for better ranking quality.

References:
- https://huggingface.co/datasets/allenai/qasper
- https://sbert.net/docs/cross_encoder/training_overview.html
- https://huggingface.co/BAAI/bge-reranker-base

## 3. Goals

The project must support:

1. Exporting PaperPilot/QASPER data into a stable local format.
2. Constructing `question, chunk, label` examples from oracle evidence spans and ColBERT candidate chunks.
3. Training a cross-encoder reranker on GPU cluster hardware.
4. Reranking ColBERT top-k chunks and comparing against raw ColBERT rank.
5. Reporting MRR, NDCG, evidence hit rate, and downstream QA pass rate.
6. Running reproducible experiments through Slurm jobs that follow the existing GAViT server workflow.

## 4. Non-Goals

The first version will not:

- Replace PaperPilot's ColBERT retriever.
- Serve the reranker as a production MCP server.
- Fine-tune ColBERT itself.
- Depend on live LLM calls during reranker training.
- Require PaperPilot to import this project as a package.

PaperPilot integration is a later phase after offline metrics prove the reranker is useful.

## 5. Repository Layout

```text
PaperPilot-Reranker/
  README.md
  environment.yml
  requirements.txt
  configs/
    data_qasper.yaml
    train_bge_base.yaml
    train_minilm.yaml
    eval_default.yaml
  jobs/
    run_export_dataset.sh
    run_train_reranker.sh
    run_eval_reranker.sh
    run_ablation.sh
  src/ppreranker/
    data/
      export_from_paperpilot.py
      build_labels.py
      dataset.py
      split.py
    models/
      cross_encoder.py
    training/
      train.py
      evaluate.py
    eval/
      retrieval_metrics.py
      downstream_qa.py
      report.py
    utils/
      io.py
      text_match.py
      seed.py
  data/
    raw/
    processed/
    splits/
  checkpoints/
  results/
    metrics/
    predictions/
    reports/
  logs/
```

Git policy:

- Track source, configs, jobs, README, and small fixture data only.
- Ignore `data/raw`, `data/processed`, `data/splits`, `checkpoints`, `logs`, and large model/result artifacts unless explicitly promoted to a small report.

## 6. Data Inputs

The project reads from the PaperPilot checkout but writes its own copies under `PaperPilot-Reranker/data`.

Primary local inputs:

- `PaperPilot/data/eval/qasper_subset.jsonl`
  - Contains `case_id`, `arxiv_id`, `paper_title`, `abstract`, `full_text`, `question`, and `oracle_spans`.
- `PaperPilot/data/traces/*.jsonl`
  - Contains PaperPilot agent events for each QASPER case.
  - Used to recover actual `mcp__colbert__search` calls, searched queries, `paper_id`, and requested `top_k`.
- `PaperPilot/data/colbert_index/*/chunks.json`
  - Contains chunk id to chunk text mappings produced by PaperPilot's ColBERT index manager.

Important PaperPilot compatibility constraint:

Current PaperPilot ColBERT search returns `paper_id`, `chunk_text`, and `score`, but not a stable `chunk_id`. Therefore the reranker dataset builder should not rely only on trace tool results. It should prefer one of these stable strategies:

1. Reconstruct candidate pools by replaying search queries against exported ColBERT/chunk data when scores and chunk ids are available.
2. Match trace `chunk_text` back to `chunks.json` by normalized text when replay is unavailable.
3. Later, add `chunk_id` to PaperPilot's search output if deeper PaperPilot integration is approved.

Strategy 1 is preferred for the full project. Strategy 2 is acceptable for an initial data audit.

## 7. Data Output Contract

The canonical supervised dataset is JSONL with one row per question-chunk pair:

```json
{
  "case_id": "qasper-1909.00694-q0",
  "paper_id": "1909.00694",
  "question": "What is the seed lexicon?",
  "chunk_id": "1909.00694::chunk_12",
  "chunk_text": "The seed lexicon consists of positive and negative predicates...",
  "colbert_query": "seed lexicon positive negative words list",
  "colbert_rank": 3,
  "colbert_score": 18.7,
  "label": 1,
  "label_reason": "oracle_span_exact",
  "oracle_spans": ["seed lexicon consists of positive and negative predicates"]
}
```

Required fields:

- `case_id`
- `paper_id`
- `question`
- `chunk_id`
- `chunk_text`
- `label`
- `oracle_spans`

Recommended fields:

- `colbert_query`
- `colbert_rank`
- `colbert_score`
- `label_reason`

## 8. Label Construction

Positive labels:

- `label=1` if any normalized QASPER oracle span appears in the normalized chunk text.
- `label=1` if fuzzy span matching passes a conservative threshold when exact matching fails.

Negative labels:

- `label=0` for ColBERT candidate chunks from the same paper and question that do not match any oracle span.
- Add optional in-paper random negatives only after hard negatives are available.

Split rule:

- Split by `paper_id`, not by row or question.
- Default split: 70% train, 15% dev, 15% test by paper.
- The same paper must not appear in more than one split.

Dataset sanity checks:

- Count questions, papers, positive pairs, negative pairs, and positive rate.
- Report questions with no positive chunk in the candidate pool.
- Report maximum possible evidence hit rate at each candidate pool size before training.
- Fail fast if a split has zero positives.

## 9. Model Design

First implementation:

- Model family: cross-encoder reranker.
- Input: `(question, chunk_text)`.
- Output: one scalar relevance score.
- Initial model: `BAAI/bge-reranker-base`.
- Fast baseline model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

Training objective:

- Version 1: binary classification with `BCEWithLogitsLoss`.
- Version 2: pairwise ranking loss within each `case_id`, where positive chunks should score higher than hard negatives.

Default hyperparameters:

- `max_length`: 512
- `batch_size`: 8 or 16, depending on GPU memory
- `epochs`: 2-5
- `learning_rate`: 2e-5
- `warmup_ratio`: 0.1
- `weight_decay`: 0.01
- `seed`: 42
- early stopping metric: dev `NDCG@10`

## 10. Training Script

Main command:

```bash
python -m ppreranker.training.train \
  --config configs/train_bge_base.yaml \
  --train data/splits/train.jsonl \
  --dev data/splits/dev.jsonl \
  --output checkpoints/bge-reranker-qasper
```

Responsibilities:

- Load JSONL pair dataset.
- Tokenize `(question, chunk_text)` pairs.
- Train with mixed precision when CUDA is available.
- Evaluate dev MRR/NDCG/hit rate after each epoch.
- Save the best checkpoint by dev `NDCG@10`.
- Write `trainer_state.json`, `metrics_dev.json`, and `config_resolved.yaml`.

## 11. Evaluation Script

Main command:

```bash
python -m ppreranker.training.evaluate \
  --config configs/eval_default.yaml \
  --checkpoint checkpoints/bge-reranker-qasper \
  --input data/splits/test.jsonl \
  --output results/predictions/test_bge_reranked.jsonl
```

Retrieval metrics:

- `MRR@5`, `MRR@10`, `MRR@20`
- `NDCG@5`, `NDCG@10`, `NDCG@20`
- `evidence_hit_rate@1`, `@3`, `@5`, `@10`
- raw ColBERT metrics over the same candidate pool
- absolute and relative gain from reranking

Downstream QA metric:

- Build a fixed prompt from top-n reranked chunks.
- Use the same QA model/provider as PaperPilot eval when available.
- Score answer pass/fail with PaperPilot's existing oracle-span containment rule.
- Report pass rate against the existing PaperPilot baseline.

Downstream QA evaluation should be separated from retrieval evaluation because it may require network/API credentials and costs.

## 12. Slurm and Server Workflow

Follow the GAViT cluster pattern:

- Local development on Windows.
- Push or otherwise sync through Git.
- Server path convention: `/home/yang1004/PaperPilot-Reranker/`.
- Run `git pull` on the server.
- Submit jobs with `sbatch jobs/*.sh`.
- Put Slurm output in `logs/`.
- Put metrics and prediction artifacts in `results/`.

Environment:

```yaml
name: ppreranker
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch
  - pytorch-cuda=12.1
  - pip
  - pip:
      - transformers
      - sentence-transformers
      - accelerate
      - datasets
      - scikit-learn
      - pandas
      - numpy
      - tqdm
      - pyyaml
      - rapidfuzz
      - evaluate
```

Training job template:

```bash
#!/bin/bash
#SBATCH --job-name=PPReranker_Train
#SBATCH --output=logs/reranker_train_%j.out
#SBATCH --error=logs/reranker_train_%j.err
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

mkdir -p logs

module load Miniforge3
source activate
conda activate ppreranker

cd /home/yang1004/PaperPilot-Reranker/

echo "===== ENV INFO ====="
which python
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
nvidia-smi
echo "===================="

python -m ppreranker.training.train \
  --config configs/train_bge_base.yaml \
  --train data/splits/train.jsonl \
  --dev data/splits/dev.jsonl \
  --output checkpoints/bge-reranker-qasper
```

## 13. Experiment Plan

Phase 1: Data audit

- Export QASPER subset, traces, and chunks.
- Build candidate pools.
- Compute raw ColBERT evidence hit rate.
- Identify cases where oracle evidence is absent from the candidate pool.

Phase 2: Baselines

- Raw ColBERT rank.
- Zero-shot `BAAI/bge-reranker-base`.
- Zero-shot MiniLM cross-encoder.

Phase 3: Fine-tuning

- Fine-tune BGE reranker with BCE loss.
- Compare with pairwise ranking loss.
- Track dev NDCG@10 and test metrics.

Phase 4: Ablations

- Candidate pool size: 10, 20, 50.
- Negative source: hard negatives only vs hard plus random negatives.
- Model: MiniLM vs BGE base.
- Input length: 256 vs 512.
- Top-n chunks passed to downstream QA: 3, 5, 8.

Phase 5: PaperPilot integration decision

- If retrieval metrics improve but QA does not, inspect prompt/context ordering.
- If retrieval and QA both improve, implement an offline rerank step first.
- Only after offline success, consider MCP/server integration.

## 14. Success Criteria

Minimum viable success:

- Dataset builder produces valid train/dev/test JSONL splits.
- Raw ColBERT baseline metrics are reproducible.
- One Slurm training run completes and saves a checkpoint.
- Evaluation writes both metrics JSON and per-case prediction JSONL.

Research success:

- Reranker improves `NDCG@10` and `MRR@10` over raw ColBERT on the held-out paper split.
- Reranker improves or preserves `evidence_hit_rate@5`.
- Downstream QA pass rate improves over the current PaperPilot baseline under the same eval protocol.

## 15. Risks and Mitigations

Risk: current traces may not preserve stable `chunk_id`.

- Mitigation: reconstruct candidates from `chunks.json` and search metadata; later add `chunk_id` to PaperPilot search output if needed.

Risk: oracle span exact matching undercounts positives because chunks are tokenized/normalized differently.

- Mitigation: implement exact match first, then conservative fuzzy matching with an audit report.

Risk: train/test leakage through same paper chunks.

- Mitigation: enforce paper-level splits and fail if a paper appears in multiple splits.

Risk: downstream QA improvement may lag retrieval metric improvement.

- Mitigation: report retrieval and QA separately, then inspect failed cases by evidence presence and answer synthesis.

Risk: cluster environment drift.

- Mitigation: print Python/Torch/CUDA/nvidia-smi at the start of every Slurm job, following the GAViT job pattern.

## 16. Implementation Order

1. Scaffold repository and ignore rules.
2. Add environment and Slurm job templates.
3. Implement export and data audit scripts.
4. Implement label builder and paper-level split.
5. Implement raw ColBERT metric computation.
6. Implement zero-shot reranker evaluation.
7. Implement training script.
8. Implement downstream QA evaluation.
9. Run ablations and write final experiment report.

## 17. Review Questions

Before implementation, confirm:

1. Should the new project live as a new GitHub repository named `PaperPilot-Reranker`?
2. Should first training use `BAAI/bge-reranker-base` as the main model and MiniLM only as a speed baseline?
3. Should downstream QA evaluation use the same DeepSeek/Anthropic-compatible client as PaperPilot Day 16/18 eval?
