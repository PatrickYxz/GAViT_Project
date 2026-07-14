# BigEarthNet GAViT Local Pretrained Smoke Test Design

## Goal

Prepare one reproducible BigEarthNet-19 GAViT experiment on Featurize without
depending on Hugging Face network access and without starting a paid full run
before the training path is verified.

The experiment compares GAViT with the completed Swin-T baseline. It therefore
uses the same ImageNet-pretrained Swin-T weights, data split, optimizer settings,
seed, epoch count, batch-size target, preprocessing, and validation metric.

## Selected Configuration

- GPU: NVIDIA GeForce RTX 4090, 24 GB
- Model: GAViT with Swin-T backbone
- BigEarthNet classes: 19
- Regions: 16
- Grouping: `attentive_spatial`
- Graph: `knn`, `knn_k=5`
- GAT: 2 layers, 4 heads, hidden dimension 256 per head
- Integration: `token_feedback`
- Dropout: 0.1
- Seed: 42
- Optimizer and schedule: existing BigEarthNet defaults
- Formal training target: 30 epochs, batch size 32
- Primary metric: validation macro mAP
- Supporting metrics: macro-F1, micro-F1, and per-class AP at threshold 0.5

Only the GAViT architecture differs from the Swin baseline. This keeps the main
comparison interpretable.

## Local Pretrained Weight Flow

Add an optional `--pretrained_path` argument to `train_bigearth.py`. The value is
passed through `GAViT` and `SwinBackbone` to `timm.create_model` using:

```python
pretrained_cfg_overlay={"file": pretrained_path}
```

The existing persistent weight file is:

```text
/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors
```

The same option also applies to the direct Swin baseline construction so future
baseline runs do not require a temporary server-only launcher.

When `--pretrained_path` is supplied, the program validates that the file exists
before model construction and fails with a clear message if it does not. When it
is omitted, existing online-pretrained behavior remains unchanged.

## Checkpoint Isolation

Add an optional `--checkpoint_path` argument to `train_bigearth.py`. If omitted,
the current model-derived checkpoint name remains unchanged. The smoke test uses:

```text
checkpoints/smoke_bigearth_gavit_K16_attentive_spatial_knn_token_feedback.pth
```

The formal run uses the existing canonical GAViT checkpoint name. A smoke test
can therefore never overwrite a formal checkpoint.

## Verification Sequence

1. Unit-test the pure pretrained configuration helper locally, including the
   missing-file failure case.
2. Run Python syntax checks and the existing local unit tests.
3. Synchronize the code to Featurize.
4. Reuse the existing smoke split: 256 train samples and 128 validation samples.
5. Run one GAViT epoch with batch size 16 and the independent smoke checkpoint.
6. Confirm local pretrained loading, CUDA device, 19-dimensional logits,
   forward/backward completion, validation metrics, and checkpoint creation.
7. Record peak GPU memory with `nvidia-smi` while the smoke run is active.
8. If peak usage leaves comfortable headroom, run a second short throughput
   check at batch size 32. This is not another training experiment; it only
   determines whether the formal run can match the baseline batch size.
9. Estimate per-epoch time and total 30-epoch duration before requesting approval
   for the paid formal run.

## Failure Handling

- A missing or invalid local weight path stops before training begins.
- CUDA out-of-memory at batch size 32 falls back to batch size 16 and is recorded
  as a resource constraint; model hyperparameters are not changed.
- Any shape, dependency, NaN loss, validation, or checkpoint failure stops the
  process after the smoke test. No full run is started.
- Smoke results are operational checks only and are never reported as paper
  performance.

## Outputs

- Reproducible local-pretrained support in the tracked training code
- Unit tests for weight-path configuration
- Independent smoke checkpoint and log
- Recorded GPU model, peak memory, throughput, and projected full-run duration
- A final formal command ready for explicit approval, but not launched
  automatically
