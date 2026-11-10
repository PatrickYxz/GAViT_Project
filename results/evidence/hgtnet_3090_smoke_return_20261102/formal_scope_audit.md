# HGTNet formal seed42 scope audit — 2026-11-02

Read-only source/protocol audit; no new training, GPU probe, web research, dependency installation or Git action. The user-returned latest status supplied to this audit reports train360/val90/test0, epoch1, 180 optimizer steps, checkpoint reload error0, wall58.2575s, allocated1538072576B/reserved1688207360B, validation OA36.6667%/macro-F129.3353%, and current fix1 identities matching. These are small-sample engineering results; no formal performance inference follows.

## Next paper question and comparison boundary

**Question:** On the already frozen NWPU 20% training-pool split, how does the explicitly disclosed HGTNet independent implementation perform against the existing Swin/GAViT/DFAGCN complete methods, and what additional parameter, memory and inference cost accompanies it?

**Hypothesis to test once at seed42:** The hierarchical Transformer-plus-graph method can produce a competitive full-test result under the same sample identities; positive, null and negative results all enter the external-method comparison. Do not select another shape/aggregation/seed from its test score. This is a complete-method comparison, not an isolated graph-module ablation: backbone, partial pretraining, optimizer, augmentation, budget and hardware differ.

Frozen manifest: `ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`; calibration5040/1260, refit pool6300, final test25200; split_seed42/val_split_seed4242; training seed42. Preserve class order and exact record order/identities. A fresh refit must start from the declared initial/pretrained state, use the selected epoch budget, and fit6300 before the one final25200 test; a calibration checkpoint alone does not qualify as the 20% final-fit method.

## What the already reviewed paper actually fixes

Source: `docs/external_baselines_20261027/hgtnet_audit.md` §5 and `results/evidence/external_baselines_20261027/hgtnet/module_spec.json`, based on IV-B/C, printed pp.20322–20323. The 2026-10-31 primary review rechecked architecture, not a new complete training-recipe transcription. No original PDF/image bytes or official source/weight file were obtained.

| Field | Primary-source fact retained in existing audit | Remaining boundary |
|---|---|---|
| Epoch budget |100 epochs|Original best/final checkpoint selection, validation size and stopping policy unconfirmed.|
| Optimizer |Adam|Betas, epsilon and historical implementation defaults unconfirmed; explicitly set and disclose local values.|
| Learning rate |1e-5|No verified scheduler or warmup; do not invent paper cosine/step schedule.|
| Weight decay |5e-5|Use explicit Adam weight_decay; substituting AdamW changes recipe.|
| Batch |16|Target for formal recipe, not yet target-device batch16 evidence.|
| Loss |Cross entropy|Label smoothing and clipping policy unconfirmed.|
| Input/augmentation |RGB224; crop/flip mentioned|Exact crop scale/ratio, probabilities, interpolation, normalization and train/eval ordering unconfirmed.|
| Pretraining |Partial ImageNet|The audited torchvision Swin-B candidate/mapping is a project choice, not the author's confirmed checkpoint.|
| Original repetitions |Five|Seeds, split-resampling mechanism and SD convention unknown. This proposed run is one fixed-split training seed.|

The smoke settings Adam1e-4/batch2/one epoch/no augmentation are deliberately engineering settings. They must not become the formal recipe merely because the smoke passed. `configs/hgtnet_independent_v0.json` defines architecture, expressly no formal optimizer/data recipe. `nwpu_engine.py` is smoke-only and checks batch2/counts; it is not an existing100-epoch/refit/test runner.

## Concrete preparation before any launch

Create a separate formal recipe/entry point retaining the current model identity (`independent_text_7_to_5`, `mr_max`, GAP sum and the frozen partial Swin-B mapping). Use paper-confirmed100-epoch calibration cap, Adam1e-5/weight_decay5e-5, CE, intended batch16, seed42 and existing manifest. Explicitly freeze every unconfirmed choice before seeing final test: Adam betas/epsilon; scheduler/warmup; exact train/eval transforms; precision/TF32; validation selection/tie rule; fresh-refit epoch transfer; best/final checkpoint policy; output paths, source/config/weight hashes and expected artifact list. A constant learning rate with no warmup, or a particular crop/flip recipe, may be a declared local engineering choice; neither is confirmed to be the original recipe. This audit does not silently freeze those choices.

The final report must name the independent implementation and its model/pretraining assumptions, retain paper-reported values separately, and report test OA, macro-F1,45-class accuracy/confusion, parameter count, whole-method inference measurements and peak memory. Existing other methods used2080Ti/3080, while HGTNet uses3090, so cross-method same-card speed or memory superiority cannot be claimed from their historical runs. One seed does not establish stability or paper five-repeat reproduction.

## Cheapest remaining resource evidence

Do not infer formal ETA or select a bigger batch from wall58.2575s: wrapper wall includes input recovery/hashing, model/pretrained setup, validation, checkpoint write/reload and identity checks. Peak allocated/reserved belongs to batch2 under the complete smoke flow, including a fresh model for reload; it is not a batch16 training peak or NVML whole-process usage.

First read the existing server file, without rerunning anything:

`/home/featurize/work/hgtnet_nwpu_smoke_3090_20261102_fix1/runs/nwpu_smoke_seed42_01/history.json`

Required fields already emitted by `nwpu_engine.py`: `epoch`, `train_seconds`, `validation_seconds`, `train_images_per_second`, `optimizer_steps`. Also retain train360/val90/batch2 and matching runtime/precision identity from summary; `first_step.json.elapsed_seconds` distinguishes first-step overhead. `history.json` is a one-element list. Its training interval synchronizes CUDA at boundaries, includes data loading and first-batch gradient/update verification, and excludes validation/checkpoint reload. It is useful only for a qualified same-batch provisional projection, not a batch16 ETA.

If selecting paper batch16 still requires capacity evidence after reviewing history, the next preparation can specify a bounded forward/backward/Adam-step memory check at batch16 with the exact formal precision, model and weights, separate temporary artifacts and no formal test access. Do not automatically rent, execute, grow batch or run another full smoke. Before a paid run, record epoch throughput and total calibration+refit budget; include final25200 evaluation and persistence overhead separately.

## Evidence inspected

- `research_diary.md` latest2026-11-02 entries; `results/comparison_table.csv` and `results/ablation_study.csv` (legacy approximate rows, not current external comparison evidence).
- `results/external_baselines_nwpu_seed42_20261030.csv` (actual fixed-manifest complete-method results).
- `docs/external_baselines_20261027/{hgtnet_audit,hgtnet_implementation_design_20261031,hgtnet_3080_smoke_design_20261031,protocol_and_acceptance}.md`.
- `results/evidence/external_baselines_20261027/hgtnet/{module_spec,source_excerpts}.json` and `results/evidence/hgtnet_local_implementation_20261031/source_review/primary_review.json`.
- `external_baselines/hgtnet/nwpu_engine.py`, `configs/hgtnet_independent_v0.json` and current smoke return README.

No independent server-file download occurred in this audit. Only this audit file was added; diary, code, reports and Git were not changed by the audit.
