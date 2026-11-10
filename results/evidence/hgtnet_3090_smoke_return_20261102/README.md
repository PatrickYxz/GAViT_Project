# HGTNet real NWPU3090 smoke completion marker — 2026-11-02

The user returned six training-progress events ending at180 optimizer steps/360 samples, complete=true progress, and the fix1 COMPLETE marker. `reported_output.txt` preserves the pasted excerpt, including its truncated first prefix (`GTNET`); it is not a complete downloaded log.

The subsequent full short-status fields are now saved in `status_reported.json` (locally normalized user-pasted JSON). **Current accepted state: SERVER_REPORTED_REAL_NWPU_3090_SMOKE_PASS_STATUS_IDENTITIES_MATCH.** Complete/exit0/counts/180steps/epoch1/UUID/config/weights/manifest match the delivery; checkpoint numeric reload is reported0.0. Wall58.257502633s, PyTorch peaks allocated1.432441711GiB/reserved1.572265625GiB; validation OA36.666666666666664% (33/90 arithmetic) and macro-F129.33525733525733% belong only to this smoke. `status_validation.json`56/56 and independent `review/status_audit.json`70/70 checks pass. No rerun is needed.

The checkpoint bytes, validation predictions, complete runtime/source snapshot and original server summary/exit files were not downloaded. Macro-F1 and zero numeric reload/checkpoint SHA remain server-reported; we do not claim to recompute them from originals. Whole-smoke wall includes recovery/setup/validation/persistence and allocator peaks include diagnostics/reload, so neither is a batch16 throughput/ETA or whole-process memory measurement. See the [result report](../../hgtnet_nwpu_smoke_3090_20261102.md) and [formal preparation](../../../docs/external_baselines_20261027/hgtnet_formal_preparation_20261102.md).

## Earlier marker-only acceptance

Accepted state: **SERVER_REPORTED_HGTNET_REAL_NWPU_SMOKE_COMPLETE_MARKER_SUMMARY_PENDING**. No rerun is needed to obtain the existing result summary.

- `validation.json`:41/41 stdlib excerpt/source checks pass. Delivered fix1 file SHA90c60dabcd185ffd0a1b575f8544e7363a469c4e3616528955cb3645c2ae7c42 matches; all17 payloads match their manifest and16 locally present source payloads match current files.
- `review/marker_source_audit.json`: independent read-only audit of the actual archive and normal success path.
- Normal code prints COMPLETE only after train360/val90/test0, one epoch, finite gradients/updated parameters, ordered validation and metrics, checkpoint save and independent-model reload error≤1e-6, history/predictions writes, final source/input/data identity rechecks, summary completed=true, run.exit0 and complete progress.

At the marker-only stage this accepted the user's reported real-data engineering completion under the delivered contract; the precise status values were still pending at that time. The subsequent user status supplies the values summarized above. Original server files still have not been read directly. No population/seed gain, formal performance result or original-paper numerical reproduction is inferred.

One read-only command obtains the short existing result:

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_nwpu_smoke_3090_20261102_fix1.pyz status
```

This does not import Torch, repeat the smoke, run test, retrain or modify artifacts. No GPU/remote/Git action was taken locally in this acceptance turn. Formal recipe/resource preparation follows the actual summary, not the selected small-sample score.
