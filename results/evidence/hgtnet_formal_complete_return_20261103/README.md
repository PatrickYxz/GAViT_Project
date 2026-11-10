# HGTNet formal completion acceptance

State: **SERVER_REPORTED_FORMAL_COMPLETE_IDENTITIES_MATCH_PREDICTIONS_PENDING**. Real training/refit/test reported complete; independent per-image result audit remains an offline follow-up, not a reason to rerun.

- [Original completion fragment](reported_output.txt) preserves the truncated firstline; no unknown key/epochwall reconstructed. Three complete progressJSONs plus finalCOMPLETEpath are parsed by [independent marker audit](review/validation.json),22/22passed.
- [User-returned normalized status](status_reported.json) saysstatecomplete/completedtrue/exit0,cal100/best95/refit95,fullcounts,68930updates,actual3090/parameter74619858 andmatchingconfig/recipe/manifest/candidate. [Local43-check acceptance](validation.json) and [independent39-check status audit](review/status_audit.json) pass.
- OA94.11904761904762%,macroF194.10529172930384%;23718correct/1482errors arithmetic only. Val1173/1260. MacroF1/perclass/CM not independentlyrecomputed; servermodel/checkpoint/test/source/runtime/history originalbytes notretrieved. Hashes stored asidentity references.
- Wall24032.617371428s=6h40m32.617,engine23899.926034053s;allocated4.40945673GiB/reserved4.783203125GiB,wholecurrentengine invocation incldiagnostics/reload. Attempt1, not billedrentalduration. Earlierfirstepoch9hcalibrationestimate replaced bycompletedactualwall, no bottleneck attribution.
- Same-manifest single-seed whole-method comparison: vsfixed4×4 only+0.015873pp/+4netcorrect;no paired correction/regression, significance or modulecausality claim. Fixed31,386,920params vsHGT74,619,858 (~57.94% fewer),backbone/pretrain/budget/GPU differ.
- [Before-table bytes](comparison_before_hgtnet.csv) preserve original4rows; [validate_result.py](validate_result.py) appendsone explicitindependent/serverreported-basis row tocurrentCSV/idempotentlychecks it. Legacyapprox96% table/ablation untouched. [Structured result](../../../results/hgtnet_nwpu_seed42_20261103.json) / [readable result](../../../results/hgtnet_nwpu_seed42_20261103.md).

No source/recipe/seed/threshold change, newGPU/test/training, Gitpublication/externalmessage, or model-byte download here. Next request is a small metadata/result export for offline prediction/history analysis; no further training needed.
