# Independent returned-prediction audit

[audit_predictions.py](audit_predictions.py) reads the user-supplied archive directly using Python stdlib tar/JSON parsing; archive content is treated as data and is never executed. It writes only this review directory. Run from the project root:

```sh
/private/tmp/gavit-hgtnet-probe-cpu-20261102/bin/python results/evidence/hgtnet_results_return_20261104/review_predictions/audit_predictions.py
```

[validation.json](validation.json) records **131/131 passed checks**, including per-class checks. The raw test file has the previously reported SHA869986974e4801f2234fd2fa3bace383af2f092fd20359fbe1a252d4af7e37ae.

All25200 unique ordered paths, supplied labels and predicted integer classes were checked against class-directory labels and the returned frozen binding. Forty-five classes each have560test examples. All2025 confusion entries, all45 class support/correct/recall/F1 fields and summary/status references agree. OA is independently counted from predictions; macro-F1 uses exact Fraction precision/recall and their harmonic mean rather than floating point confusion denominators.

- Correct23718, errors1482; OA94.11904761904762%.
- Macro-F194.10529172930384%, exactly equal to the returned float; absolute tolerance1e-10.
- Returned summary, original checkpoint_reload metadata and initialization metadata agree with test/refit/SHA/selectedepoch95 and prior status references.

The existing independently audited DFAGCN raw test file was checked against its full file SHA and record seal. The same manifest, ordered paths and class names permit a complete descriptive paired comparison:

- Both models correct:21303.
- HGT correct, DF wrong (corrections):2415.
- HGT wrong, DF correct (regressions):503.
- Both wrong:979.
- Net corrected examples:2415−503=1912; totals25200. All directions and all45 classes are retained in validation.json, including per-class paired counts.

This audit does not independently load model/checkpoint bytes or original manifest/images; those were not uploaded. Initialization equality/reload0 remain references in returned original metadata. Different trained models, recipes/backbones and one seed do not establish module causality or seed stability. HGT-vs-GAViT paired corrections/regressions are not inferred from the net four-example accuracy difference; full GAViT predictions are not supplied in this package.
