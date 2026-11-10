# Focused independent review — 2026-11-04

Result: 95/95 focused checks passed. No unresolved data, provenance or quality finding within the assigned scope.

- All 16 historical main scores and their printed strings exactly match the existing multiseed CSV. Eight means, n−1 sample SDs, paired deltas and ROUND_HALF_UP displays independently recomputed; BEN fixed SD correctly displays 0.49 from the original printed scores, without double rounding.
- All 224 explicitly unknown measurement values remain null. Original logs have no fabricated references and are absent from this schema. Historical user-report provenance and unavailable-originals scope remain explicit.
- All old internal main/auxiliary metric bullet lines are byte-identical to the before copies; methods, diagnostic and efficiency sections are byte-identical. Existing seed42 comparison rows remain unchanged.
- HGTNet OA/F1, sample/correct/error counts, selected/final epoch history, parameters, net4 comparison and DFAGCN paired counts match the current analysis JSON and comparison CSV.
- Markdown and plaintext contain the same 835 numeric tokens, including retained link targets. Current validation SHA values match both final files.
- Fixed report lines 397 and 447 were re-read: lost 46/47 originals are no longer demanded, and the DFAGCN paragraph has an explicit model subject. Both formats agree.
- English-email validation now states only edited-in-this-update=false; no byte-unchanged assertion is made or independently established.
- AID preparation preserves the historical manifest identity, 4000/1000/pool5000/test5000 split, 240/60 test0 smoke boundary, and independent single-seed/future unified-protocol limits. These future tasks are not defects in this record supplement.

Verification: `python3 results/evidence/seed46_seed47_reconstruction_20261104/review/audit.py` returned exit0, 95/95, no findings. No build_record rerun, large suite, network, GPU, installation or Git operation. Writes confined to this review directory.

Final reviewed hashes:

- `results/multiseed_test_results.csv`: `639a9b45b07625d63c1254a32969637c984a996997248622a647f261d43c0988`
- `results/seed46_seed47_reconstruction_20261104.json`: `6bec50cec48c6f69e22100bf7cac441728645bcb0694d6213c548c83ce3ea12d`
- `docs/supervisor_progress_report_20261012.md`: `b3f444f7f0e6826934610a1f319cd35549da12907bfbbf94ccb425c1acf0631e`
- `docs/supervisor_progress_report_20261012_plaintext.txt`: `a260004b49a55f0e8e3c6c04fee4100c6f30f11e7ba0c4ef77bd3075ba3f5d10`
- `results/evidence/seed46_seed47_reconstruction_20261104/report_update_validation.json`: `bef7eca24cc547ccea783f78663e7454506dd4f0670b09bac171bbe9373f8786`
