# Completed external results: analysis and thesis integration

This continues the user's already authorised external-baseline workflow after the verified HGTNet AID result. The requested minimum matrix is HGTNet on NWPU/AID and DFAGCN on NWPU, as stated in the archived supervisor email. All three single-seed independent measurements now exist. This stage uses those artifacts; it authorises no new training or test forwards.

The existing NWPU analysis is the reference workflow. The remaining deliverable is the corresponding AID analysis, a clearly labelled supplementary comparison across the three requested method/dataset cells, updated copyable supervisor material and an English thesis section. The shared split does not make the different backbones, preprocessing, optimisers and calibration budgets a unified recipe. External multi-seed/unified-protocol work remains a separate open requirement.

## Approach

Use the returned 5000 test predictions, original manifest and 100-calibration/96-refit history to reproduce metrics and derive complete class/error/curve outputs. Reuse archived Swin/GAViT seed42 summary values for numerical comparison. Pair predictions only if original ordered predictions can be located and verified; aggregate net-correct differences do not reveal corrected/introduced error counts. Do not retrain to recover predictions.

The alternative of adding DFAGCN AID expands the requested matrix and consumes further GPU work; it is unnecessary here. Replacing measurements with paper scores or merging single-seed external rows into core multi-seed statistics would invalidate provenance and is excluded.

## Deliverables and scope

- A deterministic offline AID analysis entry, small meaningful input/metric/selection checks, complete numeric outputs and publication-style PNG/SVG figures.
- Same-seed AID Swin/GAViT/HGTNet comparison with exact differences, parameter counts and source/recipe distinctions, plus a cross-dataset synthesis of the three completed external cells.
- Updated dated supplemental supervisor report/email and copyable plaintext; update current report status without fabricating absent F1 or original records for seeds46/47.
- A supplementary external-comparison chapter in the existing English thesis, with stale introduction/setup wording corrected. Preserve the thesis as a partial draft; this does not claim all results, discussion or front matter are complete.
- Source/metric checks, thesis compilation and visual inspection of newly generated figures and changed PDF pages.

Raw artifacts, all model/training sources and packages, frozen splits, old result values and historical worktree files remain unchanged. No Git staging, commit, push or external email is authorised. Preserve earlier document bytes in a dated evidence/provenance directory before editing.
