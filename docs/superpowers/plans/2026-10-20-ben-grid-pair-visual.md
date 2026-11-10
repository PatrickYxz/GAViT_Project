# BEN seed42 trained 3×3 / 4×4 visualization

Spec: main-repository docs/run_ben_grid3_seed42_20261018.md, section 训练后的可视化交付; approved by the user on 2026-10-20.

Reuse this research worktree; preserve all existing dirty files. No training, dependency installation, full-test rerun, commit or push.

1. Write and run failing tests for node-dependent topology validation, all-label paired prediction outcomes, artifact identity and lossless rendering. Implement one external capture/render tool with lazy tensor imports.
2. Validate the immutable K9 source, the completed training summary and both checkpoint hashes before loading weights. Reuse all four original-input PNGs from persistent work/pair44_feedback_k_7ehbne, with fixed SHA and validation-row labels. Capture both trained models on the identical normalized tensor; reconstruct pooling and feedback and verify capture/restoration logits within 1e-5. Save tensors, all 19 probabilities, actual edges, input/model identities and source summary.
3. Render a self-contained HTML and per-case SVG, with separate spatial/feature overlays and an optional pooling-weight overlay. Package a single-line Featurize installer/run command, preserving any existing output. Run local tests, payload/fixture checks and visual layout QA, then record prepared-only status and the precise server completion conditions.

Interface check: capture JSON feeds renderer; both use validated assignments/edges and all 19 class outcomes. PNG is the losslessly saved original-reader Resize224 RGB input, not the old JPEG gallery. The fixed four images are a historical diagnostic validation subset; no population AP/F1 or preselected positive outcome is inferred.

Ruling: use persistent original-input PNGs rather than re-extract the entire BEN archive. Verify hashes and val CSV row/label identity; normalize once and share the same tensor between models. Default CPU inference avoids requiring a paid GPU; CUDA may be explicitly selected with its UUID recorded. The server performs model/tensor checks because local torch/timm/PyG are unavailable.

Progress: preparation complete; 9 new tests and 11 existing K9 regression tests pass. Full unittest runs 199 tests with 21 pre-existing missing-dependency errors (8 torch, 13 pytest), all retained in the handoff evidence directory. Independent reviewer found no Critical/Important issues; this progress text addresses the minor stale-status finding. Installer/payload and static SVG fixtures checked; browser file navigation was blocked, so HTML interaction is not certified. Paired trained-model capture still awaits the user's Featurize execution and returned ZIP.
