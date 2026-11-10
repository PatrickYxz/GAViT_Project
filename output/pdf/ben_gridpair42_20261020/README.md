# BEN trained fixed3x3 / fixed4x4, seed42

- `ben_gridpair42_cases.pdf`: page1 actual feature edges and all threshold changes; page2 within-region pooling relative to uniform.
- `graphs_predictions_300dpi.png` and `pooling_300dpi.png`: 2126×3459 exports.
- `comparison.html`: byte-identical returned self-contained page; includes spatial/feature/pooling toggles and all19 class probabilities. Browser interaction was not retested locally.
- `build_figure.py`, `verify_figure.py`, `figure_manifest.json`, `export_qa.json`: reproducibility and verification.

Source: `results/evidence/ben_gridpair42_visual_return_20261020/server_return`. Original RGB pixels are preserved; only vector overlays added. All4 predetermined historical validation cases retained; this is not a population sample. Actual model data, not the previous synthetic layout fixture.

At threshold0.5, 4x4 corrects3 label decisions and adds1 missed label. The mixed case row28450 is retained in both directions. Context: historical K16 test mAP72.0% versus new K9 71.0753%, with separate-training-batch limitations. Detailed Chinese analysis: `docs/ben_gridpair42_visual_analysis_20261020.md`.
