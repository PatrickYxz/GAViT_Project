# HGTNet seed42 returned-result figures

Four figure groups, each provided as 300 dpi PNG, editable-text SVG and vector PDF:

- hgtnet_seed42_training_history: all 100 calibration and 95 fresh-refit epochs; train/val loss, online train accuracy / val OA / val macro-F1, selected 95 and final 100 both retained. Refit panels contain training only. No smoothing or epoch removal.
- hgtnet_seed42_all_class_performance: all 45 classes, ascending recall, 560 samples per class. Recall bars and F1 diamonds share the 0-100 metric domain; recall numbers use a separate fixed right column to avoid covering F1 markers. Overall OA is shown as a reference.
- hgtnet_seed42_confusion_matrix: all 45 x 45 cells with the diagonal, original class order, row-normalized percent 0-100. Rows=true, columns=predicted. No class or confusion cell is removed.
- hgtnet_seed42_top10_bidirectional_confusions: ten unordered pairs selected only by largest total off-diagonal counts. Both directions including zeros retained; full matrix and all errors remain in raw analysis. Pairs with nonzero errors outside the top 10 are omitted from this view only.

The plot script independently reconstructs all 25,200 true/predicted labels, 2,025 confusion cells and all 45 recall/F1 scores, checking them against returned test metrics. OA 94.11904761904762%, macro-F1 94.10529172930384%, 23,718 correct / 1,482 errors, 560 samples/class. Individual figure data/selection are recorded in plot_data.json; file hashes and source hashes are in plot_manifest.json.

Reproduction uses Python 3.12.13, Matplotlib 3.11.2 and NumPy 2.5.4 from the existing isolated plotdeps directory. No installation or modification of the project/model/Torch environment occurred. Command:

```sh
MPLCONFIGDIR=/private/tmp/hgtnet-mpl-20261104 PYTHONPATH=/private/tmp/dfagcn-plotdeps-20261031 /private/tmp/gavit-hgtnet-probe-cpu-20261102/bin/python results/evidence/hgtnet_results_return_20261104/figures/plot_diagnostics.py
```

The script intentionally refuses replacing existing figure files; preserve existing outputs before re-running. The initial draft and its manifest/script are retained under draft_01; visual review found overlap between per-class recall numbers and F1 markers. Final figures move all recall numbers into a separate column and were all rerendered.

The PDF skill authoring marker executed successfully exactly once with four expected PDF outputs. Each final PDF has one page and was rendered with Poppler 26.05.0 at maximum dimension 1900 pixels, then visually inspected: no clipping, missing labels, overlapping numbers/legends, or missing counts. Rendered QA images are under rendered/. The PDF and SVG versions retain vector/text content; PNGs are 300 dpi. visual_validation.json records the verification.

Training accuracy is online while model weights change; validation uses epoch-end weights on distinct samples. Refit started from the same initial weights with the preselected 95-epoch budget. Curve or confusion patterns are descriptive single-model/single-seed diagnostics; they do not identify a graph module causal effect or original-paper equivalence.
