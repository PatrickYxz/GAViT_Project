# HGTNet RTX3080 NWPU smoke delivery — 2026-10-31

Delivered an offline source zipapp and upload/run instructions. **Target RTX3080 and real NWPU execution are still pending.** Local JPEG fixtures and mocked CUDA boundaries do not establish those outcomes.

## Final artifacts

- `output/hgtnet_nwpu_smoke_3080_20261031.pyz`, 52,238 bytes, SHA256 `e798372589f83359f6d33d3c78d48ea1a82a760310b23f26aa0d6e689835ce78`.
- 17 payload files, bundle ID `42fcdedc83d138708db13d1501b11b06a4e384f506acbc46eb5b9c4756ade29b`; bootstrap/manifest/source bytes checked. No weights, LIBLINEAR or installer inside.
- Separate official Swin-B candidate: 351,654,207 bytes, SHA256 `68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893`; preserved under ignored `checkpoints/hgtnet_pretrained_20261031/`.
- User instructions: `output/hgtnet_nwpu_smoke_3080_README_20261031.md`.

## Engineering scope

The prospective GPU run uses full 74,619,858-parameter independent HGTNet, frozen seed42 and architecture, batch2, one epoch/180 steps, Adam1e-4/CE, actual RGB224 ImageNet normalization, no augmentation/AMP/TF32, train360/val90/test0. It stages only selected train/validation images and never calls test. CPU fixture uses four generated training JPEGs and two generated validation JPEGs with an explicit fixture scope, not real NWPU.

`inspect` is stdlib-only and checks version metadata/full weights SHA/path/capacity and the current idle physical GPU. Presence of ZIP permits explicit pending recovery; the run verifies complete ZIP SHA and each staged image before decoding. Successful ZIP completion changes the final receipt from pending to verified. GPU identity uses the current inspection UUID rather than old rental IDs. The run separately verifies actual imports/CUDA/cuDNN/settings.

Real-image smoke checkpoints have a separate metadata schema from previous synthetic checks. Validation predictions, full confusion/per-class metrics, train history, first successful optimizer receipt and source/runtime/manifest/config/weights identities are saved. Status requires both summary completion and exit0. No resume, automatic retry, hidden batch reduction or source modification occurs.

## Evidence

- `data/`, `resources/`, `delivery/`: meaningful missing-API and regression RED→GREEN logs and adapter/builder tests.
- `integration/*red.log` / `*green.log`: root engine and runner tests; first-step diagnostics require an updated representative in every positive-gradient group. `review/data_receipt_red.log`→`data_receipt_green.log` covers actual CPU training/reload through a patched CUDA boundary and correct verified recovery receipt.
- `integration/related_suite.json/.log`: **285/285**, failures0/errors0/skipped0; 137HGTNet +148DFAGCN. Tests execute12.883s, runner wall13.7997s. This is the relevant regression scope, not the legacy full-repository dependency suite.
- `full_model_cpu_fixture_02/summary.json`: final-source full HGTNet, actual candidate load318 tensors, real generated JPEG transform, four train/two validation, two optimizer steps, finite gradients in all five groups, actual save and independent-model numeric reload error0. Verification wall3.879s is CPU fixture only. Checkpoint/images preserved locally under scoped gitignore.
- `full_model_cpu_fixture_01/`: pre-receipt-fix local run retained as history; final audit uses02.
- `integration/bundle_inspect_guard.json`: actual final production pyz inspected while all ML imports are forbidden; no Torch/vision/NumPy/PIL imported and no runs created. Local blockers for missing server inputs/GPU/runtime are expected. `actual_bundle_status.json` is not_started.
- `review/`: fresh independent source review and three actual CPU failure probes. No unresolved Critical/Important finding; note that review-only counts predate final consolidated tests. Root final metadata regression is kept separately.
- `integration/validation.json`: **121/121** stdlib audits pass: final package/repository identity, full fixture receipts, current weights and protected artifacts. 52 historical metrics/sources/DF packages and all9 previous HGTNet Python files +config remain exact bytes.
- `draft_bundle/`: our own undelivered local draft before the verified-recovery receipt correction, preserved without deletion. It must not be used on the target server. Final output has the same clean filename with new bytes.
- `integration/commands_and_exits.json`: actual completed execution commands/exit codes; no target execution claimed.

No source/metric/checkpoint overwrite, remote action, new rental cost, formal experiment, Git commit or push occurred. Latest details are in the research diary; final numerical/GPU acceptance requires the user's actual server output.
