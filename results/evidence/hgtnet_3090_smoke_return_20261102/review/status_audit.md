# Independent HGTNet3090 smoke status acceptance audit

Result: **PASS**, 70/70 checks.

Accepted state: `SERVER_REPORTED_HGTNET_REAL_NWPU_SMOKE_COMPLETE_SUMMARY_IDENTITIES_MATCH`. The actual fix1 zipapp SHA and every payload digest match. Its canonical config digest, fixed manifest digest and candidate Swin-B weight provenance match the reported identities. Source status accepts complete only with summary completed=true and run.exit=0; completion follows all selected training/validation, finite-update, checkpoint save/reload and final identity checks. No rerun is needed.

Reported validation OA 36.666666666666664% is arithmetically compatible with 33/90 correct and 57 errors. Reported macro-F1 is 29.33525733525733%; aggregate OA cannot independently establish macro-F1. The smoke uses train 360 / val 90 / test 0, batch 2 and 180 steps, best/final epoch 1. Checkpoint SHA de78efa372cf79862859efaf13b34ff4e82c25fceb58af9a0caf84c43e7232e5 and zero reload error are server-reported values.

Wall 58.25750263300006 seconds starts after stdlib preflight and includes imports, possible recovery, initialization, one training epoch, validation and reload. It does not establish training/inference throughput or billing duration. Peak allocated 1538072576 B = 1466.8203125 MiB = 1.4324417114257812 GiB; reserved 1688207360 B = 1610 MiB = 1.572265625 GiB. These are PyTorch CUDA allocator peaks over the whole smoke including update diagnostics and reload, not whole-process NVML usage or formal-batch estimates.

Evidence origin: locally transcribed user status, not downloaded server-original summary, exit, runtime, history, predictions or checkpoint bytes. The status omits complete runtime fields; delivered source enforces them on its normal success path. Scope remains an engineering smoke and does not enter formal NWPU performance comparisons. No GPU, network, new numerical test, dependency install or Git action occurred in this audit.
