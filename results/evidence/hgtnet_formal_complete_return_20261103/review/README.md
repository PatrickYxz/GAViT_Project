# Independent formal completion marker review

The actual delivered v1 zipapp is byte-verified, and its engine/entry are compared with current local sources. The normal path requires fresh equal-initial-state refit through the selected epoch, model state and numeric reload checks, final test order/count and metrics/prediction-file persistence, final source/recipe/manifest/weights/pool/test byte rechecks, then summary+exit0+marker before complete progress and COMPLETE.

The returned68930 updates equal100×315 calibration updates plus95×394 refit updates. Returned checkpoint/model-state SHA values are format-checked references, not fetched model bytes. The fragment's6300/112.182047135 train throughput is56.1587images/s for the last refit training epoch only; no full epoch wall or entire-run duration can be reconstructed from its truncated beginning.

[independent_audit.py](independent_audit.py) and [validation.json](validation.json) retain current checks and derived arithmetic. If the parent raw archive is available, the script also checks its three complete JSON events and final path without parsing the truncated first row as a full event.

The initial marker-only review accepted server-reported normal-program completion while awaiting final test OA/macro-F1, total measured wall, peak allocated/reserved memory, summary identities and returned exit value. The subsequently received status is audited below. A trainingaccuracy100% line is not a test result. No source/config changes, numerical rerun, GPU action, Git mutation or direct remote retrieval was performed by this review.

## Status subsequently received and independently checked

[status_audit.py](status_audit.py) / [status_audit.json](status_audit.json):39/39 checks pass. The original marker audit was rerun against the now-archived raw output and passed22/22.

Status reports complete/exit0, realNWPU25200test records, the frozen config/recipe/manifest/candidate identities, current3090 UUID, all74619858 parameters trainable,68930 updates, equal calibration/refit initialSHA, selected/refit95 with calibrationfinal100, finalrefit/reload SHA equality and numericerror0. Hashes are references; remote300MB checkpoint bytes were not fetched.

Reported testOA94.11904761904762% implies23718/25200 correct and1482errors; validation93.0952380952381% implies1173/1260. Reported testmacroF194.10529172930384% is not independently recomputed from predictions. Wholeattemptwall24032.617371428s=6h40m32.617s; allocated4.40945673GiB/reserved4.783203125GiB, with summary's stated current-engine memory/attempt timing scopes, not billing.

Read-only baseline CSV comparison gives +0.18650794pp/47correct samples versusSwin-T, +0.015873016pp/4samples versusGAViTfixed4x4, +0.25pp/63samples versuslocalsoft, +7.58730159pp/1912samples versusDFAGCN. The four-sample single-seed difference does not establish stable superiority, statistical significance or individual graph-module causality; backbone, pretraining, recipe/budget andGPU differ. The comparison CSV was not changed by this reviewer.
