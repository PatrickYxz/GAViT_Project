# Independent history / source / runtime audit

**498/498 checks passed** with Python standard library only. [audit.py](audit.py) reads the user archive directly without extracting or executing supplied content; [audit.log](audit.log) and [validation.json](validation.json) retain the complete results.

- The gzip archive is544296B with SHA4fc4d51cec3aac5900ac94cdde919818ba58cd0f76add625867623f54d052477. Exactly the seven expected regular JSON members are present. Their raw file hashes are retained. The `.tar.tar` filename does not affect gzip/tar detection.
- History and summary history arrays match exactly:100 calibration epochs plus95 fresh-refit epochs,315/394updates per epoch, cumulative68930. All195 records contain both actual epoch wall and precommit duration; counts, online training-accuracy integer arithmetic and sample/time throughput agree.
- All100 validation confusion matrices,45class supports (28each), per-class accuracy/F1 and global accuracy/macro-F1 were recomputed. Maximum floating difference is5.684341886080802e-14. This is aggregate-matrix arithmetic validation, not recovery of each epoch's individual validation predictions.
- Epoch95 is the unique maximum validation correct count:1173/1260,OA93.0952380952381%,macro-F193.03452094979133%. The actual final epoch100 is1165/1260,OA92.46031746031746%,macro-F192.42454111106593%. Best−final is8correct/0.6349206349OA pp/0.6099798387macro-F1 pp. Validation loss is0.3547356235 atbest95 versus0.3699770268 atfinal100. The observed difference is descriptive and does not identify its cause.
- Initialization metadata, summary initialization hashes and refit initialization hashes are equal. The returned checkpoint-reload JSON equals summary, reports numerical error0, and references the same final-refit SHA as test.json. Trained model tensor bytes are absent, so numerical inference/state hashing is not independently repeated locally.
- Every binding source identity equals the actual delivered v1 payload SHA, including the raw formal recipe. The actual zipapp SHA isb2bc9077cf37e75c83fbb72e7c4047d2fb9ede04db8fb4e4a56344500ff572e1. Config canonicalSHA, recipe canonicalSHA, candidate-referenceSHA and manifest-referenceSHA agree. The complete original manifest, raw NWPU images and original candidate/model tensors were not part of this returned archive.
- Runtime file equals summary runtime and confirmsPython3.11.8/Torch2.2.2+cu121/torchvision0.17.2+cu121/NumPy1.26.4/Pillow10.3.0/CUDA12.1/cuDNN8902. Actualflags areFP32/seed42/8CPUthreads/noAMP/noTF32/cudnn deterministic true,3090UUID4f625a02...3cb1. No new global deterministic-algorithm or GPU-bitwise-resume claim is made.
- Ordered test labels/paths equal binding test identities; val/test paths are disjoint,45classes have28/560records respectively. Test raw SHA exactly matches the previous reported869986974e4801f2234fd2fa3bace383af2f092fd20359fbe1a252d4af7e37ae. Root/parallel prediction analysis separately reconstructs the25200prediction metrics.

Timing sums:

- Calibration train8674.566730196s,validation897.982860996s,complete epoch walls11778.342573141s.
- Refit train10066.053316566s,complete epoch walls11862.193852847s.
- Test180.762088437s. All epoch walls+test23821.298514425s; engine other78.627519628s; wrapper beyond engine132.691337375s.
- Weighted training-stage throughput is58.10088454images/s for calibration and59.45726504images/s for refit. It includes reading/augmentation/transfer/checks/updates, and does not measure pure GPU inference. Mean complete epoch wall is117.78342573s/124.86519845s for calibration/refit. No rental bill or idle-time cost is inferred.

The archive supplies actual saved metadata/test bytes, so prior “metadata not downloaded” limits can now be narrowed. It still does not contain model checkpoints, original image pixels, full original manifest, per-epoch validation predictions, or rawrun.exit/completion.marker files. Those remain references or prior reported observations.
