# HGTNet AID input inspector: independent review

Date: 2026-11-04. Scope: Task 1 and inspector/package portions of Task 3 in `docs/superpowers/plans/2026-11-04-hgtnet-aid-inputs.md`, against `docs/external_baselines_20261027/hgtnet_aid_preparation_20261104.md`. This review does not accept a future AID trainer or the separate seed46/47 reconstruction.

**Spec compliance: PASS for the local inspector and delivery scope. Code quality: APPROVED. No actionable defects found.** No reviewed source, tests, package, configuration, Git state, or existing evidence was modified. Only this review record was added.

## Reviewed identities

- Source `tools/inspect_hgtnet_aid_inputs.py`: 18930 bytes; SHA256 `4e3b66b91f115d99d01e39fe168e8cf5dd18503ec8593e45c39dbccfcd0eb7b4`.
- Actual package `output/hgtnet_aid_inputs_20261104.pyz`: 6520 bytes; SHA256 `1cd2ab967cdbc386b294b397c6804bf06c5796c01332c709e44646f98d03f036`.
- Fresh read-only verification: 11/11 checks passed. These cover both frozen hashes, the two expected package members, exact source/payload equality, package source identity, source/test/package/README receipt identities, standard-library import roots, and the preserved original NWPU formal package SHA.
- Package members are exactly `__main__.py` and `source_identity.json`; no model tensors or third-party payloads are included.

## Compliance assessment

- The public CLI fixes the original manifest SHA and full candidate Swin-B SHA; the test-only expected-manifest-SHA argument is not exposed as a CLI override. Mismatched or missing manifests cannot reach image checks or asset verification.
- Schema2, AID taxonomy30, frozen group-v2 metadata, integer seeds, records10000 and train4000/val1000/test5000 are checked. Pool5000 is correctly derived from train+val; the original manifest generator does not declare a separate pool count.
- The original `scene_classification/data.py` was inspected without importing it. The duplicate report fields/order, floor-per-class arithmetic, class naming normalization, and selected-record ordering are compatible. The inspector deliberately trusts the fixed original manifest SHA rather than regenerating split assignments; its output discloses `split_assignment_recomputed:false`.
- Nonuniform class counts are supported. Every class is checked against its own floor-per-class pool/validation counts, then exactly eight train and two validation records per class are selected in path order. Only those 300 file byte streams are opened; test records receive metadata validation. Existing tests establish successful verification with only the selected files present and reject test image access.
- Unsafe metadata paths, duplicate paths, invalid labels/hash fields, inconsistent duplicate reports, cross-class/cross-split pixel groups, and selected-file byte changes are rejected. Same-class same-split duplicate groups are retained. The descriptor-relative `O_NOFOLLOW` chain rejects symlink ancestors and final files without a symlink-following fallback.
- Weights are hashed in full with bounded chunks and are labelled as an initialization candidate. There is no reuse path for a trained NWPU checkpoint.
- GPU inventory runs only the fixed `nvidia-smi` query with a 10-second timeout. Dependency versions use distribution metadata; no ML package, image decoder, model, or CUDA compute import/run is present. Disk observations identify the actual measured ancestor for a missing path.
- Inventory status and documentation explicitly separate input asset verification from smoke implementation and training readiness. No unknown AID ZIP identity, GPU UUID, throughput, price, or successful model run is invented.

## Existing verification evidence assessed

- `implementation/green_13_tests_final.log` reports 13 tests, zero failures/errors/skips; test source was reviewed and matches its delivery identity. Cases cover metadata rejection, no test image reads, nonuniform classes, duplicate groups, missing/changed files, symlinks, fixed CLI identity, full weight hashing, bounded GPU inventory, and read-only/std-library operation.
- `delivery/validation.json` reports 15/15 delivery checks. `delivery/guarded_bundle_inventory.json` records execution of the actual final package under non-standard-library import and file creation/write/delete/rename guards, exit0 with missing assets, no training entry, and no stderr.
- The earlier failed guard attempt remains recorded and is not presented as a pass. The final guarded attempt and payload identities refer to the reviewed package.
- No unit tests, ML engineering suite, actual zipapp inventory run, model execution, installation, network operation, training, or GPU computation was repeated for this review. Fresh checks read bytes and inspect package/AST identities only.

## Boundaries and handoff

The original manifest bytes, real AID image bytes, and target GPU were unavailable locally. Synthetic metadata/file fixtures and missing-asset execution do not prove their availability or correctness on the target instance. Even a later `INPUT_ASSETS_VERIFIED_SMOKE_PREPARATION_PENDING` result does not demonstrate an implemented AID data binding, forward/backward pass, checkpoint save/reload, GPU smoke, formal training, or paper results. No changes are requested for this inspector; proceed with the documented read-only target-instance inventory and retain its complete JSON before preparing the next stage.
