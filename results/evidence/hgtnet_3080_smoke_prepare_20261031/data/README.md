# HGTNet NWPU data adapter local evidence

- Implementation: `external_baselines/hgtnet/nwpu_data.py`.
- RED: `red.log`, 11 failures because the new data module/API was absent, before production code was written.
- GREEN: `green.log`, 11/11 tests passed using real local files, ZIPs, JPEG decodes, and Torch tensors.
- Existing helper regression: `helper_regression.log`, 11/11 tests passed. The helper `tools/run_dfagcn_nwpu_smoke.py` was reused without modification; its CLI/training entry point was not called.

Command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE /private/tmp/gavit-dfagcn-cpu-20261027/bin/python -m unittest discover -s tests -p test_hgtnet_nwpu_data.py -v`

The manifest fixture contains all 31,500 rows with 45 classes and 112/28/560 records per class. Its digest override is a Python test interface only. Fixed selection is 8 train and 2 validation records per class, preserving sorted selected order; test count is zero. A separate Python process prohibits Torch, torchvision, NumPy, and PIL imports while preparing this manifest, validating files, and restoring a ZIP.

Image tests read generated RGB and grayscale JPEG fixtures. They check deterministic RGB conversion, bilinear resize to 224, exact ImageNet normalization, finite float32 tensors, scalar long labels, preservation of caller record order, copied records, and image SHA validation before construction and each access. Invalid labels, inconsistent class mappings, test records, duplicate/unsafe paths, ZIP traversal/symlinks/ambiguous members, changed ZIP/image hashes, and output collisions are rejected.

These are local fixture checks. They do not establish real NWPU image availability, real-data accuracy, or any RTX3080 execution result.
