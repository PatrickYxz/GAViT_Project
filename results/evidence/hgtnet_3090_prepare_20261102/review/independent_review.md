# Independent scoped review — RTX3090 extension

Reviewed 2026-11-02. Scope: hardware compatibility extension from explicit RTX3080 to explicit RTX3090, using the old delivered zipapp as the source baseline. No actual GPU, CUDA, network, model training, or Git mutations were used.

No unresolved Critical/Important/Minor findings in the reviewed production changes.

- One minor direct-source CLI issue was reproduced and reported during review: `--target-gpu 3090` selected the3090 gate while defaulting scratch to an3080 namespace when no delivery environment was set. The implementer fixed this with `_TargetParser`, deriving scratch from the parsed target. Independent re-read and the added regression passed. The bundle path was already correct because bootstrap binds the target.
- The old3080 zipapp SHA remains `e798372589f83359f6d33d3c78d48ea1a82a760310b23f26aa0d6e689835ce78`. Its current-source payload comparison changes only `resources.py` and the runner; 14 other locally present payloads, including model/core, data, engine, pretrained loader, and fixed configuration, match exactly. Bootstrap changes are restricted to explicit target validation, manifest binding, and environment propagation.
- All14 new target tests passed independently, including exact name/24576MiB/UUID selection, ambiguity and invalid-type rejection, unchanged software and precision gates, mocked CUDA-visible checks, source scratch namespace, and actual bootstrap propagation. The existing15 resource tests and10 delivery tests also passed:39 targeted tests total,0 failures/errors/skips.
- `independent_probe.py` built the real current source zipapp `independent_target3090_final.pyz`. In separate processes with ML imports forbidden and an explicit NVIDIA-query fixture matching the supplied3090 inventory, inspect selected `GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1`, exact name and24576MiB, despite an inherited3080 environment. Scratch uses the3090 namespace. Status reports `not_started`; no runs directory was created. This proves local bootstrap/source wiring only; it does not prove real CUDA or training success.
- Default3080 resource APIs and delivery build remain compatible. New runner metadata carries the explicit target; no batch, seed, data protocol, optimizer, model, pretrained-weight metadata, numeric precision, or automatic-run scope change was found.

Evidence: `audit.json`, `inspect_actual_source_bundle.json`, `status_actual_source_bundle.json`, and the reproducible probe. The first `independent_target3090.pyz` is a retained local review draft made before the direct-source scratch fix; it is not the final reviewed package or user deliverable.

Commands used `/private/tmp/gavit-dfagcn-cpu-20261027/bin/python -m unittest discover -s tests -p` with patterns `test_hgtnet_3090*.py`, `test_hgtnet_resources.py`, and `test_hgtnet_nwpu_delivery.py`; thread limits were `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE`.
