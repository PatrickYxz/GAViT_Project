# HGTNet source-only delivery checks

`red.log` and `red_full.log` capture the missing builder API before implementation. `added_sources_red.log` captures acceptance of five untracked workspace sources/configs before the immutable-reuse guard was added. `green.log` records all 10 delivery tests passing.

The tests use a minimal repository fixture and real zipapps/subprocesses. They verify exact payload hashes and pinned external Swin-B provenance, build CLI, inspect/status/run argument forwarding, standard-library-only status under an ML import blocker, explicit-run behavior, immutable reuse and archive/workspace rejection paths. They do not invoke real training or a GPU.

The final production bundle is built by the parent task after all new runtime files exist. `validation.json` records this scope and the checked source hashes. No existing DFAGCN source, data, weights or prior package is modified.
