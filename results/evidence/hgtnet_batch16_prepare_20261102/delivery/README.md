# HGTNet batch16 registered delivery entry

The existing source-only builder now accepts
`build_bundle(output_path, *, target_gpu='3080', runner_member=RUNNER)`.
`runner_member` accepts only the original `tools/run_hgtnet_nwpu_smoke.py`
or `external_baselines/hgtnet/batch_probe.py`. The original entry remains
the default and is still required in the payload for shared helper imports.
The selected entry must exist in the scoped source payload before building,
and the immutable installer checks both registration and payload membership.

The build CLI accepts `--runner-member` with those same two choices. The
existing bootstrap dispatches the selected checked manifest entry.

`red.log` / `red.json` record two failures before the production change:
the API lacked `runner_member` and the CLI lacked `--runner-member`.
`green.log` / `green.json` record 23 passing stdlib tests: 8 new delivery
cases, 10 original delivery cases, and 5 existing 3090 delivery cases.
The new cases use minimal temporary repositories and actual zipapp
subprocesses to verify inspect/status routing with ML imports forbidden,
target propagation, single probe membership, retained shared runner,
missing/unregistered/typo entry rejection, and immutable source reuse.

These checks are source-delivery evidence. They do not execute the numerical
probe, access a GPU, validate real NWPU images, install dependencies, or
create the production probe zipapp. Root integration owns the broader CPU
suite and builds the production package only after the probe source freezes.
