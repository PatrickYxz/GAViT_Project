"""Record source identity and actual parameter counts for this local run."""
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from external_baselines.hgtnet.graph import DepthwiseProjection, GraphBlock

evidence = Path(__file__).resolve().parent
parts = {
    "transition1_512_to_256": DepthwiseProjection(512, 256, padding=1),
    "block_C256": GraphBlock(256, 20),
    "transition2_256_to_512": DepthwiseProjection(256, 512, padding=0),
    "block_C512": GraphBlock(512, 20),
}
counts = {name: sum(p.numel() for p in module.parameters()) for name, module in parts.items()}
summary = {
    "schema": "hgtnet-independent-graph-local-v1",
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "source_sha256": {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("external_baselines/hgtnet/graph.py", "tests/test_hgtnet_graph.py")
    },
    "test_command": "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE /private/tmp/gavit-dfagcn-cpu-20261027/bin/python -m unittest discover -s tests -p test_hgtnet_graph.py -v",
    "red": {"exit_code": 1, "tests_run": 27, "subtest_failures": 79, "reason": "Missing graph implementation raised intended AssertionError", "log": "red.log"},
    "green": {"exit_code": 0, "tests_run": 27, "failures": 0, "errors": 0, "log": "green.log"},
    "native_openmp": {"unit_test_failure_observed": False, "summary_stdin_helper_failure_observed": True, "failure_log": "parameter_count_native_omp_failure.log"},
    "parameter_counts": counts,
    "full_graph_parameter_count": counts["transition1_512_to_256"] + 6 * counts["block_C256"] + counts["transition2_256_to_512"] + 2 * counts["block_C512"],
    "boundaries": [
        "Real CPU module tests, not dataset evaluation or GPU validation",
        "max-relative is default; relative_mean is only an operator ambiguity probe",
        "exclude_self=True requires N>=k+1; explicit False permits k==N",
        "Finite fp32 distance overflow avoided with float64; overflow beyond float64 range rejected",
        "Stable index tie-breaking is deterministic but exact ties need not be permutation equivariant",
        "Parent integration owns complete-model and full relevant-suite checks",
    ],
}
(evidence / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"parameter_counts": counts, "full_graph_parameter_count": summary["full_graph_parameter_count"]}, indent=2))
