import json
from pathlib import Path
p=Path("/home/featurize/work/nwpu_refit_test42_20260929_01")
print("QUEUE_EXIT",(p/"queue.exit").read_text().strip())
print((p/"queue_summary.json").read_text())
names=("swin","fixed","soft")
r={n:json.loads((p/(n+"_refit")/"run.json").read_text()) for n in names}
keys=("status","phase","model","architecture","seed","counts","final_epoch","checkpoint_sha256","refit_source","reload_max_logit_diff","graph_capture","elapsed_seconds","cuda_peak_allocated_bytes","cuda_peak_reserved_bytes")
print(json.dumps({n:{k:r[n][k] for k in keys} for n in names},indent=2))
print("STAGE_EXITS",{n+"_"+k:(p/(n+"_"+k)/"run.exit").read_text().strip() for n in names for k in ("refit","test")})
