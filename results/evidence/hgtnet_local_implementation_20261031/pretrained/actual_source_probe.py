"""Read-only verification of the acquired torchvision candidate's complete state."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import torch
import torchvision
from external_baselines.hgtnet.pretrained import _validate_official_source
path=ROOT/'checkpoints/hgtnet_pretrained_20261031/swin_b-68c6b09e.pth'
start=time.monotonic()
with path.open('rb') as stream:
    sha=hashlib.file_digest(stream,'sha256').hexdigest()
    assert sha=='68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893'
    stream.seek(0)
    state=torch.load(stream,map_location='cpu',weights_only=True)
validation=_validate_official_source(state)
report={'status':'actual_official_torchvision_candidate_source_verified',
        'model_loaded':False,'is_original_hgtnet_weights':False,
        'source_url':'https://download.pytorch.org/models/swin_b-68c6b09e.pth',
        'path':str(path),'sha256':sha,'bytes':path.stat().st_size,
        'validation':validation,'software':{'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__},
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start}
(Path(__file__).resolve().parent/'actual_source_probe.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
