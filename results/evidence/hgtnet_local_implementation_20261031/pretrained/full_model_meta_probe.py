"""Schema-only full actual HGTNet/official-source mapping audit; no value load."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import torch
from torchvision.models import swin_b
from external_baselines.hgtnet.config import load_config, config_sha256
from external_baselines.hgtnet.model import HGTNet
from external_baselines.hgtnet.pretrained import audit_pretrained_state
config=load_config(ROOT/'configs/hgtnet_independent_v0.json')
with torch.device('meta'):
    target=HGTNet(config)
    source=swin_b(weights=None)
report=audit_pretrained_state(target,source.state_dict())
assert report['mode']=='schema_only'
assert report['can_apply'] is False
assert 'loaded' not in report
assert len(report['compatible'])==318
for field in ['missing','unexpected','shape_mismatch','dtype_mismatch','nonfinite','invalid']:
    assert not report[field], (field,report[field])
assert len(report['source_intentionally_skipped'])==35
assert 'stages.2.downsample.projection.weight' in report['intentionally_initialized'] or 'stages.2.downsample.conv.weight' in report['intentionally_initialized']
assert target.pretraining_loaded is False
report.update(status='actual_full_hgtnet_and_official_source_meta_schema_audited',
              config_sha256=config_sha256(config),
              pretraining_loaded=target.pretraining_loaded,
              timestamp_utc=datetime.now(timezone.utc).isoformat(),
              interpretation='Full actual target/source structures only; no CPU values copied and no ImageNet-loaded claim.',
              source_files_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT/'external_baselines/hgtnet/pretrained.py',ROOT/'external_baselines/hgtnet/model.py',ROOT/'external_baselines/hgtnet/attention.py',ROOT/'external_baselines/hgtnet/graph.py',ROOT/'external_baselines/hgtnet/config.py']})
output=Path(__file__).resolve().parent/'full_model_meta_probe.json'
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key:report[key] for key in ['status','mode','can_apply','pretraining_loaded','coverage','config_sha256']},indent=2))
