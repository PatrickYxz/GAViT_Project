"""Bounded stdlib review case; no model, AID image, GPU or training involved."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from external_baselines.hgtnet import aid_entry
from tools import package_hgtnet_aid_smoke as delivery

REVIEW = Path(__file__).resolve().parent
evidence = REVIEW / 'status_partial_fixture'
evidence.mkdir(exist_ok=False)
checkpoint = b'not-a-torch-checkpoint; explicit stdlib review fixture'
(evidence / 'smoke_epoch1.pt').write_bytes(checkpoint)
summary = dict(
    schema=aid_entry.SCHEMA, completed=True, training_started=False,
    formal_training_started=False, counts=dict(aid_entry.COUNTS), optimizer_steps=120,
    input_scope='cpu_fixture', runtime={'device': 'cpu', 'dtype': 'float64'},
    manifest_sha256='bad', config_sha256='bad', weights_sha256='bad',
    source_identity=aid_entry._source_identity(),
    checkpoint=dict(sha256=hashlib.sha256(checkpoint).hexdigest(),
                    reload_max_abs_error=0.0, reload_tolerance=1e-6),
)
(evidence / 'summary.json').write_text(json.dumps(summary))
(evidence / 'progress.json').write_text(json.dumps(dict(stage='complete', completed=True)))
(evidence / 'run.exit').write_text('0\n')
report = aid_entry.status(evidence)
result = dict(
    schema='hgtnet-aid-independent-review-bounded-case-v1',
    no_ML_import=not any(name in sys.modules for name in ('torch', 'torchvision', 'numpy', 'PIL')),
    actual_GPU_or_training=False, status_state=report['state'], status_completed=report['completed'],
    missing_artifacts=[name for name in ('first_step.json','history.json','val_predictions.json',
                                       'binding.json','runtime.json','selected_inputs.json')
                       if not (evidence/name).exists()],
    malformed_summary_fields={key: summary[key] for key in
                              ('training_started','input_scope','runtime','manifest_sha256',
                               'weights_sha256','config_sha256')},
    whitelist_equal=aid_entry.SOURCE_FILES == delivery.SOURCE_FILES,
    source_member_count=len(delivery.SOURCE_FILES),
    current_source_hashes=aid_entry._source_identity(),
    source_AST_parse={name: ast.parse((ROOT/name).read_text()) is not None
                      for name in aid_entry.SOURCE_FILES if name.endswith('.py')},
)
(REVIEW / 'status_partial_repro.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
print(json.dumps(result, indent=2, sort_keys=True))
