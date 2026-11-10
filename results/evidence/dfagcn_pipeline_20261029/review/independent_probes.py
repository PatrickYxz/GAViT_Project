"""Read-only checks of the retained engineering run, plus rejection probes."""
import json
import sys
from pathlib import Path

import torch

from external_baselines.dfagcn.cache import FeatureCache
from external_baselines.dfagcn.model import DFAGCNBranches
from external_baselines.dfagcn.pipeline import FORMAL_MANIFESTS, _check_mode, evaluate, source_sha256

root=Path.cwd()
run=root/'results/evidence/dfagcn_pipeline_20261029'/('smoke_run_02' if len(sys.argv)<2 else sys.argv[1])
refit_record=json.loads((run/'refit/refit.json').read_text())
assert refit_record['source_sha256']==source_sha256(), 'probe must use a current-source refit artifact'
recipe=json.loads((run/'recipe.json').read_text())
torch.manual_seed(recipe['seed'])
initial=DFAGCNBranches(recipe,2).state_dict()
payload=torch.load(run/'refit/model.pt',map_location='cpu',weights_only=True)
changed={name:sum(not torch.equal(value,payload['state_dict'][key])
                  for key,value in initial.items() if key.startswith('branches.'+name+'.'))
         for name in ('conv3','conv4','conv5')}
assert all(count>0 for count in changed.values()), changed
selection=json.loads((run/'calibration/selection.json').read_text())
expected=min(score['C'] for score in selection['scores']
             if score['val_oa_percent']==max(row['val_oa_percent'] for row in selection['scores']))
assert selection['selected_C']==expected
wrong_cache_rejected=False
try:
    evaluate(run/'manifest.json',run/'pool_cache',run/'refit',
             root/'results/evidence/dfagcn_pipeline_20261029/review/wrong_test_cache_should_not_exist',
             predict_binary=root/'results/evidence/dfagcn_pipeline_20261029/liblinear/bin/predict')
except ValueError as exc:
    assert 'cache identity' in str(exc), str(exc)
    wrong_cache_rejected=True
assert wrong_cache_rejected
index=json.loads((run/'train_cache/index.json').read_text())
cache=FeatureCache(run/'train_cache',expected_identity=index['identity'])
random_formal_rejected=False
try:
    _check_mode('formal',{'dataset':'NWPU-RESISC45'},FORMAL_MANIFESTS['NWPU-RESISC45'],cache)
except ValueError:
    random_formal_rejected=True
assert random_formal_rejected
record={'engineering_run':str(run.relative_to(root)),'current_source_sha256':source_sha256(),'scope':'retained synthetic engineering artifacts and pure guard checks, no NWPU evaluation',
        'refit_parameter_tensors_changed_from_seeded_initialization':changed,
        'selected_C_matches_best_OA_then_smallest_C':True,
        'pool_cache_rejected_as_test_cache':wrong_cache_rejected,
        'formal_weight_guard_rejects_actual_random_cache_with_known_formal_manifest_argument':random_formal_rejected}
(root/'results/evidence/dfagcn_pipeline_20261029/review/independent_probes.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
