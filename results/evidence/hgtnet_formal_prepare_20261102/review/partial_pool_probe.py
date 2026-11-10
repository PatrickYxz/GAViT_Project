"""Isolate resumed staging after a hard interruption left an incomplete pool."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from external_baselines.hgtnet import formal_entry as entry,formal_protocol as p
HERE=Path(__file__).resolve().parent
label=sys.argv[1] if len(sys.argv)>1 else 'pre'
recipe=copy.deepcopy(p.RECIPE);recipe.update(max_epochs=2,batch_size=2,eval_batch_size=2)
rows=[]
for split in ['train','val','test']:
    for index in [0,1]:
        raw=(split+str(index)).encode()
        rows.append(dict(path='c%02d/%s%d.jpg'%(index,split,index),label=index,split=split,
            sha256=hashlib.sha256(raw).hexdigest(),_raw=raw))
selected={split:[{k:v for k,v in row.items() if k!='_raw'} for row in rows if row['split']==split] for split in ['train','val','test']}
selected['pool']=selected['train']+selected['val']
inputs=dict(class_names=['c%02d'%i for i in range(45)],counts=dict(train=2,val=2,pool=4,test=2),selected=selected)
binding=dict(schema='hgtnet-formal-binding-v1',input_scope='cpu_fixture',recipe=recipe,recipe_sha256=p.recipe_sha256(recipe),
    class_names=inputs['class_names'],counts=inputs['counts'],records={s:[dict(path=r['path'],label=r['label']) for r in selected[s]] for s in ['val','test']},
    manifest_sha256='a'*64,config_sha256='b'*64,weights_sha256='c'*64,source_identity={'fixture.py':'d'*64})
reached=[]
with tempfile.TemporaryDirectory() as d:
    root=Path(d).resolve();output=root/'output';output.mkdir();scratch=root/'scratch';scratch.mkdir()
    partial=scratch/'pool';partial.mkdir();(partial/'incomplete.raw').write_bytes(b'keep interrupted data')
    archive=root/'archive.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for row in rows:z.writestr(row['path'],row['_raw'])
    archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    args=entry.parser().parse_args(['resume','--gpu-uuid','GPU-fixture','--output',str(output),'--scratch-root',str(scratch),
        '--archive',str(archive),'--data-root',str(root/'missing')])
    def model_stage(a,*args):reached.append(str(a._pool_root));raise RuntimeError('stop after successful resumed staging')
    with patch.object(entry,'inspect_resources',return_value=dict(ready=True,selected_gpu={},archive_available=True)),\
        patch.object(entry.DATA,'prepare_inputs',return_value=inputs),patch.object(entry,'_binding',return_value=binding),\
        patch.object(entry.COMMON,'_runtime_and_torch',return_value=(object(),{})),\
        patch.object(entry.DATA,'ARCHIVE_SHA',archive_sha),patch.object(entry,'_assert_unchanged'),\
        patch.object(entry,'_run_model',side_effect=model_stage):
        result=entry.run_formal_entry(args,resume=True)
    summary=json.loads((output/'summary.json').read_text())
    record=dict(scope='mocked runtime/ownership with real selected ZIP recovery; no training/GPU',
        exit_code=result,model_stage_reached=bool(reached),pool_roots=reached,error=summary['error'],
        partial_directory_preserved=(partial/'incomplete.raw').read_bytes()==b'keep interrupted data',
        entry_sha256=hashlib.sha256((ROOT/'external_baselines/hgtnet/formal_entry.py').read_bytes()).hexdigest())
(HERE/('partial_pool_'+label+'_validation.json')).write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(record,indent=2,sort_keys=True))
