"""Frozen, disclosed independent formal recipe; no ML imports."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

RECIPE = dict(schema='hgtnet-independent-formal-recipe-v1',seed=42,max_epochs=100,
    batch_size=16,eval_batch_size=16,optimizer='Adam',lr=1e-5,weight_decay=5e-5,
    betas=[0.9,0.999],eps=1e-8,loss='CrossEntropyLoss',label_smoothing=0.0,
    scheduler=None,warmup_epochs=0,gradient_clipping=False,amp=False,tf32=False,
    num_workers=0,pin_memory=False,drop_last=False,epoch_seed='42+epoch',
    train_transform=['RGB','Resize256x256_bilinear','RandomCrop224','RandomHorizontalFlip_p0.5',
                     'ToTensor','ImageNet_normalize'],
    eval_transform=['RGB','Resize256x256_bilinear','CenterCrop224','ToTensor','ImageNet_normalize'],
    selection='validation_correct_count_strict_greater_tie_earliest',
    refit='fresh_equal_initial_state_pool_selected_epoch',
    disclosure='independent nonofficial implementation; paper confirms epochs/batch/Adam/lr/WD/CE and crop/flip only; precise augmentation, defaults, LR schedule, selection and preprocessing are explicit engineering choices')
REAL_COUNTS=dict(train=5040,val=1260,pool=6300,test=25200)
IDENTITIES=dict(manifest_sha256='ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e',
    config_sha256='af6ef410125e9d47c87acd0acec1a6db6e0119656793124ceafcb2379eedac25',
    weights_sha256='68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893')

def recipe_sha256(recipe):
    return hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def load_recipe(path):
    value=json.loads(Path(path).read_text())
    if value!=RECIPE or recipe_sha256(value)!=recipe_sha256(RECIPE):
        raise ValueError('Frozen independent formal recipe changed')
    return value

def validate_binding(value):
    b=json.loads(json.dumps(value,sort_keys=True,allow_nan=False))
    if b.get('schema')!='hgtnet-formal-binding-v1' or b.get('input_scope') not in ('real_nwpu','cpu_fixture'):
        raise ValueError('Explicit real_nwpu or cpu_fixture formal binding required')
    recipe=b.get('recipe');counts=b.get('counts');names=b.get('class_names')
    if not isinstance(recipe,dict) or b.get('recipe_sha256')!=recipe_sha256(recipe):
        raise ValueError('Recipe SHA mismatch')
    real=b['input_scope']=='real_nwpu'
    allowed={} if real else {k:recipe.get(k) for k in ['max_epochs','batch_size','eval_batch_size']}
    expected=dict(RECIPE,**allowed)
    if recipe_sha256(recipe)!=recipe_sha256(expected) or any(type(recipe[k]) is not int or not 1<=recipe[k]<=limit
        for k,limit in [('max_epochs',100 if real else 3),('batch_size',16),('eval_batch_size',16)]):
        raise ValueError('Formal recipe or explicit CPU fixture bounds changed')
    if (not isinstance(counts,dict) or set(counts)!=set(REAL_COUNTS)
            or any(type(n) is not int or n<1 for n in counts.values())
            or counts['pool']!=counts['train']+counts['val'] or (real and counts!=REAL_COUNTS)):
        raise ValueError('Formal split counts changed')
    if not isinstance(names,list) or len(names)!=45 or len(set(names))!=45 or any(not isinstance(n,str) or not n for n in names):
        raise ValueError('Forty-five class identities required')
    for key,expected_sha in IDENTITIES.items():
        if not isinstance(b.get(key),str) or not re.fullmatch('[0-9a-f]{64}',b[key]) or (real and b[key]!=expected_sha):
            raise ValueError('Input identity mismatch: '+key)
    if not isinstance(b.get('source_identity'),dict) or not b['source_identity'] or any(
            not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in b['source_identity'].values()):
        raise ValueError('Complete source identities required')
    records=b.get('records',{})
    for split in ['val','test']:
        rows=records.get(split)
        if not isinstance(rows,list) or len(rows)!=counts[split]:raise ValueError('Ordered evaluation identities required')
        paths=[]
        for row in rows:
            name=row.get('path');label=row.get('label')
            if (not isinstance(name,str) or not name or '\\' in name or ':' in name
                    or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
                    or PurePosixPath(name).as_posix()!=name or type(label) is not int or not 0<=label<45
                    or (real and (len(PurePosixPath(name).parts)!=2 or name.split('/')[0]!=names[label]))):
                raise ValueError('Invalid ordered evaluation record')
            paths.append(name)
        if len(set(paths))!=len(paths):raise ValueError('Duplicate evaluation paths')
    if set(r['path'] for r in records['val']) & set(r['path'] for r in records['test']):
        raise ValueError('Validation/test overlap')
    return b
