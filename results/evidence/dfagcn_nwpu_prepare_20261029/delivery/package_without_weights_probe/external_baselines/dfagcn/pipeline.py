"""Independent train/validation selection, fresh refit and gated final test.

Frozen cache extraction is separate. Dataset metrics come from manifest labels,
never LIBLINEAR's dummy-label stdout. Engineering outputs are not paper scores.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from scene_classification.data import select_records
from scene_classification.metrics import classification_metrics
from .cache import FeatureCache, sha256_file
from .config import validate_recipe
from .extraction import PREPROCESSING
from .identity import build_cache_identity
from .linear import train_linear, predict_linear
from .model import DFAGCNBranches


FORMAL_MANIFESTS = {
    'NWPU-RESISC45': 'ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e',
    'AID': '30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22',
}
SCHEMA = 'dfagcn-independent-pipeline-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def feature_source_sha256():
    return digest({name: sha256_file(Path(__file__).with_name(name))
                   for name in ('features.py', 'extraction.py', 'cache.py', 'identity.py')})


def source_sha256():
    return digest({name: sha256_file(Path(__file__).with_name(name)) for name in
                   ('config.py', 'model.py', 'graph.py', 'fusion.py', 'pipeline.py', 'linear.py')})


def _write(path, value, *, replace=False):
    path = Path(path)
    if replace:
        temp = path.with_name(path.name + '.tmp')
        temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        temp.replace(path)
    else:
        with path.open('x') as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')


def _sealed(value):
    result = copy.deepcopy(value)
    result['record_sha256'] = digest(value)
    return result


def _read_sealed(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('invalid pipeline record')
    checksum = value.pop('record_sha256', None)
    if checksum != digest(value) or value.get('schema') != SCHEMA:
        raise ValueError('pipeline record changed or unsupported schema')
    return value


def _manifest(path):
    manifest = json.loads(Path(path).read_text())
    names, rows = manifest.get('class_names'), manifest.get('records')
    if (not isinstance(names, list) or len(names) < 2 or len(set(names)) != len(names)
            or names != sorted(names) or not isinstance(rows, list) or not rows):
        raise ValueError('invalid manifest class map or records')
    seen = set()
    for row in rows:
        relative, label = row['path'], row['label']
        rel = Path(relative)
        if (type(label) is not int or not 0 <= label < len(names)
                or row['split'] not in ('train','val','test') or relative in seen
                or rel.is_absolute() or '..' in rel.parts or rel.as_posix() != relative
                or len(rel.parts) < 2 or rel.parts[0] != names[label]
                or len(row['sha256']) != 64):
            raise ValueError('invalid manifest record')
        seen.add(relative)
    if dict(Counter(r['split'] for r in rows)) != manifest.get('counts'):
        raise ValueError('manifest counts mismatch')
    return manifest, sha256_file(path)


def _rows_identity(rows):
    return [{'path': r['path'], 'sha256': r['sha256']} for r in rows]


def _fit_digest(rows):
    return digest([{'path': r['path'], 'sha256': r['sha256'], 'label': r['label']} for r in rows])


def subset_limits(train_per_class=None,val_per_class=None):
    for value in (train_per_class,val_per_class):
        if value is not None and (type(value) is not int or value<1):
            raise ValueError('subset limits must be positive integers')
    return {'train_per_class':train_per_class,'val_per_class':val_per_class}


def selected_rows(manifest,split,limits):
    if split=='pool':
        return sorted(selected_rows(manifest,'train',limits)+selected_rows(manifest,'val',limits),key=lambda r:r['path'])
    return select_records(manifest,split,per_class=limits.get(split+'_per_class'))


def _open_cache(path, manifest, manifest_sha, split, recipe, limits=None):
    rows = selected_rows(manifest, split, limits or subset_limits())
    index = json.loads((Path(path) / 'index.json').read_text())
    declared = index['identity']
    preprocessing = declared['preprocessing']
    if any(preprocessing.get(k) != v for k,v in PREPROCESSING.items()):
        raise ValueError('cache preprocessing differs from this implementation')
    if declared['feature_spec'].get('pooling_hw') != recipe['pooling_hw']:
        raise ValueError('cache pooling differs from recipe')
    if (declared['feature_spec'].get('conv_activation') != 'pre_relu'
            or declared['feature_spec'].get('fc_activation') != 'post_relu'):
        raise ValueError('cache activation identity is invalid')
    expected = build_cache_identity(manifest_sha256=manifest_sha,
        class_names=manifest['class_names'], samples=_rows_identity(rows),
        weights_sha256=declared['weights_sha256'], preprocessing=preprocessing,
        feature_spec=declared['feature_spec'], source_sha256=feature_source_sha256(),
        stage='frozen_pretrained')
    return FeatureCache(path, expected_identity=expected), rows


def _context(cache):
    result = copy.deepcopy(cache.identity)
    result.pop('samples')
    return result


def _check_mode(mode, manifest, manifest_sha, cache):
    if mode not in ('engineering','formal'):
        raise ValueError('mode must be engineering or formal')
    kind = cache.identity['preprocessing'].get('weights_kind')
    if kind not in ('random_smoke','torchvision_imagenet1k_v1'):
        raise ValueError('unknown backbone weight provenance')
    if mode == 'formal':
        if FORMAL_MANIFESTS.get(manifest['dataset']) != manifest_sha:
            raise ValueError('formal mode requires the archived frozen manifest')
        if kind != 'torchvision_imagenet1k_v1' or not cache.identity['weights_sha256'].startswith('397923af'):
            raise ValueError('formal mode requires the identified official pretrained VGG16 file')


def _batch(cache, rows, indices, device):
    samples = [cache[int(i)] for i in indices]
    values = {name: torch.stack([sample[name] for sample in samples]).to(device)
              for name in ('conv3','conv4','conv5','fc')}
    labels = torch.tensor([rows[int(i)]['label'] for i in indices], device=device)
    return values, labels


def _save_checkpoint(path, model, metadata, *, replace=False):
    payload = {'metadata': copy.deepcopy(metadata),
               'state_dict': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
    if replace:
        temp = path.with_name(path.name + '.tmp')
        torch.save(payload, temp)
        temp.replace(path)
    else:
        with path.open('xb') as stream:
            torch.save(payload, stream)


def _load_checkpoint(path, expected, recipe, classes, device):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('metadata') != expected:
        raise ValueError('checkpoint metadata does not match phase identity')
    model = DFAGCNBranches(recipe, classes).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model


def _train(cache, rows, recipe, output, metadata, device, *, val=None, epochs=None):
    torch.manual_seed(recipe['seed'])
    model = DFAGCNBranches(recipe, len(cache.identity['class_names'])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe['learning_rate'],
                                 weight_decay=recipe['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=recipe['lr_step'], gamma=recipe['lr_gamma'])
    generator = torch.Generator().manual_seed(recipe['seed'])
    history, best_loss, best_epoch = [], math.inf, None
    batch_size = recipe['batch_size']
    for epoch in range(1, (epochs or recipe['epochs']) + 1):
        started = time.monotonic()
        model.train()
        loss_sum = 0.0
        order = torch.randperm(len(rows), generator=generator).tolist()
        for start in range(0,len(rows),batch_size):
            indices = order[start:start+batch_size]
            features, labels = _batch(cache,rows,indices,device)
            optimizer.zero_grad(set_to_none=True)
            losses = [F.cross_entropy(logits,labels) for logits in model(features).values()]
            loss = sum(losses)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite branch training loss')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        entry = {'epoch':epoch,'train_ce_sum':loss_sum/len(rows),
                 'learning_rate':optimizer.param_groups[0]['lr']}
        if val is not None:
            model.eval()
            val_sum = 0.0
            with torch.no_grad():
                for start in range(0,len(val[1]),batch_size):
                    indices=list(range(start,min(start+batch_size,len(val[1]))))
                    values,labels=_batch(val[0],val[1],indices,device)
                    val_sum += sum(float(F.cross_entropy(x,labels)) for x in model(values).values()) * len(indices)
            entry['val_ce_sum'] = val_sum / len(val[1])
            if not math.isfinite(entry['val_ce_sum']):
                raise ValueError('nonfinite validation loss')
            if entry['val_ce_sum'] < best_loss:
                best_loss, best_epoch = entry['val_ce_sum'], epoch
                _save_checkpoint(output/'best.pt',model,dict(metadata,epoch=epoch),replace=True)
        entry['elapsed_seconds']=time.monotonic()-started
        history.append(entry)
        _write(output/'history.json',history,replace=True)
        scheduler.step()
    final_epoch=history[-1]['epoch']
    _save_checkpoint(output/'final.pt',model,dict(metadata,epoch=final_epoch))
    if val is None:
        return model,final_epoch,history
    model=_load_checkpoint(output/'best.pt',dict(metadata,epoch=best_epoch),recipe,
                           len(cache.identity['class_names']),device)
    return model,best_epoch,history


def _export(model,cache,rows,recipe,device):
    values=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(rows),recipe['batch_size']):
            indices=list(range(start,min(start+recipe['batch_size'],len(rows))))
            features,_=_batch(cache,rows,indices,device)
            values.append(model.fused_features(features).cpu().numpy())
    return np.concatenate(values,axis=0)


def calibrate(manifest_path, train_cache_path, val_cache_path, recipe, output_dir,
              *, train_binary, predict_binary, mode='engineering', device='cpu',
              train_per_class=None,val_per_class=None):
    recipe=validate_recipe(recipe)
    limits=subset_limits(train_per_class,val_per_class)
    partial=any(x is not None for x in limits.values())
    if mode=='formal' and partial:
        raise ValueError('formal calibration cannot use a subset')
    manifest,msha=_manifest(manifest_path)
    train,train_rows=_open_cache(train_cache_path,manifest,msha,'train',recipe,limits)
    val,val_rows=_open_cache(val_cache_path,manifest,msha,'val',recipe,limits)
    if _context(train)!=_context(val):
        raise ValueError('training/validation extraction identities differ')
    _check_mode(mode,manifest,msha,train)
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    metadata={'schema':SCHEMA,'phase':'calibration','fit_scope':'train_subset' if partial else 'train','mode':mode,
              'limits':limits,
              'manifest_sha256':msha,'recipe_sha256':digest(recipe),'source_sha256':source_sha256(),
              'fit_records_sha256':_fit_digest(train_rows),'cache_key':train.key,
              'cache_content_sha256':train.content_sha256,'seed':recipe['seed']}
    model,epoch,history=_train(train,train_rows,recipe,output,metadata,device,val=(val,val_rows))
    fit_features=_export(model,train,train_rows,recipe,device)
    validation_features=_export(model,val,val_rows,recipe,device)
    scores=[]
    for i,C in enumerate(sorted(recipe['C_grid'])):
        linear_model=train_linear(fit_features,[r['label'] for r in train_rows],output/f'C_{i}',
                                  train_binary=train_binary,C=C,solver=recipe['liblinear_solver'],bias=recipe['liblinear_bias'])
        predictions=predict_linear(validation_features,linear_model,output/f'val_C_{i}',predict_binary=predict_binary)
        metrics=classification_metrics([r['label'] for r in val_rows],predictions,manifest['class_names'])
        scores.append({'C':C,'val_oa_percent':metrics['oa_percent'],'metrics':metrics})
    selected=max(scores,key=lambda s:(s['val_oa_percent'],-s['C']))
    record=dict(metadata,recipe=recipe,selected_epoch=epoch,selected_C=selected['C'],
                val_cache_key=val.key,val_cache_content_sha256=val.content_sha256,
                feature_identity=_context(train),scores=scores,best_checkpoint_sha256=sha256_file(output/'best.pt'),
                final_checkpoint_sha256=sha256_file(output/'final.pt'),
                elapsed_seconds=time.monotonic()-started,completed=True)
    _write(output/'selection.json',_sealed(record))
    return record


def _selection(path,manifest,msha):
    selected=_read_sealed(path)
    recipe=validate_recipe(selected['recipe'])
    limits=subset_limits(**selected['limits'])
    scope='train_subset' if any(x is not None for x in limits.values()) else 'train'
    if (selected.get('phase')!='calibration' or not selected.get('completed')
            or selected.get('fit_scope')!=scope or selected['manifest_sha256']!=msha
            or selected['recipe_sha256']!=digest(recipe) or selected['source_sha256']!=source_sha256()
            or selected['fit_records_sha256']!=_fit_digest(selected_rows(manifest,'train',limits))
            or type(selected['selected_epoch']) is not int or not 1<=selected['selected_epoch']<=recipe['epochs']
            or selected['selected_C'] not in recipe['C_grid']):
        raise ValueError('selection does not match data/recipe/source or valid selection scope')
    for name in ('best','final'):
        if sha256_file(Path(path).parent/(name+'.pt'))!=selected[name+'_checkpoint_sha256']:
            raise ValueError('calibration checkpoint changed after selection')
    return selected,recipe


def refit(manifest_path,pool_cache_path,selection_path,output_dir,*,train_binary,predict_binary,device='cpu'):
    manifest,msha=_manifest(manifest_path)
    selected,recipe=_selection(selection_path,manifest,msha)
    limits=subset_limits(**selected['limits'])
    partial=any(x is not None for x in limits.values())
    if selected['mode']=='formal' and partial:
        raise ValueError('formal refit cannot use a subset')
    cache,rows=_open_cache(pool_cache_path,manifest,msha,'pool',recipe,limits)
    if _context(cache)!=selected['feature_identity']:
        raise ValueError('refit frozen extraction differs from calibration')
    _check_mode(selected['mode'],manifest,msha,cache)
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    metadata={'schema':SCHEMA,'phase':'refit','fit_scope':'train_plus_val_subset' if partial else 'train_plus_val',
              'mode':selected['mode'],'limits':limits,
              'manifest_sha256':msha,'recipe_sha256':digest(recipe),'source_sha256':source_sha256(),
              'fit_records_sha256':_fit_digest(rows),'cache_key':cache.key,
              'cache_content_sha256':cache.content_sha256,'seed':recipe['seed'],
              'selection_sha256':sha256_file(selection_path),'selected_C':selected['selected_C']}
    model,epoch,history=_train(cache,rows,recipe,output,metadata,device,epochs=selected['selected_epoch'])
    _save_checkpoint(output/'model.pt',model,dict(metadata,epoch=epoch))
    features=_export(model,cache,rows,recipe,device)
    linear_model=train_linear(features,[r['label'] for r in rows],output/'linear',train_binary=train_binary,
                               C=selected['selected_C'],solver=recipe['liblinear_solver'],bias=recipe['liblinear_bias'])
    # Verify reload and actual classifier inference on the fitting pool. Not test accuracy.
    reloaded=_load_checkpoint(output/'model.pt',dict(metadata,epoch=epoch),recipe,len(manifest['class_names']),device)
    reexported=_export(reloaded,cache,rows,recipe,device)
    reload_error=float(np.max(np.abs(features-reexported)))
    if reload_error>1e-6:
        raise ValueError('refit feature reload mismatch')
    predictions=predict_linear(reexported,linear_model,output/'fit_predictions',predict_binary=predict_binary)
    report=dict(metadata,recipe=recipe,feature_identity=_context(cache),epoch=epoch,
                fit_samples=len(rows),class_names=manifest['class_names'],reload_max_abs_error=reload_error,
                fit_metrics=classification_metrics([r['label'] for r in rows],predictions,manifest['class_names']),
                model_sha256=sha256_file(output/'model.pt'),linear_model_sha256=sha256_file(linear_model),
                elapsed_seconds=time.monotonic()-started,completed=True)
    _write(output/'refit.json',_sealed(report))
    return report


def evaluate(manifest_path,test_cache_path,refit_dir,output_dir,*,predict_binary,device='cpu'):
    manifest,msha=_manifest(manifest_path)
    directory=Path(refit_dir)
    if not (directory/'refit.json').is_file():
        raise ValueError('final test requires a completed fresh refit bundle')
    report=_read_sealed(directory/'refit.json')
    recipe=validate_recipe(report['recipe'])
    pool_rows=select_records(manifest,'pool')
    if (report.get('phase')!='refit' or report.get('fit_scope')!='train_plus_val' or not report.get('completed')
            or report['manifest_sha256']!=msha or report['recipe_sha256']!=digest(recipe)
            or report['source_sha256']!=source_sha256() or report['fit_records_sha256']!=_fit_digest(pool_rows)
            or report['fit_samples']!=len(pool_rows) or report['class_names']!=manifest['class_names']):
        raise ValueError('refit bundle does not match full fitting pool/data/source')
    if any(value is not None for value in subset_limits(**report['limits']).values()):
        raise ValueError('final test requires the full fitting pool')
    checkpoint=directory/'model.pt'
    linear_model=directory/'linear'/'model.txt'
    if sha256_file(checkpoint)!=report['model_sha256'] or sha256_file(linear_model)!=report['linear_model_sha256']:
        raise ValueError('refit model bytes changed')
    cache,rows=_open_cache(test_cache_path,manifest,msha,'test',recipe)
    if _context(cache)!=report['feature_identity']:
        raise ValueError('test frozen extraction differs from refit')
    _check_mode(report['mode'],manifest,msha,cache)
    expected={key:report[key] for key in ('schema','phase','fit_scope','mode','limits','manifest_sha256','recipe_sha256',
              'source_sha256','fit_records_sha256','cache_key','cache_content_sha256','seed','selection_sha256','selected_C')}
    model=_load_checkpoint(checkpoint,dict(expected,epoch=report['epoch']),recipe,len(manifest['class_names']),device)
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    features=_export(model,cache,rows,recipe,device)
    predictions=predict_linear(features,linear_model,output/'predictions',predict_binary=predict_binary)
    result={'schema':SCHEMA,'phase':'test','mode':report['mode'],'manifest_sha256':msha,
            'refit_record_sha256':sha256_file(directory/'refit.json'),'cache_content_sha256':cache.content_sha256,
            'metrics':classification_metrics([r['label'] for r in rows],predictions,manifest['class_names']),
            'predictions':predictions,'sample_paths':[r['path'] for r in rows],
            'cached_pipeline_elapsed_seconds':time.monotonic()-started,
            'timing_scope':'pooled frozen-cache read, graph/fusion and linear prediction; excludes VGG extraction',
            'completed':True}
    _write(output/'test.json',_sealed(result))
    return result
