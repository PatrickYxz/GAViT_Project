"""Real CPU image→VGG→graph→LIBLINEAR engineering fixture, with retained files."""
import copy
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
import torch
import torchvision

from scene_classification.data import prepare_manifest, select_records
from .cache import FeatureCache, write_feature_cache
from .config import load_recipe
from .extraction import PREPROCESSING, random_vgg_adapter, adapter_sha256, extract_cache
from .identity import build_cache_identity
from .pipeline import (calibrate, refit, evaluate, feature_source_sha256, sha256_file,
                       _write, digest)


DEFAULT_RECIPE = Path(__file__).resolve().parents[2] / 'configs' / 'dfagcn_independent_v1.json'


def run_smoke(output_dir, *, train_binary, predict_binary, device='cpu'):
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    images=output/'images'
    for label,name in enumerate(('class_a','class_b')):
        (images/name).mkdir(parents=True)
        for index in range(10):
            # Actual unique image bytes; invented scenes are never called NWPU.
            pixels=np.zeros((24,24,3),dtype=np.uint8)
            pixels[:,:,0]=label*140+index
            pixels[:,:,1]=np.arange(24,dtype=np.uint8)[None,:]*7
            pixels[:,:,2]=index*17
            Image.fromarray(pixels).save(images/name/f'{index:02}.png')
    manifest_path=output/'manifest.json'
    manifest=prepare_manifest(images,manifest_path,dataset='synthetic')
    recipe=load_recipe(DEFAULT_RECIPE)
    recipe.update(pooling_hw=[3,4],knn_k=2,gcn_dims=[8],fc_dims=[512],dropout=0.0,
                  epochs=1,batch_size=4,C_grid=[0.1,1.0])
    recipe['choices']['engineering_fixture']='Synthetic/random weights,12 nodes,GCN8,FC512,1 epoch; not the production recipe or a dataset score.'
    _write(output/'recipe.json',recipe)
    extraction_started=time.monotonic()
    adapter=random_vgg_adapter(recipe['seed']).to(device)
    weights_digest=adapter_sha256(adapter)
    all_rows=sorted(manifest['records'],key=lambda r:r['path'])
    def identity(rows):
        return build_cache_identity(manifest_sha256=sha256_file(manifest_path),
            class_names=manifest['class_names'],samples=[{'path':r['path'],'sha256':r['sha256']} for r in rows],
            weights_sha256=weights_digest,preprocessing=dict(PREPROCESSING,weights_kind='random_smoke'),
            feature_spec={'pooling_hw':recipe['pooling_hw'],'conv_activation':'pre_relu','fc_activation':'post_relu'},
            source_sha256=feature_source_sha256(),stage='frozen_pretrained')
    frozen=extract_cache(adapter,images,all_rows,identity(all_rows),output/'all_frozen',
                          pool_hw=recipe['pooling_hw'],batch_size=1,device=device)
    extraction_seconds=time.monotonic()-extraction_started
    del adapter
    lookup={row['path']:i for i,row in enumerate(all_rows)}
    for split in ('train','val','pool','test'):
        rows=select_records(manifest,split)
        def batches():
            for start in range(0,len(rows),recipe['batch_size']):
                samples=[frozen[lookup[row['path']]] for row in rows[start:start+recipe['batch_size']]]
                yield {name:torch.stack([sample[name] for sample in samples]) for name in ('conv3','conv4','conv5','fc')}
        write_feature_cache(output/f'{split}_cache',identity(rows),batches())
    selection=calibrate(manifest_path,output/'train_cache',output/'val_cache',recipe,output/'calibration',
                         train_binary=train_binary,predict_binary=predict_binary,mode='engineering',device=device)
    report=refit(manifest_path,output/'pool_cache',output/'calibration'/'selection.json',output/'refit',
                 train_binary=train_binary,predict_binary=predict_binary,device=device)
    test=evaluate(manifest_path,output/'test_cache',output/'refit',output/'evaluation',
                   predict_binary=predict_binary,device=device)
    result={'schema':'dfagcn-real-engineering-smoke-v1','completed':True,'mode':'engineering',
            'dataset':'synthetic','pretrained':False,'weights_kind':'random_smoke',
            'images':len(all_rows),'counts':manifest['counts'],'recipe_sha256':digest(recipe),
            'weights_tensor_sha256':weights_digest,'selected_epoch':selection['selected_epoch'],
            'selected_C':selection['selected_C'],'refit_samples':report['fit_samples'],
            'reload_max_abs_error':report['reload_max_abs_error'],
            'runtime':{'python':platform.python_version(),'torch':torch.__version__,
                       'torchvision':torchvision.__version__,'numpy':np.__version__,
                       'device':str(device),'cpu_threads':torch.get_num_threads()},
            'vgg_and_frozen_cache_seconds':extraction_seconds,
            'total_elapsed_seconds':time.monotonic()-started,
            'test_samples':test['metrics']['samples'],
            'claim_scope':'Engineering image/VGG/graph/train/refit/LIBLINEAR/test chain only; no NWPU score or pretrained-parity claim.'}
    _write(output/'smoke.json',result)
    return result
