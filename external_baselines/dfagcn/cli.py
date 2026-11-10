"""Explicit independent DFAGCN stages. Nothing downloads weights automatically."""
import argparse
import json
from pathlib import Path
import sys

import torch

from scene_classification.data import load_manifest, select_records
from .config import load_recipe
from .extraction import PREPROCESSING, load_vgg_adapter, extract_cache
from .identity import build_cache_identity
from .pipeline import (FORMAL_MANIFESTS, calibrate, refit, evaluate,
                       feature_source_sha256, sha256_file,subset_limits,selected_rows)
from .smoke import run_smoke


def parser():
    root=argparse.ArgumentParser(description='Independent DFAGCN cache / calibrate / refit / test / smoke')
    root.add_argument('--cpu-threads',type=int,default=1)
    commands=root.add_subparsers(dest='command',required=True)
    for name in ('cache','calibrate','refit','test','smoke'):
        sub=commands.add_parser(name)
        sub.add_argument('--output',required=True)
        sub.add_argument('--device',default='cpu')
        if name!='smoke':
            sub.add_argument('--manifest',required=True)
        if name in ('calibrate','refit','smoke'):
            sub.add_argument('--train-binary',required=True)
        if name in ('calibrate','refit','test','smoke'):
            sub.add_argument('--predict-binary',required=True)
        if name in ('cache','calibrate'):
            sub.add_argument('--recipe',required=True)
            sub.add_argument('--mode',choices=('engineering','formal'),default='engineering')
            sub.add_argument('--train-per-class',type=int)
            sub.add_argument('--val-per-class',type=int)
        if name=='cache':
            sub.add_argument('--data-root',required=True)
            sub.add_argument('--split',choices=('train','val','pool','test'),required=True)
            sub.add_argument('--weights',required=True)
            sub.add_argument('--weights-sha256',required=True)
            sub.add_argument('--weights-kind',choices=('random_smoke','torchvision_imagenet1k_v1'),required=True)
            sub.add_argument('--batch-size',type=int,default=1)
        elif name=='calibrate':
            sub.add_argument('--train-cache',required=True)
            sub.add_argument('--val-cache',required=True)
        elif name=='refit':
            sub.add_argument('--pool-cache',required=True)
            sub.add_argument('--selection',required=True)
        elif name=='test':
            sub.add_argument('--test-cache',required=True)
            sub.add_argument('--refit',required=True)
    return root


def main(argv=None):
    command_parser=parser()
    args=command_parser.parse_args(argv)
    if args.cpu_threads<1:
        command_parser.error('--cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    try:
        if args.command=='cache':
            recipe=load_recipe(args.recipe)
            limits=subset_limits(args.train_per_class,args.val_per_class)
            partial=any(value is not None for value in limits.values())
            if partial and (args.mode=='formal' or args.split=='test'):
                raise ValueError('subset smoke is restricted to engineering train/validation/pool')
            manifest=load_manifest(args.manifest,args.data_root)
            msha=sha256_file(args.manifest)
            if args.mode=='formal' and (FORMAL_MANIFESTS.get(manifest['dataset'])!=msha
                    or args.weights_kind!='torchvision_imagenet1k_v1'
                    or not args.weights_sha256.startswith('397923af')):
                raise ValueError('formal cache requires frozen manifest and identified pretrained VGG16')
            rows=selected_rows(manifest,args.split,limits)
            identity=build_cache_identity(manifest_sha256=msha,class_names=manifest['class_names'],
                samples=[{'path':r['path'],'sha256':r['sha256']} for r in rows],
                weights_sha256=args.weights_sha256,preprocessing=dict(PREPROCESSING,weights_kind=args.weights_kind),
                feature_spec={'pooling_hw':recipe['pooling_hw'],'conv_activation':'pre_relu','fc_activation':'post_relu'},
                source_sha256=feature_source_sha256(),stage='frozen_pretrained')
            adapter=load_vgg_adapter(args.weights,args.weights_sha256,device=args.device)
            cache=extract_cache(adapter,args.data_root,rows,identity,args.output,pool_hw=recipe['pooling_hw'],
                                batch_size=args.batch_size,device=args.device)
            result={'completed':True,'samples':len(cache),'cache_key':cache.key,
                    'content_sha256':cache.content_sha256,'split':args.split,'mode':args.mode}
        elif args.command=='calibrate':
            result=calibrate(args.manifest,args.train_cache,args.val_cache,load_recipe(args.recipe),args.output,
                              train_binary=args.train_binary,predict_binary=args.predict_binary,
                              mode=args.mode,device=args.device,
                              train_per_class=args.train_per_class,val_per_class=args.val_per_class)
        elif args.command=='refit':
            result=refit(args.manifest,args.pool_cache,args.selection,args.output,
                          train_binary=args.train_binary,predict_binary=args.predict_binary,device=args.device)
        elif args.command=='test':
            result=evaluate(args.manifest,args.test_cache,args.refit,args.output,
                             predict_binary=args.predict_binary,device=args.device)
        else:
            result=run_smoke(args.output,train_binary=args.train_binary,predict_binary=args.predict_binary,device=args.device)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        print(f'DFAGCN stage failed: {exc}',file=sys.stderr)
        return 2
