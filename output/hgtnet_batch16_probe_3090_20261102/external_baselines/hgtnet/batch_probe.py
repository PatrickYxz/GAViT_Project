"""Bounded real-NWPU batch16 capacity/compute probe; never a benchmark or epoch."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import run_hgtnet_nwpu_smoke as COMMON

DATA, RES = COMMON.DATA, COMMON.RES
SCHEMA = 'hgtnet-batch16-capacity-probe-v1'
RESTORED_CACHE = Path('/tmp/hgtnet_nwpu_smoke_3090_20261102_fix1_rtx3090_seed42_01/images')
SCOPE = ('16 unique frozen train images reused for one warmup and three measured '
         'disposable optimizer updates; shape/capacity/compute only; no validation, '
         'test, checkpoint, accuracy, epoch, formal training or full-data loader ETA')


def select_probe_records(inputs):
    """Select first path per label0..15 from the frozen smoke train list."""
    rows = DATA._records(inputs['selected']['train'])
    if any(row['split'] != 'train' for row in rows):
        raise ValueError('probe selection accepts training records only')
    first = {}
    for row in sorted(rows, key=lambda value: value['path']):
        if row['label'] < 16: first.setdefault(row['label'], dict(row))
    if set(first) != set(range(16)):
        raise ValueError('probe requires training labels0 through15')
    return [first[label] for label in range(16)]


def _inputs(args):
    return DATA.prepare_inputs(args.manifest, expected_sha256=DATA.MANIFEST_SHA)


def _image_source(args, records):
    for root in dict.fromkeys([RESTORED_CACHE, args.data_root]):
        try:
            DATA.validate_selected(root, records)
            return {'data_source':'verified_images','data_root_used':str(root),'archive_sha_verified':False}
        except (OSError, ValueError): pass
    if not args.archive.is_file() or args.archive.is_symlink() or args.archive.stat().st_size <= 0:
        raise ValueError('16 selected training images unavailable and persistent NWPU ZIP missing')
    return {'data_source':'persistent_zip_pending_full_hash_and_selected_recovery',
            'data_root_used':None,'archive_sha_verified':False,
            'archive_expected_sha256':DATA.ARCHIVE_SHA,'archive_bytes':args.archive.stat().st_size}


def run_command(args, uuid):
    entry = os.environ.get('HGTNET_DELIVERY_ENTRY', str(Path(__file__).resolve()))
    command = [sys.executable, entry, 'run', '--gpu-uuid', uuid, '--target-gpu', '3090']
    for name in ['manifest','data_root','archive','weights','config','output','scratch_root']:
        command += ['--'+name.replace('_','-'), str(getattr(args,name))]
    return shlex.join(command)


def inspect_resources(args):
    blockers = []
    report = dict(schema=SCHEMA,action='inspect',ready=False,formal_training_started=False,
        probe_scope=SCOPE,target_gpu=args.target_gpu,requested_gpu_uuid=args.gpu_uuid,
        manifest=str(args.manifest),weights=str(args.weights),output=str(args.output),
        scratch_root=str(args.scratch_root),blockers=blockers,counts=dict(train=16,val=0,test=0),
        expected_unique_images=16,expected_optimizer_steps=4,expected_processed_images=64,
        batch_size=16,warmup_steps=1,measured_steps=3,
        readiness_scope='stdlib gates; run verifies actual imports/CUDA and full ZIP hash if recovery needed')
    if args.target_gpu != '3090' or os.environ.get('HGTNET_TARGET_GPU','3090') != '3090':
        blockers.append('target_profile')
    try:
        if COMMON.config_sha256(COMMON.load_config(args.config)) != COMMON.CONFIG_SHA:
            raise ValueError('frozen HGTNet config changed')
        report['config_sha256'] = COMMON.CONFIG_SHA
    except (OSError, ValueError, KeyError, TypeError) as exc:
        blockers.append('config'); report['config_error'] = str(exc)
    try:
        records = select_probe_records(_inputs(args))
        report.update(manifest_sha256=DATA.MANIFEST_SHA, selected_paths=[row['path'] for row in records])
        try: report.update(_image_source(args, records))
        except (OSError,ValueError) as exc: blockers.append('data');report['data_error']=str(exc)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        blockers.append('manifest'); report['manifest_error'] = str(exc)
    try:
        if (args.weights.is_symlink() or args.weights.stat().st_size != COMMON.WEIGHTS_BYTES
                or COMMON._sha(args.weights) != COMMON.WEIGHTS_SHA):
            raise ValueError('complete official Swin-B candidate bytes/SHA required')
        report.update(weights_sha256=COMMON.WEIGHTS_SHA,weights_bytes=COMMON.WEIGHTS_BYTES)
    except (OSError,ValueError) as exc: blockers.append('weights');report['weights_error']=str(exc)
    report['dependencies'] = RES.dependency_inventory()
    blockers.extend(RES.dependency_blockers(report['dependencies']))
    report['nvidia'] = RES.inventory_gpu()
    try: report['selected_gpu'] = RES.select_gpu(args.gpu_uuid,report['nvidia'],target_name='NVIDIA GeForce RTX 3090')
    except ValueError as exc: blockers.append('target_gpu');report['gpu_error']=str(exc)
    try: RES.require_new_paths(args.output,args.scratch_root)
    except (OSError,ValueError,RuntimeError) as exc: blockers.append('output_or_scratch');report['path_error']=str(exc)
    report['disk'] = RES.capacity(args.output,args.scratch_root)
    blockers.extend(report['disk']['blockers']);report['ready'] = not blockers
    if report['ready']: report['run_command'] = run_command(args, report['selected_gpu']['uuid'])
    return report


def _run_model(torch, cfg, root, records, weights, output, progress):
    """The only ML boundary: decode only selected train16, initialize on CPU."""
    from external_baselines.hgtnet.model import HGTNet, parameter_counts
    from external_baselines.hgtnet.pretrained import load_pretrained_file
    from external_baselines.hgtnet.batch_probe_engine import run_batch_probe
    dataset = DATA.make_dataset(root, records)
    samples = [dataset[index] for index in range(16)]
    images = torch.stack([sample[0] for sample in samples])
    labels = torch.stack([sample[1] for sample in samples]).to(dtype=torch.long)
    model = HGTNet(cfg)
    pretraining = load_pretrained_file(model, weights, COMMON.WEIGHTS_SHA)
    model.pretraining_loaded = True
    COMMON._write(output/'pretraining.json', pretraining)
    result = run_batch_probe(model, images, labels, device='cuda:0', batch_size=16,
                             warmup_steps=1, measured_steps=3, on_progress=progress)
    result.update(parameters=parameter_counts(model),pretraining=pretraining)
    return result


def _observed_steps(output):
    try: return max(0, int(json.loads((output/'optimizer_progress.json').read_text()).get('optimizer_steps',0)))
    except (OSError,ValueError,TypeError): return 0


def run_probe(args):
    if not args.gpu_uuid: raise ValueError('run requires explicit current physical --gpu-uuid from inspect')
    preflight = inspect_resources(args)
    if not preflight['ready']: raise ValueError('preflight blockers: '+', '.join(preflight['blockers']))
    args.output.mkdir(parents=True,exist_ok=False)
    COMMON._write(args.output/'preflight.json',preflight)
    COMMON._write(args.output/'command.json',dict(argv=list(sys.argv),cwd=os.getcwd(),pid=os.getpid(),
                                                command=run_command(args,args.gpu_uuid)))
    result = dict(schema=SCHEMA,completed=False,training_started=False,formal_training_started=False,
        action='run',probe_scope=SCOPE,input_scope='real_nwpu',target_gpu='3090',seed=42,batch_size=16,
        counts=dict(train=16,val=0,test=0),expected_unique_images=16,expected_optimizer_steps=4,
        expected_processed_images=64,warmup_steps=1,measured_steps=3,optimizer_steps=0)
    started = time.perf_counter()
    with (args.output/'run.log').open('x') as log,contextlib.redirect_stdout(COMMON._Tee(sys.stdout,log)),contextlib.redirect_stderr(COMMON._Tee(sys.stderr,log)):
        try:
            sources = COMMON._source_identity();COMMON._write(args.output/'source_identity.json',sources)
            torch,runtime = COMMON._runtime_and_torch(args,preflight['selected_gpu'])
            COMMON._write(args.output/'runtime.json',runtime)
            cfg = COMMON.load_config(args.config);records = select_probe_records(_inputs(args))
            COMMON._write(args.output/'config.json',cfg)
            COMMON._write(args.output/'selection.json',dict(records=records,unique_images=16,
                counts=result['counts'],selection_rule='first path in frozen smoke train per label0..15',
                manifest_sha256=DATA.MANIFEST_SHA,probe_scope=SCOPE))
            args.scratch_root.mkdir(parents=True,exist_ok=False)
            used_archive = preflight['data_source'] != 'verified_images'
            data_root = (DATA.recover_selected(args.archive,records,args.scratch_root/'images',expected_sha256=DATA.ARCHIVE_SHA)
                         if used_archive else Path(preflight['data_root_used']))
            DATA.validate_selected(data_root,records)
            provenance = dict(data_source='persistent_zip_verified_selected_recovery' if used_archive else 'verified_images',
                archive_sha_verified=used_archive,archive_sha256=DATA.ARCHIVE_SHA if used_archive else None,
                data_root_used=str(data_root),manifest_sha256=DATA.MANIFEST_SHA,
                weights_sha256=COMMON.WEIGHTS_SHA,config_sha256=COMMON.CONFIG_SHA,source_identity=sources)
            COMMON._write(args.output/'provenance.json',provenance)
            def progress(value):
                value = dict(value,pid=os.getpid())
                if value.get('optimizer_step_succeeded') is True or value.get('optimizer_step_completed') is True:
                    COMMON._write(args.output/'optimizer_progress.json',value,replace=True)
                    if not (args.output/'first_step.json').exists(): COMMON._write(args.output/'first_step.json',value)
                COMMON._write(args.output/'progress.json',value,replace=True)
                print('HGTNET_BATCH16_PROBE_PROGRESS '+json.dumps(value,sort_keys=True),flush=True)
            result.update(_run_model(torch,cfg,data_root,records,args.weights,args.output,progress))
            if result.get('completed') is not True or result.get('optimizer_steps') != 4:
                raise ValueError('engine did not complete exactly four optimizer updates')
            if (COMMON._source_identity()!=sources or COMMON._sha(args.manifest)!=DATA.MANIFEST_SHA
                    or COMMON._sha(args.weights)!=COMMON.WEIGHTS_SHA
                    or COMMON.config_sha256(COMMON.load_config(args.config))!=COMMON.CONFIG_SHA):
                raise ValueError('inputs or implementation changed during batch16 probe')
            DATA.validate_selected(data_root,records)
            result.update(provenance,runtime=runtime,physical_gpu_uuid=args.gpu_uuid,completed=True,input_scope='real_nwpu',
                training_started=True,formal_training_started=False,wall_seconds=time.perf_counter()-started,
                timing_scope='after stdlib preflight; includes imports/recovery/CPU initialization and bounded updates')
            COMMON._write(args.output/'summary.json',result)
            with (args.output/'run.exit').open('x') as stream: stream.write('0\n')
            with (args.output/'completion.marker').open('x') as stream: stream.write('HGTNET_BATCH16_PROBE_COMPLETE\n')
            progress(dict(stage='complete',completed=True,training_started=True,optimizer_steps=4))
            print('HGTNET_BATCH16_PROBE_COMPLETE '+str(args.output/'summary.json'),flush=True)
            return 0
        except BaseException as exc:
            steps = _observed_steps(args.output)
            result.update(completed=False,training_started=steps>0,optimizer_steps=steps,
                error=type(exc).__name__+': '+str(exc),wall_seconds=time.perf_counter()-started)
            traceback.print_exc()
            if not (args.output/'summary.json').exists(): COMMON._write(args.output/'summary.json',result)
            with (args.output/'run.exit').open('x') as stream: stream.write('1\n')
            COMMON._write(args.output/'progress.json',dict(stage='failed',completed=False,
                training_started=steps>0,optimizer_steps=steps,error=result['error']),replace=True)
            return 1


def status(output):
    output = Path(output)
    report = dict(schema=SCHEMA,state='incomplete' if output.exists() else 'not_started',
                  output=str(output),completed=False)
    for filename,key in [('summary.json','summary'),('progress.json','progress')]:
        try: report[key] = json.loads((output/filename).read_text())
        except (OSError,ValueError): pass
    try: report['exit_code'] = int((output/'run.exit').read_text().strip())
    except (OSError,ValueError): report['exit_code'] = None
    summary = report.pop('summary',{})
    for key in ['training_started','formal_training_started','target_gpu','counts','batch_size','optimizer_steps','probe_scope',
                'warmup_steps','measured_steps','expected_unique_images','expected_processed_images',
                'unique_images','processed_images','median_step_seconds','compute_images_per_second',
                'measured_step_seconds','measured_images_per_second','step_seconds','cuda_peak_allocated_bytes',
                'cuda_peak_reserved_bytes','wall_seconds','physical_gpu_uuid','weights_sha256',
                'config_sha256','manifest_sha256','error']:
        if key in summary: report[key] = summary[key]
    if summary.get('completed') is True and report['exit_code']==0 and (output/'completion.marker').is_file():
        report.update(state='complete',completed=True)
    if report['exit_code'] not in (None,0): report['state']='failed'
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest='action',required=True)
    for action in ['inspect','run','status']:
        command = sub.add_parser(action)
        command.add_argument('--output',type=Path,default=ROOT/'runs/batch16_seed42_01')
        if action=='status': continue
        command.add_argument('--target-gpu',choices=['3090'],default='3090')
        command.add_argument('--gpu-uuid')
        command.add_argument('--manifest',type=Path,default=DATA.MANIFEST)
        command.add_argument('--data-root',type=Path,default=DATA.DATA_ROOT)
        command.add_argument('--archive',type=Path,default=DATA.ARCHIVE)
        command.add_argument('--weights',type=Path,default=ROOT.parent/'swin_b-68c6b09e.pth')
        command.add_argument('--config',type=Path,default=ROOT/'configs/hgtnet_independent_v0.json')
        command.add_argument('--scratch-root',type=Path,default=Path('/tmp')/(ROOT.name+'_rtx3090_seed42_01'))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action=='run': return run_probe(args)
    print(json.dumps(status(args.output) if args.action=='status' else inspect_resources(args),indent=2,sort_keys=True))
    return 0


if __name__=='__main__': raise SystemExit(main())
