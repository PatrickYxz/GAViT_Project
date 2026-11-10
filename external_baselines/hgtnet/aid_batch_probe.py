"""AID30 fixed train16, four disposable updates on one idle RTX3060.

inspect/status import only stdlib. Completion verifies recorded artifacts, not
GPU recomputation. This capacity gate provides no accuracy or formal epoch.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from external_baselines.hgtnet import aid_entry as SAFE
DATA,RES=SAFE.DATA,SAFE.RES
SCHEMA='hgtnet-aid-batch16-capacity-probe-v1'
DELIVERY_SCHEMA='hgtnet-aid-batch16-delivery-v1'
CONFIG_SHA,WEIGHTS_SHA,WEIGHTS_BYTES=SAFE.CONFIG_SHA,SAFE.WEIGHTS_SHA,SAFE.WEIGHTS_BYTES
COUNTS={'train':16,'val':0,'test':0}
SOURCE_FILES=(
    'external_baselines/__init__.py','external_baselines/hgtnet/__init__.py',
    *('external_baselines/hgtnet/'+name+'.py' for name in (
        'aid_data','aid_entry','aid_batch_probe','aid_batch_probe_engine','config','model','attention','graph',
        'pretrained','smoke','checkpoint','nwpu_engine','resources')),
    'configs/hgtnet_aid_independent_v0.json','tools/inspect_hgtnet_aid_inputs.py',
    'tools/package_hgtnet_aid_batch16_probe.py')
SELECTION_RULE='first path in frozen smoke train per label0..15'
SCOPE='real AID train16 repeated for warmup1/measured3 disposable updates; capacity/compute only; no validation/test/accuracy/checkpoint/epoch'
COMPLETION_CHECKS=dict(source_identity=True,config=True,manifest=True,weights=True,selected_images=True,
                       selected_image_count=16,test_image_bytes_read=0,val_image_bytes_read=0)
GROUPS={'patch_embed','transformer','dual_attention','graph','head'}
_sha,_write,_read_json=SAFE._sha,SAFE._write,SAFE._read_json
_new_output,_run_lock,lock_path=SAFE._new_output,SAFE._run_lock,SAFE.lock_path
select_gpu,check_runtime,_check_cuda_device=SAFE.select_gpu,SAFE.check_runtime,SAFE._check_cuda_device
_check_config,_runtime_and_torch=SAFE._check_config,SAFE._runtime_and_torch
_same_json=SAFE._same_json

def _source_identity():
    identity={name:_sha(ROOT/name) for name in SOURCE_FILES}
    metadata=ROOT/'delivery_manifest.json'
    if metadata.exists() or metadata.is_symlink():
        manifest=_read_json(metadata)
        expected={name:manifest['files'][name]['sha256'] for name in SOURCE_FILES}
        if manifest.get('schema')!=DELIVERY_SCHEMA or identity!=expected:
            raise ValueError('installed batch16 immutable source identity changed')
    return identity

def select_probe_records(inputs):
    rows=DATA._records(inputs['selected']['train'])
    if any(row['split']!='train' for row in rows):raise ValueError('probe accepts original train records only')
    first={}
    for row in sorted(rows,key=lambda item:item['path']):
        if row['label']<16:first.setdefault(row['label'],row)
    if set(first)!=set(range(16)):raise ValueError('probe requires training labels0 through15')
    return [first[label] for label in range(16)]

def run_command(args,uuid):
    entry=os.environ.get('HGTNET_AID_BATCH16_DELIVERY_ENTRY',str(Path(__file__).resolve()))
    command=[sys.executable,str(Path(entry).absolute()),'run','--gpu-uuid',uuid]
    for name in ('manifest','data_root','weights','config','output'):
        command+=['--'+name.replace('_','-'),str(Path(getattr(args,name)).absolute())]
    return shlex.join(command)

def _check_weights(path):
    with DATA.inspector.open_regular_readonly(path) as stream:
        size=os.fstat(stream.fileno()).st_size;digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if size!=WEIGHTS_BYTES or digest!=WEIGHTS_SHA:raise ValueError('complete candidate weights size/SHA mismatch')
    return dict(weights_sha256=digest,weights_bytes=size)

def inspect_resources(args):
    blockers=[]
    report=dict(schema=SCHEMA,action='inspect',ready=False,training_started=False,formal_training_started=False,
        input_scope='real_aid',counts=dict(COUNTS),target_gpu='3060',requested_gpu_uuid=args.gpu_uuid,
        manifest=str(args.manifest),data_root=str(args.data_root),weights=str(args.weights),output=str(args.output),
        batch_size=16,warmup_steps=1,measured_steps=3,expected_unique_images=16,expected_processed_images=64,
        expected_optimizer_steps=4,probe_scope=SCOPE,blockers=blockers,
        readiness_scope='stdlib full-source/config/manifest/weights/train16 bytes; run verifies actual imports/CUDA')
    for label,check in [('source',lambda:report.update(source_identity=_source_identity())),
                        ('config',lambda:_check_config(args.config)),('output',lambda:_new_output(args.output)),
                        ('weights',lambda:report.update(_check_weights(args.weights)))]:
        try:check()
        except (OSError,ValueError,KeyError,RuntimeError) as exc:blockers.append(label);report[label+'_error']=str(exc)
    try:
        inputs=DATA.prepare_inputs(args.manifest);records=select_probe_records(inputs)
        report.update(manifest_sha256=inputs['manifest_sha256'],selected_paths=[row['path'] for row in records],
                      selected_images=DATA.validate_selected(args.data_root,records))
    except (OSError,ValueError,KeyError,TypeError) as exc:blockers.append('inputs');report['input_error']=str(exc)
    report['dependencies']=RES.dependency_inventory();blockers.extend(SAFE._dependency_blockers(report['dependencies']))
    report['nvidia']=RES.inventory_gpu()
    try:report['selected_gpu']=select_gpu(args.gpu_uuid,report['nvidia'])
    except ValueError as exc:blockers.append('target_gpu');report['gpu_error']=str(exc)
    try:
        report['disk']=RES._capacity_path(args.output,RES.WORK_BYTES)
        if report['disk']['free_bytes']<RES.WORK_BYTES:blockers.append('disk_capacity')
    except (OSError,ValueError) as exc:blockers.append('disk_capacity');report['disk_error']=str(exc)
    report['ready']=not blockers
    if report['ready']:report.update(config_sha256=CONFIG_SHA,run_command=run_command(args,report['selected_gpu']['uuid']))
    return report

def _positive(value):return type(value) in (int,float) and math.isfinite(value) and value>0

def _gradients_ok(value,updated):
    if (not isinstance(value,dict) or value.get('passed') is not True or value.get('missing')!=[]
            or value.get('nonfinite')!=[] or set(value.get('groups',{}))!=GROUPS
            or not _same_json(updated,{name:True for name in GROUPS})):return False
    groups=value['groups']
    return (all(type(item.get('tensors')) is int and item['tensors']>0 and _positive(item.get('l2')) for item in groups.values())
            and sum(item['tensors'] for item in groups.values())==451)

def _completion_evidence_ok(output,summary,*,final_progress=True):
    try:
        output=Path(output)
        fixed=dict(schema=SCHEMA,completed=True,training_started=True,formal_training_started=False,input_scope='real_aid',
            target_gpu='3060',counts=COUNTS,seed=42,batch_size=16,unique_images=16,processed_images=64,optimizer_steps=4,
            warmup_steps=1,measured_steps=3,device='cuda:0',config_sha256=CONFIG_SHA,manifest_sha256=DATA.MANIFEST_SHA,
            weights_sha256=WEIGHTS_SHA,weights_bytes=WEIGHTS_BYTES,completion_checks=COMPLETION_CHECKS,
            data_source='verified_images',parameters_finite_after=True,gradient_clipping=False,accuracy_metrics=None,checkpoint_saved=False)
        if any(not _same_json(summary.get(key),value) for key,value in fixed.items()):return False
        if any(key in summary for key in ('val_metrics','test_metrics','checkpoint','best_epoch','final_epoch')):return False
        uuid=summary['physical_gpu_uuid'];runtime=summary['runtime']
        if not RES._uuid(uuid):return False
        check_runtime(runtime,uuid);_check_cuda_device(runtime['cuda_device'],runtime['gpu'])
        if not _same_json(_read_json(output/'runtime.json'),runtime):return False
        sources=_source_identity()
        if summary['source_identity']!=sources or _read_json(output/'source_identity.json')!=sources:return False
        if SAFE.config_sha256(_read_json(output/'config.json'))!=CONFIG_SHA:return False
        selection=_read_json(output/'selection.json');rows=DATA._records(selection['records'])
        if (len(rows)!=16 or [row['label'] for row in rows]!=list(range(16)) or any(row['split']!='train' for row in rows)
                or selection.get('unique_images')!=16 or not _same_json(selection.get('counts'),COUNTS)
                or selection.get('manifest_sha256')!=DATA.MANIFEST_SHA or selection.get('selection_rule')!=SELECTION_RULE
                or not _same_json(selection,summary['selection'])):return False
        provenance=_read_json(output/'provenance.json')
        provenance_keys=('schema','input_scope','counts','source_identity','config_sha256','manifest_sha256','weights_sha256',
                         'weights_bytes','data_source','data_root_used','selection','runtime')
        if any(not _same_json(provenance[key],summary[key]) for key in provenance_keys):return False
        first=_read_json(output/'first_step.json');steps=_read_json(output/'steps.json')
        if (not isinstance(steps,list) or len(steps)!=4 or not _same_json(first,steps[0])
                or not _gradients_ok(first['gradients'],first['updated_groups'])
                or not _same_json(first['gradients'],summary['first_step_gradients'])
                or not _same_json(first['updated_groups'],summary['first_step_updated_groups'])):return False
        for index,event in enumerate(steps,1):
            expected=dict(stage='first_optimizer_step' if index==1 else 'measurement',optimizer_step_completed=True,
                optimizer_step_succeeded=True,optimizer_steps=index,step_index=index,phase='warmup' if index==1 else 'measured',
                training_started=True,formal_training_started=False,completed=False,batch_size=16,samples_seen=index*16,
                post_update_checks_passed=True)
            if (any(not _same_json(event.get(key),value) for key,value in expected.items())
                    or not _positive(event.get('elapsed_seconds')) or not _positive(event.get('loss'))
                    or not _same_json(_read_json(output/f'step_{index:02}.json'),event)):return False
        if not _same_json(_read_json(output/'optimizer_progress.json'),steps[-1]):return False
        measured=[event['elapsed_seconds'] for event in steps[1:]]
        if not _same_json(summary['step_seconds'],measured) or not _same_json(summary['warmup_seconds'],steps[0]['elapsed_seconds']):return False
        for key,expected in [('median_step_seconds',statistics.median(measured)),('mean_step_seconds',statistics.mean(measured)),
                             ('compute_images_per_second',16/statistics.median(measured))]:
            if not _positive(summary[key]) or not math.isclose(summary[key],expected,rel_tol=1e-12):return False
        if not _positive(summary['wall_seconds']):return False
        if (type(summary.get('cuda_peak_allocated_bytes')) is not int or summary['cuda_peak_allocated_bytes']<=0
                or type(summary.get('cuda_peak_reserved_bytes')) is not int
                or summary['cuda_peak_reserved_bytes']<summary['cuda_peak_allocated_bytes']):return False
        params=summary['parameters']
        if any(type(params.get(key)) is not int or params[key]!=74612163 for key in ('total','trainable')):return False
        optimizer=summary['optimizer']
        for key,value in dict(name='Adam',lr=1e-5,weight_decay=5e-5,betas=[.9,.999],eps=1e-8,amsgrad=False,foreach=None,fused=None).items():
            if not _same_json(optimizer.get(key),value):return False
        if final_progress:
            progress=_read_json(output/'progress.json')
            expected=dict(stage='complete',completed=True,training_started=True,formal_training_started=False,optimizer_steps=4,samples_seen=64)
            if any(not _same_json(progress.get(key),value) for key,value in expected.items()):return False
        return True
    except (OSError,ValueError,KeyError,TypeError,AttributeError,OverflowError,ZeroDivisionError):return False

def _run_model(torch,cfg,root,records,weights,output,progress):
    from external_baselines.hgtnet.model import HGTNet,parameter_counts
    from external_baselines.hgtnet.pretrained import load_pretrained_file
    from external_baselines.hgtnet.aid_batch_probe_engine import run_batch_probe
    dataset=DATA.make_dataset(root,records);samples=[dataset[index] for index in range(16)]
    images=torch.stack([item[0] for item in samples]);labels=torch.stack([item[1] for item in samples]).to(dtype=torch.long)
    model=HGTNet(cfg)
    counts=parameter_counts(model)
    if counts['total']!=74612163 or counts['trainable']!=74612163 or len(list(model.parameters()))!=451:
        raise ValueError('actual complete AID model parameter/tensor identity changed')
    pretraining=load_pretrained_file(model,weights,WEIGHTS_SHA);model.pretraining_loaded=True
    _write(output/'pretraining.json',pretraining)
    result=run_batch_probe(model,images,labels,device='cuda:0',batch_size=16,warmup_steps=1,measured_steps=3,on_progress=progress)
    result.update(parameters=parameter_counts(model),pretraining=pretraining)
    return result

def _end_checks(args,sources,records):
    if (_source_identity()!=sources or _sha(args.manifest)!=DATA.MANIFEST_SHA
            or _check_weights(args.weights)['weights_sha256']!=WEIGHTS_SHA
            or SAFE.config_sha256(_check_config(args.config))!=CONFIG_SHA):raise ValueError('probe sources/config/input changed')
    selected=DATA.validate_selected(args.data_root,records)
    if selected.get('checked_count')!=16 or selected.get('verified_count')!=16 or selected.get('test_image_bytes_read')!=0:
        raise ValueError('probe ending train16 byte verification incomplete')
    return dict(COMPLETION_CHECKS)

def run_probe(args):
    preflight=inspect_resources(args)
    if not preflight['ready']:raise ValueError('preflight blockers: '+', '.join(preflight['blockers']))
    selected=preflight['selected_gpu'];args.gpu_uuid=selected['uuid']
    with _run_lock(args.output):
        args.output.mkdir(exist_ok=False)
        _write(args.output/'preflight.json',preflight);_write(args.output/'command.json',dict(argv=list(sys.argv),cwd=os.getcwd(),pid=os.getpid(),command=run_command(args,args.gpu_uuid)))
        result=dict(schema=SCHEMA,action='run',completed=False,training_started=False,formal_training_started=False,
                    input_scope='real_aid',target_gpu='3060',counts=dict(COUNTS),seed=42,batch_size=16,optimizer_steps=0)
        started=time.perf_counter();events=[]
        with (args.output/'run.log').open('x') as log,contextlib.redirect_stdout(SAFE._Tee(sys.stdout,log)),contextlib.redirect_stderr(SAFE._Tee(sys.stderr,log)):
            try:
                sources=_source_identity()
                if sources!=preflight['source_identity']:raise ValueError('source changed after preflight')
                _write(args.output/'source_identity.json',sources)
                torch,runtime=_runtime_and_torch(args,selected);_write(args.output/'runtime.json',runtime)
                cfg=_check_config(args.config);records=select_probe_records(DATA.prepare_inputs(args.manifest))
                DATA.validate_selected(args.data_root,records);_write(args.output/'config.json',cfg)
                selection=dict(records=records,unique_images=16,counts=dict(COUNTS),manifest_sha256=DATA.MANIFEST_SHA,selection_rule=SELECTION_RULE)
                _write(args.output/'selection.json',selection)
                provenance=dict(schema=SCHEMA,input_scope='real_aid',counts=dict(COUNTS),source_identity=sources,
                    config_sha256=CONFIG_SHA,manifest_sha256=DATA.MANIFEST_SHA,weights_sha256=WEIGHTS_SHA,weights_bytes=WEIGHTS_BYTES,
                    data_source='verified_images',data_root_used=str(args.data_root),selection=selection,runtime=runtime)
                _write(args.output/'provenance.json',provenance)
                def progress(value):
                    value=dict(value,pid=os.getpid())
                    if value.get('optimizer_step_succeeded') is True or value.get('optimizer_step_completed') is True:
                        steps=value.get('optimizer_steps')
                        if type(steps) is not int or steps!=len(events)+1 or not 1<=steps<=4:raise ValueError('nonsequential actual optimizer event')
                        _write(args.output/f'step_{steps:02}.json',value);events.append(value)
                        _write(args.output/'steps.json',events,replace=(args.output/'steps.json').exists())
                        _write(args.output/'optimizer_progress.json',value,replace=True)
                        if steps==1:_write(args.output/'first_step.json',value)
                    _write(args.output/'progress.json',value,replace=True)
                    print('HGTNET_AID_BATCH16_PROBE_PROGRESS '+json.dumps(value,sort_keys=True),flush=True)
                engine=_run_model(torch,cfg,args.data_root,records,args.weights,args.output,progress)
                if engine.get('schema')!=SCHEMA or engine.get('completed') is not True or engine.get('optimizer_steps')!=4:raise ValueError('engine did not complete four AID updates')
                result.update(engine);result.update(provenance)
                result.update(target_gpu='3060',seed=42,physical_gpu_uuid=selected['uuid'],completion_checks=_end_checks(args,sources,records),
                    wall_seconds=time.perf_counter()-started,probe_scope=SCOPE,
                    timing_scope='after stdlib preflight; imports/input decode/initialization and four disposable updates')
                if not _completion_evidence_ok(args.output,result,final_progress=False):raise ValueError('complete AID batch16 evidence inconsistent')
                _write(args.output/'summary.json',result)
                _write(args.output/'progress.json',dict(stage='complete',completed=True,training_started=True,formal_training_started=False,optimizer_steps=4,samples_seen=64),replace=True)
                with (args.output/'run.exit').open('x') as stream:stream.write('0\n')
                print('HGTNET_AID_BATCH16_PROBE_COMPLETE '+str(args.output/'summary.json'),flush=True);return 0
            except BaseException as exc:
                observed=SAFE._observed_training(args.output);actual=getattr(exc,'optimizer_steps',0)
                observed['optimizer_steps']=max(observed['optimizer_steps'],len(events),actual if type(actual) is int else 0,result.get('optimizer_steps',0))
                observed['training_started']=observed['training_started'] or observed['optimizer_steps']>0 or getattr(exc,'training_started',False) is True
                result.update(observed,completed=False,formal_training_started=False,error=type(exc).__name__+': '+str(exc),wall_seconds=time.perf_counter()-started)
                traceback.print_exc();_write(args.output/'summary.json',result,replace=(args.output/'summary.json').exists())
                with (args.output/'run.exit').open('w') as stream:stream.write('1\n')
                _write(args.output/'progress.json',dict(stage='failed',completed=False,formal_training_started=False,error=result['error'],**observed),replace=True)
                return 1

def status(output):
    output=Path(output);report=dict(schema=SCHEMA,action='status',state='not_started',output=str(output),completed=False,formal_training_started=False,counts=dict(COUNTS))
    if not output.exists() and not output.is_symlink():return report
    report['state']='incomplete'
    try:summary=_read_json(output/'summary.json')
    except (OSError,ValueError):summary={}
    if not isinstance(summary,dict):summary={}
    try:report['progress']=_read_json(output/'progress.json')
    except (OSError,ValueError):pass
    try:
        with DATA.inspector.open_regular_readonly(output/'run.exit') as stream:report['exit_code']=int(stream.read().strip())
    except (OSError,ValueError):report['exit_code']=None
    for key in ('target_gpu','physical_gpu_uuid','config_sha256','manifest_sha256','weights_sha256','unique_images','processed_images',
                'batch_size','warmup_steps','measured_steps','step_seconds','median_step_seconds','compute_images_per_second',
                'cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','wall_seconds','error'):
        if key in summary:report[key]=summary[key]
    observed=SAFE._observed_training(output);steps=summary.get('optimizer_steps',0)
    report['optimizer_steps']=max(observed['optimizer_steps'],steps if type(steps) is int and steps>=0 else 0)
    report['training_started']=observed['training_started'] or summary.get('training_started') is True
    try:source_ok=summary.get('source_identity')==_source_identity() if summary else None
    except (OSError,ValueError,KeyError):source_ok=False
    report['source_identity_matches']=source_ok;report['completion_evidence_valid']=_completion_evidence_ok(output,summary)
    if report['completion_evidence_valid'] and report['exit_code']==0 and source_ok:report.update(state='complete',completed=True)
    if summary and source_ok is False:report['state']='source_changed'
    if report['exit_code'] not in (None,0):report['state']='failed'
    return report

def parser():
    result=argparse.ArgumentParser(description=__doc__);sub=result.add_subparsers(dest='action',required=True)
    for action in ('inspect','run','status'):
        command=sub.add_parser(action);command.add_argument('--output',type=Path,default=ROOT/'runs/batch16_seed42_01')
        if action=='status':continue
        command.add_argument('--gpu-uuid');command.add_argument('--manifest',type=Path,default=Path(DATA.MANIFEST))
        command.add_argument('--data-root',type=Path,default=Path(DATA.DATA_ROOT))
        command.add_argument('--weights',type=Path,default=Path('/home/featurize/work/swin_b-68c6b09e.pth'))
        command.add_argument('--config',type=Path,default=ROOT/'configs/hgtnet_aid_independent_v0.json')
    return result

def main(argv=None):
    args=parser().parse_args(argv)
    if args.action=='status':print(json.dumps(status(args.output),indent=2,sort_keys=True));return 0
    if args.action=='inspect':print(json.dumps(inspect_resources(args),indent=2,sort_keys=True));return 0
    return run_probe(args)
if __name__=='__main__':raise SystemExit(main())
