"""Offline independent HGTNet formal NWPU stages; inspection imports no ML."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import run_hgtnet_nwpu_smoke as COMMON
from external_baselines.hgtnet import formal_data as DATA, formal_protocol as PROTO
RES=COMMON.RES
SCHEMA='hgtnet-nwpu-formal-v1'
MARKER='HGTNET_NWPU_FORMAL_COMPLETE\n'

def _write(path,value):
    path=Path(path)
    if path.is_symlink():raise ValueError('Owned result must not be a symbolic link')
    temporary=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            if isinstance(value,str):stream.write(value)
            else:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)

def _json(path):
    path=Path(path)
    if path.is_symlink():raise ValueError('Owned metadata must not be a symbolic link')
    return json.loads(path.read_text())

def _binding(args,inputs):
    recipe=PROTO.load_recipe(args.recipe)
    sources=dict(COMMON._source_identity(),formal_recipe_bytes=COMMON._sha(args.recipe))
    return PROTO.validate_binding(dict(schema='hgtnet-formal-binding-v1',input_scope='real_nwpu',
        class_names=inputs['class_names'],counts=inputs['counts'],recipe=recipe,
        recipe_sha256=PROTO.recipe_sha256(recipe),manifest_sha256=DATA.MANIFEST_SHA,
        config_sha256=COMMON.CONFIG_SHA,weights_sha256=COMMON.WEIGHTS_SHA,source_identity=sources,
        records={split:[dict(path=r['path'],label=r['label']) for r in inputs['selected'][split]] for split in ['val','test']}))

def _binding_hash(binding):return hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
def _resume(args):return args.action=='resume' or getattr(args,'resume',False)

def _owner(args,binding):
    return dict(schema=SCHEMA,binding_sha256=_binding_hash(binding),
        output=str(args.output.resolve()),scratch_root=str(args.scratch_root.resolve()))

def _require_owned(args,binding):
    if args.output.is_symlink() or args.scratch_root.is_symlink() or not args.output.is_dir():
        raise ValueError('Resume requires an existing owned output directory')
    if _json(args.output/'binding.json')!=binding or _json(args.output/'run_identity.json')!=_owner(args,binding):
        raise ValueError('Resume binding/source/recipe/paths differ from the owned run')
    if status(args.output)['completed']:raise ValueError('Owned formal run is already complete')
    if args.scratch_root.exists() and (not args.scratch_root.is_dir() or _json(args.scratch_root/'.formal_owner.json')!=_owner(args,binding)):
        raise ValueError('Existing scratch is not owned by this run')
    valid=[]
    for slot in [0,1]:
        path=args.output/'recovery'/('slot%d.pt'%slot)
        try:
            if path.is_symlink() or path.parent.is_symlink():continue
            receipt=_json(str(path)+'.json')
            if (receipt['binding_sha256']==_binding_hash(binding) and type(receipt['generation']) is int
                    and receipt['generation']>0 and receipt['phase'] in ('calibration','refit')
                    and type(receipt['epoch']) is int and 1<=receipt['epoch']<=100
                    and COMMON._sha(path)==receipt['sha256']):valid.append(receipt)
        except (OSError,ValueError,KeyError,TypeError):pass
    if not valid:raise ValueError('No SHA-valid compatible completed-epoch snapshot; no restart is inferred')
    return max(valid,key=lambda row:row['generation'])

def _capacity(output,scratch):
    result=dict(ready=False,blockers=[],same_volume=None)
    for name,path,budget in [('output',output,6*2**30),('scratch',scratch,2*2**30)]:
        try:result[name]=RES._capacity_path(path,budget)
        except (OSError,ValueError,RuntimeError) as exc:
            result[name]=dict(path=str(path),required_bytes=budget,error=str(exc));result['blockers'].append('disk_capacity:'+name)
    if not result['blockers']:
        a,b=result['output'],result['scratch'];result['same_volume']=a['device']==b['device']
        if result['same_volume']:
            if min(a['free_bytes'],b['free_bytes'])<8*2**30:result['blockers'].append('disk_capacity:combined')
        else:
            for name in ['output','scratch']:
                if result[name]['free_bytes']<result[name]['required_bytes']:result['blockers'].append('disk_capacity:'+name)
    result['ready']=not result['blockers'];return result

def run_command(args,current_uuid):
    entry=os.environ.get('HGTNET_DELIVERY_ENTRY',str(Path(__file__).resolve()))
    command=[sys.executable,entry,'resume' if _resume(args) else 'run','--gpu-uuid',current_uuid,'--target-gpu','3090']
    for name in ['manifest','data_root','archive','weights','config','recipe','output','scratch_root']:
        command+=['--'+name.replace('_','-'),str(getattr(args,name))]
    return shlex.join(command)

def inspect_resources(args):
    blockers=[];report=dict(schema=SCHEMA,action='inspect',ready=False,formal_training_started=False,
        target_gpu='3090',requested_gpu_uuid=args.gpu_uuid,resume=_resume(args),output=str(args.output),
        scratch_root=str(args.scratch_root),blockers=blockers,
        readiness_scope='stdlib metadata/full candidate hash; actual run checks imports/CUDA, archive and snapshot tensors')
    if args.target_gpu!='3090' or os.environ.get('HGTNET_TARGET_GPU','3090')!='3090':blockers.append('target_profile')
    try:
        if COMMON.config_sha256(COMMON.load_config(args.config))!=COMMON.CONFIG_SHA:raise ValueError('Frozen model config changed')
        report['config_sha256']=COMMON.CONFIG_SHA
    except (OSError,ValueError,KeyError,TypeError) as exc:blockers.append('config');report['config_error']=str(exc)
    try:
        inputs=DATA.prepare_inputs(args.manifest,expected_sha256=DATA.MANIFEST_SHA);binding=_binding(args,inputs)
        report.update(counts=inputs['counts'],manifest_sha256=DATA.MANIFEST_SHA,recipe_sha256=binding['recipe_sha256'],
            max_calibration_epochs=100,batch_size=16,selection=PROTO.RECIPE['selection'])
        try:DATA.validate_selected(args.data_root,inputs['selected']['pool']);report['pool_data_source']='verified_images'
        except (OSError,ValueError):report['pool_data_source']='archive_recovery_pending'
        if args.archive.is_file() and not args.archive.is_symlink() and args.archive.stat().st_size>0:
            report.update(archive_available=True,archive_sha_verified=False,archive_expected_sha256=DATA.ARCHIVE_SHA,
                archive_bytes=args.archive.stat().st_size,test_data_source='archive_deferred_until_final_refit')
        else:
            if report['pool_data_source']!='verified_images':raise ValueError('Full pool6300 images or persistent ZIP required')
            DATA.validate_selected(args.data_root,inputs['selected']['test'])
            report.update(archive_available=False,test_data_source='verified_existing_bytes_not_decoded')
    except (OSError,ValueError,KeyError,TypeError) as exc:blockers.append('inputs');report['input_error']=str(exc)
    try:
        if args.weights.is_symlink() or args.weights.stat().st_size!=COMMON.WEIGHTS_BYTES or COMMON._sha(args.weights)!=COMMON.WEIGHTS_SHA:
            raise ValueError('Full pinned official Swin-B candidate bytes required')
        report.update(weights_sha256=COMMON.WEIGHTS_SHA,weights_bytes=COMMON.WEIGHTS_BYTES)
    except (OSError,ValueError) as exc:blockers.append('weights');report['weights_error']=str(exc)
    report['dependencies']=RES.dependency_inventory();blockers.extend(RES.dependency_blockers(report['dependencies']))
    report['nvidia']=RES.inventory_gpu()
    try:report['selected_gpu']=RES.select_gpu(args.gpu_uuid,report['nvidia'],target_name='NVIDIA GeForce RTX 3090')
    except ValueError as exc:blockers.append('target_gpu');report['gpu_error']=str(exc)
    try:
        if _resume(args):report['resume_snapshot_receipt']=_require_owned(args,binding)
        else:RES.require_new_paths(args.output,args.scratch_root)
    except (OSError,ValueError,KeyError,TypeError,UnboundLocalError) as exc:
        blockers.append('resume_ownership' if _resume(args) else 'output_or_scratch');report['path_error']=str(exc)
    report['disk']=_capacity(args.output,args.scratch_root);blockers.extend(report['disk']['blockers'])
    report['ready']=not blockers
    if report['ready']:report['run_command']=run_command(args,report['selected_gpu']['uuid'])
    return report

def _assert_unchanged(args,inputs,binding,*,include_test=False):
    if (_binding(args,inputs)!=binding or COMMON._sha(args.manifest)!=DATA.MANIFEST_SHA
            or COMMON._sha(args.weights)!=COMMON.WEIGHTS_SHA
            or COMMON.config_sha256(COMMON.load_config(args.config))!=COMMON.CONFIG_SHA):
        raise ValueError('Formal source/config/recipe/manifest/initial weights changed')
    if getattr(args,'_archive_verified',False) and COMMON._sha(args.archive)!=DATA.ARCHIVE_SHA:
        raise ValueError('Pinned archive changed during formal run')
    DATA.validate_selected(args._pool_root,inputs['selected']['pool'])
    if include_test:DATA.validate_selected(args._test_root,inputs['selected']['test'])

def _observed_steps(output):
    try:return max(0,int(_json(Path(output)/'optimizer_progress.json')['optimizer_steps']))
    except (OSError,ValueError,TypeError,KeyError):return 0

def _run_model(args,torch,runtime,inputs,binding,progress,resume):
    from external_baselines.hgtnet.model import HGTNet,parameter_counts
    from external_baselines.hgtnet.pretrained import load_pretrained_file
    from external_baselines.hgtnet.formal_engine import run_formal
    datasets={};factory_calls=0;parameters=None
    def model_factory():
        nonlocal factory_calls,parameters
        model=HGTNet(COMMON.load_config(args.config));pretraining=load_pretrained_file(model,args.weights,COMMON.WEIGHTS_SHA)
        if pretraining.get('mode')!='applied' or pretraining.get('source_identity_verified') is not True:
            raise ValueError('Verified candidate values were not actually loaded')
        model.pretraining_loaded=True;parameters=parameter_counts(model);factory_calls+=1
        _write(args._attempt/('pretraining_%03d.json'%factory_calls),pretraining);return model
    def loader_factory(split,epoch):
        if split not in ('train','val','pool','test'):raise ValueError('Unknown formal split')
        if split=='test' and not getattr(args,'_test_authorized',False):raise ValueError('Test loader requested before completed refit/reload gate')
        if split not in datasets:
            root=args._pool_root
            if split=='test':
                try:DATA.validate_selected(args.data_root,inputs['selected']['test']);root=args.data_root
                except (OSError,ValueError):
                    target=args.scratch_root/'test'
                    if target.exists() and not (target/'recovery.json').exists():target=args.scratch_root/('test_attempt_'+args._attempt.name)
                    root=DATA.recover_selected(args.archive,inputs['selected']['test'],target,
                        expected_sha256=DATA.ARCHIVE_SHA,resume=target.exists())
                args._test_root=Path(root)
            datasets[split]=DATA.make_dataset(root,inputs['selected'][split],training=split in ('train','pool'))
        return torch.utils.data.DataLoader(datasets[split],batch_size=16,shuffle=split in ('train','pool'),
            generator=torch.Generator().manual_seed(42+epoch),num_workers=0,pin_memory=False,drop_last=False)
    result=run_formal(model_factory,loader_factory,args.output,binding=binding,device='cuda:0',on_progress=progress,resume=resume)
    result.update(parameters=parameters,pretraining_factory_calls=factory_calls);return result

def run_formal_entry(args,resume=False):
    if not args.gpu_uuid:raise ValueError('Explicit current physical --gpu-uuid required')
    args.resume=resume or args.action=='resume'
    args._test_authorized=False
    if hasattr(args,'_test_root'):del args._test_root
    preflight=inspect_resources(args)
    if not preflight['ready']:raise ValueError('Formal preflight blockers: '+', '.join(preflight['blockers']))
    inputs=DATA.prepare_inputs(args.manifest,expected_sha256=DATA.MANIFEST_SHA);binding=_binding(args,inputs)
    if not args.resume:
        args.output.mkdir(parents=True,exist_ok=False);_write(args.output/'binding.json',binding);_write(args.output/'run_identity.json',_owner(args,binding))
    attempts=args.output/'attempts'
    if attempts.is_symlink():raise ValueError('Attempts directory must not be a symbolic link')
    attempts.mkdir(exist_ok=True);number=1
    while (attempts/('%04d'%number)).exists():number+=1
    args._attempt=attempts/('%04d'%number);args._attempt.mkdir()
    for name in ['summary.json','progress.json','run.exit','completion.marker','optimizer_progress.json']:
        old=args.output/name
        if old.is_symlink():raise ValueError('Owned result must not be a symbolic link')
        if old.is_file():shutil.copyfile(old,args._attempt/('previous_'+name))
    _write(args._attempt/'preflight.json',preflight)
    _write(args._attempt/'command.json',dict(argv=list(sys.argv),cwd=os.getcwd(),pid=os.getpid(),command=run_command(args,args.gpu_uuid)))
    result=dict(schema=SCHEMA,completed=False,training_started=_observed_steps(args.output)>0,formal_training_started=_observed_steps(args.output)>0,
        optimizer_steps=_observed_steps(args.output),input_scope='real_nwpu',counts=inputs['counts'],seed=42,batch_size=16,
        target_gpu='3090',recipe_sha256=binding['recipe_sha256'],config_sha256=COMMON.CONFIG_SHA,
        manifest_sha256=DATA.MANIFEST_SHA,weights_sha256=COMMON.WEIGHTS_SHA,attempt=number)
    _write(args.output/'summary.json',result);_write(args.output/'run.exit','running\n');started=time.perf_counter()
    with (args._attempt/'run.log').open('x') as log,contextlib.redirect_stdout(COMMON._Tee(sys.stdout,log)),contextlib.redirect_stderr(COMMON._Tee(sys.stderr,log)):
        try:
            torch,runtime=COMMON._runtime_and_torch(args,preflight['selected_gpu']);_write(args._attempt/'runtime.json',runtime)
            if not args.scratch_root.exists():args.scratch_root.mkdir(parents=True,exist_ok=False);_write(args.scratch_root/'.formal_owner.json',_owner(args,binding))
            args._archive_verified=False
            if preflight['archive_available']:
                if COMMON._sha(args.archive)!=DATA.ARCHIVE_SHA:raise ValueError('Persistent ZIP full SHA mismatch before formal training')
                args._archive_verified=True
            try:DATA.validate_selected(args.data_root,inputs['selected']['pool']);args._pool_root=args.data_root
            except (OSError,ValueError):
                target=args.scratch_root/'pool'
                if target.exists() and not (target/'recovery.json').exists():target=args.scratch_root/('pool_attempt_'+args._attempt.name)
                args._pool_root=DATA.recover_selected(args.archive,inputs['selected']['pool'],target,
                    expected_sha256=DATA.ARCHIVE_SHA,resume=target.exists())
            _write(args._attempt/'data_provenance.json',dict(pool_root=str(args._pool_root),archive_sha_verified=args._archive_verified,
                archive_sha256=DATA.ARCHIVE_SHA if args._archive_verified else None,test_decoded=False))
            _assert_unchanged(args,inputs,binding)
            def progress(value):
                value=dict(value,pid=os.getpid(),attempt=number)
                if value.get('stage')=='before_test':_assert_unchanged(args,inputs,binding);args._test_authorized=True
                if value.get('optimizer_step_succeeded') is True or value.get('optimizer_step_completed') is True or value.get('stage')=='resumed':
                    _write(args.output/'optimizer_progress.json',value)
                    if value.get('optimizer_step_succeeded') is True and not (args.output/'first_step.json').exists():_write(args.output/'first_step.json',value)
                _write(args.output/'progress.json',value);print('HGTNET_FORMAL_PROGRESS '+json.dumps(value,sort_keys=True),flush=True)
            result.update(_run_model(args,torch,runtime,inputs,binding,progress,args.resume))
            if result.get('completed') is not True or not getattr(args,'_test_authorized',False) or not hasattr(args,'_test_root'):
                raise ValueError('Formal engine must complete refit/reload/final-test gates')
            _assert_unchanged(args,inputs,binding,include_test=True)
            result.update(schema=SCHEMA,runtime=runtime,physical_gpu_uuid=args.gpu_uuid,completed=True,
                training_started=True,formal_training_started=True,wall_seconds=time.perf_counter()-started,
                wall_scope='current attempt after stdlib preflight; includes setup/training/validation/checkpoints/refit/test/checks',
                archive_sha_verified=args._archive_verified,output=str(args.output))
            _write(args.output/'summary.json',result);_write(args.output/'run.exit','0\n');_write(args.output/'completion.marker',MARKER)
            progress(dict(stage='complete',completed=True,training_started=True,formal_training_started=True,optimizer_steps=result['optimizer_steps']))
            print('HGTNET_NWPU_FORMAL_COMPLETE '+str(args.output/'summary.json'),flush=True);return 0
        except BaseException as exc:
            steps=_observed_steps(args.output);result.update(completed=False,training_started=steps>0,formal_training_started=steps>0,
                optimizer_steps=steps,error=type(exc).__name__+': '+str(exc),wall_seconds=time.perf_counter()-started)
            traceback.print_exc();_write(args.output/'summary.json',result);_write(args.output/'run.exit','1\n')
            _write(args.output/'progress.json',dict(stage='failed',completed=False,training_started=steps>0,formal_training_started=steps>0,
                optimizer_steps=steps,error=result['error'],attempt=number));return 1

def status(output):
    output=Path(output);result=dict(schema=SCHEMA,state='incomplete' if output.exists() else 'not_started',output=str(output),completed=False)
    try:summary=_json(output/'summary.json')
    except (OSError,ValueError):summary={}
    try:result['progress']=_json(output/'progress.json')
    except (OSError,ValueError):pass
    try:result['exit_code']=int((output/'run.exit').read_text())
    except (OSError,ValueError):result['exit_code']=None
    for key in ['training_started','formal_training_started','input_scope','counts','batch_size','target_gpu','optimizer_steps',
                'selected_epoch','best_epoch','final_epoch','refit_epochs','checkpoint','checkpoints','checkpoint_reload','parameters',
                'test_path','test_sha256','initial_state_sha256','refit_initial_state_sha256','engine_wall_seconds','memory_scope','timing_scope',
                'cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','wall_seconds','physical_gpu_uuid',
                'config_sha256','recipe_sha256','manifest_sha256','weights_sha256','attempt','error']:
        if key in summary:result[key]=summary[key]
    for split in ['calibration','refit','test']:
        value=summary.get(split+'_metrics',summary.get(split,{}))
        if isinstance(value,dict):
            if isinstance(value.get('metrics'),dict):value=value['metrics']
            compact={k:v for k,v in value.items() if k in ['accuracy_percent','macro_f1_percent','loss','correct','samples','seconds','scope']}
            if compact:result[split+'_metrics']=compact
    if isinstance(summary.get('best_val_metrics'),dict):
        result['best_val_metrics']={k:v for k,v in summary['best_val_metrics'].items() if k in ['accuracy_percent','macro_f1_percent','scope']}
    try:marker=(output/'completion.marker').read_text()==MARKER
    except OSError:marker=False
    if summary.get('completed') is True and result['exit_code']==0 and marker:result.update(state='complete',completed=True)
    elif result['exit_code'] not in (None,0):result['state']='failed'
    return result

def parser():
    result=argparse.ArgumentParser(description=__doc__);sub=result.add_subparsers(dest='action',required=True)
    for action in ['inspect','run','resume','status']:
        command=sub.add_parser(action);command.add_argument('--output',type=Path,default=ROOT/'runs/nwpu_seed42_01')
        if action=='status':continue
        if action=='inspect':command.add_argument('--resume',action='store_true')
        command.add_argument('--target-gpu',choices=['3090'],default='3090');command.add_argument('--gpu-uuid')
        command.add_argument('--manifest',type=Path,default=DATA.MANIFEST);command.add_argument('--data-root',type=Path,default=DATA.DATA_ROOT)
        command.add_argument('--archive',type=Path,default=DATA.ARCHIVE);command.add_argument('--weights',type=Path,default=ROOT.parent/'swin_b-68c6b09e.pth')
        command.add_argument('--config',type=Path,default=ROOT/'configs/hgtnet_independent_v0.json')
        command.add_argument('--recipe',type=Path,default=ROOT/'configs/hgtnet_nwpu_formal_v1.json')
        command.add_argument('--scratch-root',type=Path,default=Path('/tmp')/(ROOT.name+'_rtx3090_seed42_01'))
    return result

def main(argv=None):
    args=parser().parse_args(argv)
    if args.action in ('run','resume'):return run_formal_entry(args,resume=args.action=='resume')
    print(json.dumps(status(args.output) if args.action=='status' else inspect_resources(args),indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
