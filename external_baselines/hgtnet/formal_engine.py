"""Frozen calibration, equal-initial-state refit and delayed final evaluation."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import torch
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler

from .formal_protocol import validate_binding
from .formal_checkpoints import (atomic_json, state_sha256, save_snapshot, load_snapshot,
                                apply_model_state, save_model, load_model, validate_optimizer)
from .nwpu_engine import _batch, _logits, _metrics, _sync
from .smoke import gradient_report, _group


def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _finite_parameters(model):
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def _fresh(factory, seed):
    torch.manual_seed(seed)
    model=factory()
    if not list(model.parameters()) or any(not p.requires_grad or p.dtype!=torch.float32 or p.is_meta for p in model.parameters()):
        raise ValueError('Entire actual model must be trainable FP32')
    if not _finite_parameters(model):raise ValueError('Initial parameters nonfinite')
    return model,state_sha256(model)


def _optimizer(model,recipe):
    return torch.optim.Adam(model.parameters(),lr=recipe['lr'],weight_decay=recipe['weight_decay'],
        betas=tuple(recipe['betas']),eps=recipe['eps'],amsgrad=False,foreach=None,fused=None)


def _loader(factory, split, epoch, binding):
    loader=factory(split,epoch)
    expected=binding['counts'][split];recipe=binding['recipe']
    batch=recipe['batch_size'] if split in ('train','pool') else recipe['eval_batch_size']
    if (len(loader.dataset)!=expected or loader.batch_size!=batch or loader.drop_last
            or loader.num_workers!=0 or loader.pin_memory):
        raise ValueError('Formal loader count/batch/workers/drop/pinning changed: '+split)
    sampler=RandomSampler if split in ('train','pool') else SequentialSampler
    if not isinstance(loader.sampler,sampler):raise ValueError('Formal sample order/sampler changed: '+split)
    if split in ('train','pool') and (loader.sampler.replacement or loader.sampler.num_samples!=expected):
        raise ValueError('Training sampler must consume each image once without replacement')
    if split in ('val','test'):
        records=binding['records'][split]
        for name,key in [('paths','path'),('labels','label')]:
            if hasattr(loader.dataset,name) and tuple(getattr(loader.dataset,name))!=tuple(row[key] for row in records):
                raise ValueError('Evaluation dataset order differs from bound records: '+split)
    return loader


def _notify(callback,state,stage,**extra):
    event=dict(stage=stage,optimizer_steps=state['optimizer_steps'],training_started=state['optimizer_steps']>0,
        formal_training_started=state['binding']['input_scope']=='real_nwpu' and state['optimizer_steps']>0,
        completed=False,**extra)
    if callback:callback(event)
    return event


def _train_epoch(model,optimizer,loader,*,state,phase,epoch,device,callback,diagnose):
    model.train();seen=correct=steps=0;loss_sum=0.;first=None;last_post_check=True
    _sync(device);started=time.perf_counter()
    try:
        for images,labels in loader:
            images,labels=_batch(images,labels,45,device)
            optimizer.zero_grad(set_to_none=True)
            logits=_logits(model,images,45);loss=nn.functional.cross_entropy(logits,labels)
            if not torch.isfinite(loss).item():raise ValueError('Training loss nonfinite')
            loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            if not math.isfinite(float(norm)):raise ValueError('Training gradients nonfinite')
            if diagnose and steps==0:
                gradients=gradient_report(model)
                if not gradients['passed']:raise ValueError('First batch gradients missing, zero or nonfinite')
                probes={}
                for name,p in model.named_parameters():
                    group=_group(name)
                    if group not in probes and p.grad is not None and p.grad.count_nonzero().item():
                        probes[group]=(name,p.detach().clone())
            optimizer.step();_sync(device)
            state['optimizer_steps']+=1;steps+=1;seen+=len(images)
            loss_sum+=float(loss.detach())*len(images);correct+=int((logits.argmax(1)==labels).sum())
            if diagnose and steps==1:
                named=dict(model.named_parameters())
                updates={group:not torch.equal(value,named[name]) for group,(name,value) in probes.items()}
                del probes
                last_post_check=False
                try:
                    if set(updates)!=set(gradients['groups']) or not all(updates.values()):
                        raise ValueError('First batch did not update every gradient group')
                    if not _finite_parameters(model):raise ValueError('Updated parameters nonfinite')
                    last_post_check=True
                finally:
                    first=_notify(callback,state,'first_optimizer_step',phase=phase,epoch=epoch,
                        optimizer_step_completed=True,optimizer_step_succeeded=True,
                        post_update_checks_passed=last_post_check,samples_seen=seen,
                        gradients=gradients,updated_groups=updates,loss=float(loss.detach()),device=str(device))
            elif steps%30==0:
                _notify(callback,state,'training',phase=phase,epoch=epoch,samples_seen=seen,
                    optimizer_step_completed=True,optimizer_step_succeeded=True)
        _sync(device);seconds=time.perf_counter()-started
        expected=state['binding']['counts']['train' if phase=='calibration' else 'pool']
        if seen!=expected or steps!=math.ceil(expected/loader.batch_size):
            raise ValueError('Training did not consume every bound record once')
        if not _finite_parameters(model):raise ValueError('Epoch final parameters nonfinite')
    except BaseException:
        if steps:
            _notify(callback,state,'last_successful_update_before_failure',phase=phase,epoch=epoch,
                samples_seen=seen,optimizer_step_completed=True,optimizer_step_succeeded=True,
                post_update_checks_passed=last_post_check,attempt_failed=True)
        raise
    optimizer.zero_grad(set_to_none=True)
    _notify(callback,state,'training_epoch_complete',phase=phase,epoch=epoch,samples_seen=seen,
        epoch_optimizer_steps=steps,optimizer_step_completed=True,optimizer_step_succeeded=True,
        post_update_checks_passed=True,train_seconds=seconds)
    return dict(epoch=epoch,train_loss=loss_sum/seen,train_accuracy_percent=100*correct/seen,
        train_seconds=seconds,train_images_per_second=seen/seconds,epoch_optimizer_steps=steps,
        optimizer_steps=state['optimizer_steps']),first


def _evaluate(model,loader,binding,split,device):
    model.eval();labels=[];predictions=[];loss_sum=0.;reference=None
    _sync(device);started=time.perf_counter()
    with torch.no_grad():
        for images,target in loader:
            images,target=_batch(images,target,45,device);logits=_logits(model,images,45)
            loss_sum+=float(nn.functional.cross_entropy(logits,target))*len(target)
            labels.extend(target.cpu().tolist());predictions.extend(logits.argmax(1).cpu().tolist())
            if reference is None:reference=(images.detach().cpu(),logits.detach().cpu())
    _sync(device);seconds=time.perf_counter()-started
    if labels!=[row['label'] for row in binding['records'][split]]:
        raise ValueError('Evaluation order/labels differ from bound records: '+split)
    metrics=_metrics(labels,predictions,binding['class_names'])
    metrics['scope']=('cpu_fixture_' if binding['input_scope']=='cpu_fixture' else 'formal_')+('validation' if split=='val' else 'test')
    packet=dict(schema='hgtnet-formal-evaluation-v1',phase='validation' if split=='val' else 'test',
        input_scope=binding['input_scope'],paths=[row['path'] for row in binding['records'][split]],
        labels=labels,predictions=predictions,metrics=metrics,loss=loss_sum/len(labels),seconds=seconds,
        manifest_sha256=binding['manifest_sha256'],recipe_sha256=binding['recipe_sha256'])
    return packet,reference


def _publish_best(receipt,destination):
    source=Path(receipt['path']);destination=Path(destination)
    if any(p.is_symlink() for p in [source,*source.parents,destination,*destination.parents]):
        raise ValueError('Model publication paths must not be symlinks')
    if _sha(source)!=receipt['sha256']:raise ValueError('Best checkpoint SHA mismatch')
    if source.resolve()==destination.resolve():return dict(receipt)
    temporary=destination.with_name(destination.name+'.tmp.'+uuid.uuid4().hex)
    try:
        with source.open('rb') as src,temporary.open('xb') as dst:
            shutil.copyfileobj(src,dst,1024*1024);dst.flush();os.fsync(dst.fileno())
        if _sha(temporary)!=receipt['sha256']:raise ValueError('Copied best checkpoint SHA mismatch')
        os.replace(temporary,destination)
    finally:temporary.unlink(missing_ok=True)
    result=dict(receipt,path=str(destination.absolute()),bytes=destination.stat().st_size)
    atomic_json(str(destination)+'.json',result)
    return result


def _history_file(output,state):
    atomic_json(output/'history.json',dict(input_scope=state['binding']['input_scope'],
        calibration=state['calibration_history'],refit=state['refit_history']))


def _phase_estimate(history,budget,epoch):
    observed=[row.get('epoch_wall_seconds',row['epoch_precommit_seconds']) for row in history]
    mean=sum(observed)/len(observed)
    return dict(observed_epoch_wall_mean_seconds=mean,estimated_remaining_phase_seconds=max(0,budget-epoch)*mean,
        estimate_scope='running completed-epoch mean for current phase only; recovered last commit may use precommit time excluding snapshot IO; excludes other phases/startup/final test/billing; not guaranteed')


def _snapshot(output,state,model,optimizer,phase,epoch):
    state.update(generation=state['generation']+1,phase=phase,epoch=epoch)
    payload=dict(state,schema='hgtnet-formal-snapshot-v1',model_state=model.state_dict(),optimizer_state=optimizer.state_dict())
    receipt=save_snapshot(output,payload)
    _history_file(output,state)
    return receipt


def _validate_best_packet(payload,binding,best):
    packet=payload.get('best_validation_packet');rows=binding['records']['val']
    if (not isinstance(packet,dict) or packet.get('schema')!='hgtnet-formal-evaluation-v1'
            or packet.get('epoch')!=payload['best_epoch'] or packet.get('phase')!='validation'
            or packet.get('input_scope')!=binding['input_scope']
            or packet.get('manifest_sha256')!=binding['manifest_sha256']
            or packet.get('recipe_sha256')!=binding['recipe_sha256']
            or packet.get('paths')!=[row['path'] for row in rows]
            or packet.get('labels')!=[row['label'] for row in rows]
            or any(type(label) is not int for label in packet.get('labels',[]))):
        raise ValueError('Recovery best validation packet identity/order mismatch')
    predictions=packet.get('predictions')
    if (not isinstance(predictions,list) or len(predictions)!=len(rows)
            or any(type(value) is not int or not 0<=value<45 for value in predictions)):
        raise ValueError('Recovery best validation packet predictions invalid')
    correct=sum(a==b for a,b in zip(packet['labels'],predictions))
    metrics=_metrics(packet['labels'],predictions,binding['class_names'])
    metrics['scope']=('cpu_fixture_' if binding['input_scope']=='cpu_fixture' else 'formal_')+'validation'
    if (correct!=payload['best_correct'] or correct!=best['val_correct']
            or packet.get('metrics')!=metrics or payload['best_metrics']!=metrics or best['val_metrics']!=metrics):
        raise ValueError('Recovery best validation packet metrics/selection mismatch')


def _validate_recovery(payload,binding):
    recipe=binding['recipe'];cal=payload['calibration_history'];refit=payload['refit_history']
    phase=payload['phase'];epoch=payload['epoch'];maximum=recipe['max_epochs']
    if phase not in ('calibration','refit') or not 1<=epoch<=(maximum if phase=='calibration' else maximum):
        raise ValueError('Recovery phase/epoch invalid')
    if ([r['epoch'] for r in cal]!=list(range(1,len(cal)+1)) or [r['epoch'] for r in refit]!=list(range(1,len(refit)+1))
            or len(cal)>maximum or payload['generation']!=len(cal)+len(refit)
            or (phase=='calibration' and (len(cal)!=epoch or refit))
            or (phase=='refit' and (len(cal)!=maximum or len(refit)!=epoch))):
        raise ValueError('Recovery history/phase/generation mismatch')
    best=max(cal,key=lambda row:row['val_correct'])
    if payload['best_epoch']!=best['epoch'] or payload['best_correct']!=best['val_correct']:
        raise ValueError('Recovery selection differs from earliest maximum validation count')
    _validate_best_packet(payload,binding,best)
    selected=best['epoch'] if len(cal)==maximum else None
    if payload['selected_epoch']!=selected or (phase=='refit' and epoch>selected):
        raise ValueError('Recovery refit selected epoch mismatch')
    train_steps=math.ceil(binding['counts']['train']/recipe['batch_size'])
    pool_steps=math.ceil(binding['counts']['pool']/recipe['batch_size'])
    expected=len(cal)*train_steps+len(refit)*pool_steps
    if payload['optimizer_steps']!=expected:raise ValueError('Recovery optimizer step count mismatch')
    for index,row in enumerate(cal,1):
        if row['optimizer_steps']!=index*train_steps:raise ValueError('Recovery calibration history counters mismatch')
    for index,row in enumerate(refit,1):
        if row['optimizer_steps']!=maximum*train_steps+index*pool_steps:raise ValueError('Recovery refit history counters mismatch')
    if type(payload.get('best_improvement_count')) is not int or not 1<=payload['best_improvement_count']<=len(cal):
        raise ValueError('Recovery best-checkpoint alternation missing')
    for receipt in payload['checkpoints'].values():
        if _sha(receipt['path'])!=receipt['sha256']:raise ValueError('Recovery referenced model checkpoint SHA mismatch')


def run_formal(model_factory,loader_factory,output,*,binding,device,on_progress=None,resume=False):
    """Run the frozen whole method. CPU execution is always explicitly a fixture."""
    binding=validate_binding(binding);recipe=binding['recipe'];output=Path(output)
    real=binding['input_scope']=='real_nwpu'
    if str(torch.device(device))!=('cuda:0' if real else 'cpu'):
        raise ValueError('Actual formal/fixture device differs from explicit input scope')
    output.mkdir(parents=True,exist_ok=True)
    if not resume and (any((output/name).exists() or (output/name).is_symlink() for name in ['history.json','initialization.json','test.json','recovery'])
            or any(output.glob('*.pt')) or any(output.glob('*.pt.json'))):
        raise FileExistsError('Training artifacts exist; use explicit compatible resume')
    started=time.perf_counter();model,initial=_fresh(model_factory,recipe['seed']);model.to(device)
    if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    state=dict(binding=binding,generation=0,phase='calibration',epoch=0,optimizer_steps=0,
        calibration_history=[],refit_history=[],best_epoch=None,best_correct=-1,best_metrics=None,best_validation_packet=None,
        selected_epoch=None,initial_state_sha256=initial,refit_initial_state_sha256=None,
        best_improvement_count=0,checkpoints={})
    optimizer=_optimizer(model,recipe);resumed=None
    if resume:
        payload,resumed=load_snapshot(output,binding);_validate_recovery(payload,binding)
        if payload['initial_state_sha256']!=initial:raise ValueError('Recovered initial model identity changed')
        if payload['phase']=='refit' and payload['refit_initial_state_sha256']!=initial:
            raise ValueError('Recovered refit initial state differs')
        apply_model_state(model,payload['model_state']);optimizer.load_state_dict(payload['optimizer_state'])
        validate_optimizer(optimizer,recipe)
        state={k:v for k,v in payload.items() if k not in ('model_state','optimizer_state','schema')}
        _notify(on_progress,state,'resumed',phase=state['phase'],epoch=state['epoch'],recovery=resumed)
    else:atomic_json(output/'initialization.json',dict(input_scope=binding['input_scope'],calibration_state_sha256=initial))
    diagnose=True
    if state['phase']=='calibration':
        for epoch in range(state['epoch']+1,recipe['max_epochs']+1):
            epoch_started=time.perf_counter();torch.manual_seed(recipe['seed']+epoch)
            loader=_loader(loader_factory,'train',epoch,binding)
            record,first=_train_epoch(model,optimizer,loader,state=state,phase='calibration',epoch=epoch,
                device=device,callback=on_progress,diagnose=diagnose);diagnose=False
            packet,_=_evaluate(model,_loader(loader_factory,'val',epoch,binding),binding,'val',device)
            packet['epoch']=epoch
            correct=sum(a==b for a,b in zip(packet['labels'],packet['predictions']))
            record.update(val_loss=packet['loss'],validation_seconds=packet['seconds'],val_correct=correct,val_metrics=packet['metrics'])
            atomic_json(output/'last_val_predictions.json',packet)
            if correct>state['best_correct']:
                state.update(best_epoch=epoch,best_correct=correct,best_metrics=packet['metrics'],best_validation_packet=packet,
                    best_improvement_count=state['best_improvement_count']+1)
                slot=state['best_improvement_count']%2
                state['checkpoints']['best_calibration']=save_model(model,output/('best_calibration_slot%d.pt'%slot),
                    binding=binding,metadata=dict(phase='calibration',epoch=epoch,val_correct=correct))
                atomic_json(output/'best_val_predictions.json',packet)
            state['calibration_history'].append(record)
            if epoch==recipe['max_epochs']:
                state['selected_epoch']=state['best_epoch']
                state['checkpoints']['final_calibration']=save_model(model,output/'final_calibration.pt',binding=binding,
                    metadata=dict(phase='calibration',epoch=epoch,best_epoch=state['best_epoch']))
            record['epoch_precommit_seconds']=time.perf_counter()-epoch_started
            _snapshot(output,state,model,optimizer,'calibration',epoch)
            record['epoch_wall_seconds']=time.perf_counter()-epoch_started
            _history_file(output,state)
            _notify(on_progress,state,'epoch_complete',phase='calibration',epoch=epoch,selected_epoch=state['best_epoch'],
                train_loss=record['train_loss'],train_accuracy_percent=record['train_accuracy_percent'],
                train_images_per_second=record['train_images_per_second'],val_accuracy_percent=packet['metrics']['accuracy_percent'],
                val_macro_f1_percent=packet['metrics']['macro_f1_percent'],best_val_accuracy_percent=state['best_metrics']['accuracy_percent'],
                train_seconds=record['train_seconds'],validation_seconds=record['validation_seconds'],epoch_wall_seconds=record['epoch_wall_seconds'],
                **_phase_estimate(state['calibration_history'],recipe['max_epochs'],epoch))
        atomic_json(output/'best_val_predictions.json',state['best_validation_packet'])
        state['checkpoints']['best_calibration']=_publish_best(state['checkpoints']['best_calibration'],output/'best_calibration.pt')
        optimizer.zero_grad(set_to_none=True);del optimizer
        model.cpu();del model
        model,refit_initial=_fresh(model_factory,recipe['seed'])
        if refit_initial!=initial:raise ValueError('Fresh refit initial model differs from calibration initial state')
        state['refit_initial_state_sha256']=refit_initial;model.to(device);optimizer=_optimizer(model,recipe)
        state.update(phase='refit',epoch=0);diagnose=True
        atomic_json(output/'initialization.json',dict(input_scope=binding['input_scope'],calibration_state_sha256=initial,refit_state_sha256=refit_initial,equal=True))
        _notify(on_progress,state,'fresh_refit',phase='refit',epoch=0,selected_epoch=state['selected_epoch'],initial_state_sha256=refit_initial)
    for epoch in range(state['epoch']+1,state['selected_epoch']+1):
        epoch_started=time.perf_counter();torch.manual_seed(recipe['seed']+epoch)
        record,first=_train_epoch(model,optimizer,_loader(loader_factory,'pool',epoch,binding),state=state,
            phase='refit',epoch=epoch,device=device,callback=on_progress,diagnose=diagnose);diagnose=False
        state['refit_history'].append(record)
        if epoch==state['selected_epoch']:
            state['checkpoints']['final_refit']=save_model(model,output/'final_refit.pt',binding=binding,
                metadata=dict(phase='refit',epoch=epoch,selected_epoch=state['selected_epoch'],initial_state_sha256=initial))
        record['epoch_precommit_seconds']=time.perf_counter()-epoch_started
        _snapshot(output,state,model,optimizer,'refit',epoch)
        record['epoch_wall_seconds']=time.perf_counter()-epoch_started;_history_file(output,state)
        _notify(on_progress,state,'epoch_complete',phase='refit',epoch=epoch,selected_epoch=state['selected_epoch'],
            train_loss=record['train_loss'],train_accuracy_percent=record['train_accuracy_percent'],
            train_images_per_second=record['train_images_per_second'],train_seconds=record['train_seconds'],
            epoch_wall_seconds=record['epoch_wall_seconds'],**_phase_estimate(state['refit_history'],state['selected_epoch'],epoch))
    _notify(on_progress,state,'refit_complete',phase='refit',epoch=state['selected_epoch'])
    optimizer.zero_grad(set_to_none=True);del optimizer
    model.eval();loader=_loader(loader_factory,'val',state['selected_epoch'],binding)
    images,labels=next(iter(loader));images,labels=_batch(images,labels,45,device)
    with torch.no_grad():reference=_logits(model,images,45).detach().cpu()
    reference_images=images.detach().cpu();del images
    expected_state=state_sha256(model);model.cpu();del model
    fresh,_=_fresh(model_factory,recipe['seed']);receipt=state['checkpoints']['final_refit']
    reload_record=load_model(fresh,receipt['path'],expected_sha256=receipt['sha256'],binding=binding)
    if state_sha256(fresh)!=expected_state:raise ValueError('Independent checkpoint reload state differs')
    fresh.to(device);fresh.eval()
    with torch.no_grad():actual=_logits(fresh,reference_images.to(device),45).cpu()
    error=float((actual-reference).abs().max())
    if not math.isfinite(error) or error>1e-6:raise ValueError('Independent checkpoint numeric reload differs')
    reload_record.update(reload_max_abs_error=error,reload_tolerance=1e-6,state_sha256=expected_state)
    atomic_json(output/'checkpoint_reload.json',reload_record)
    atomic_json(output/'best_val_predictions.json',state['best_validation_packet'])
    _notify(on_progress,state,'before_test',phase='test',epoch=state['selected_epoch'],checkpoint_reload=reload_record)
    test,_=_evaluate(fresh,_loader(loader_factory,'test',state['selected_epoch'],binding),binding,'test',device)
    test.update(refit_checkpoint_sha256=receipt['sha256'],selected_epoch=state['selected_epoch'])
    atomic_json(output/'test.json',test)
    test_sha=_sha(output/'test.json')
    test_summary=dict(rows=len(test['labels']),path=str((output/'test.json').absolute()),sha256=test_sha,
        metrics=test['metrics'],loss=test['loss'],seconds=test['seconds'],input_scope=binding['input_scope'])
    _sync(device)
    is_cuda=torch.device(device).type=='cuda'
    return dict(schema='hgtnet-nwpu-formal-result-v1',completed=True,training_started=state['optimizer_steps']>0,
        formal_training_started=real,input_scope=binding['input_scope'],counts=binding['counts'],recipe=recipe,
        best_epoch=state['best_epoch'],final_epoch=recipe['max_epochs'],selected_epoch=state['selected_epoch'],
        refit_epochs=state['selected_epoch'],optimizer_steps=state['optimizer_steps'],
        initial_state_sha256=initial,refit_initial_state_sha256=state['refit_initial_state_sha256'],
        best_val_metrics=state['best_metrics'],calibration_history=state['calibration_history'],refit_history=state['refit_history'],
        checkpoints=state['checkpoints'],checkpoint_reload=reload_record,test=test_summary,test_metrics=test['metrics'],
        test_path=str((output/'test.json').absolute()),test_sha256=test_sha,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if is_cuda else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if is_cuda else None,
        memory_scope='whole current engine invocation including diagnostics/checkpoint reload; resumed sessions separate',
        engine_wall_seconds=time.perf_counter()-started,resumed_from=resumed,
        timing_scope='epoch train includes reads/augmentation/transfer/diagnostics/updates; epoch wall also validation/checkpoint/history IO; not billed time')
