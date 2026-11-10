"""Real small CPU tensors exercise the formal stages, selection and recovery."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler


torch.set_num_threads(1)

def engine():
    try:return importlib.import_module('external_baselines.hgtnet.aid_formal_engine')
    except ModuleNotFoundError as exc:raise AssertionError('formal stage engine is missing') from exc


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed=nn.Linear(3,3)
        self.stages=nn.ModuleList([nn.ModuleDict({'transform':nn.Linear(3,3),'da':nn.Linear(3,3)})])
        self.graph_layer=nn.Linear(3,3)
        self.head=nn.Linear(3,30)
        with torch.no_grad():self.head.bias.zero_();self.head.bias[0]=4
    def forward(self,x):
        x=self.patch_embed(x.mean((2,3)))
        x=self.stages[0]['da'](self.stages[0]['transform'](x))
        return self.head(self.graph_layer(x))


class FormalEngineTests(unittest.TestCase):
    def binding(self):
        p=importlib.import_module('external_baselines.hgtnet.aid_formal_protocol')
        r=copy.deepcopy(p.RECIPE);r.update(max_epochs=2,batch_size=2,eval_batch_size=2)
        return dict(schema='hgtnet-aid-formal-binding-v1',input_scope='cpu_fixture',recipe=r,recipe_sha256=p.recipe_sha256(r),
            class_names=['c%02d'%i for i in range(30)],counts=dict(train=2,val=2,pool=4,test=2),
            records={k:[dict(path=k+str(i),label=i) for i in range(2)] for k in ['val','test']},
            manifest_sha256='a'*64,config_sha256='b'*64,weights_sha256='c'*64,source_identity={'fixture.py':'d'*64})
    def loaders(self,binding,events,calls,*,bad_val_order=False):
        def factory(split,epoch):
            calls.append((split,epoch))
            if split=='test':
                self.assertTrue(any(v['stage']=='before_test' for v in events))
                self.assertTrue(any(v['stage']=='refit_complete' for v in events))
            count=binding['counts'][split];labels=torch.arange(count)%2
            if split=='val' and bad_val_order:labels=1-labels
            x=torch.arange(count,dtype=torch.float32).reshape(-1,1,1,1).expand(-1,3,224,224).clone()/10+0.2
            return DataLoader(TensorDataset(x,labels),batch_size=2,shuffle=split in ('train','pool'),
                generator=torch.Generator().manual_seed(42+epoch))
        return factory
    def run_fixture(self,path,*,resume=False,events=None,calls=None,callback=None,factory=Tiny,bad_val_order=False):
        b=self.binding();events=[] if events is None else events;calls=[] if calls is None else calls
        def progress(v):
            events.append(dict(v))
            if callback:callback(v)
        return engine().run_formal(factory,self.loaders(b,events,calls,bad_val_order=bad_val_order),path,
            binding=b,device='cpu',on_progress=progress,resume=resume)

    def test_earliest_tie_fresh_refit_and_complete_metrics_without_early_test(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d).resolve();events=[];calls=[];r=self.run_fixture(out,events=events,calls=calls)
            self.assertTrue(r['completed']);self.assertFalse(r['formal_training_started'])
            self.assertEqual((r['best_epoch'],r['final_epoch'],r['selected_epoch'],r['refit_epochs']),(1,2,1,1))
            self.assertEqual(r['optimizer_steps'],4)
            self.assertEqual(r['initial_state_sha256'],r['refit_initial_state_sha256'])
            self.assertEqual(len(r['calibration_history']),2);self.assertEqual(len(r['refit_history']),1)
            self.assertEqual(r['checkpoint_reload']['reload_max_abs_error'],0.0)
            self.assertEqual(r['test_metrics']['scope'],'cpu_fixture_test')
            self.assertEqual(r['test'].get('rows'),2)
            self.assertFalse(any(key in r['test'] for key in ['paths','labels','predictions']))
            self.assertEqual(len(r['test_metrics']['per_class']),30)
            self.assertEqual(sum(map(sum,r['test_metrics']['confusion_matrix'])),2)
            packet=json.loads((out/'test.json').read_text())
            self.assertEqual(packet['paths'],['test0','test1']);self.assertEqual(packet['labels'],[0,1])
            self.assertEqual([s for s,e in calls].count('test'),1)
            self.assertTrue(all((out/name).is_file() for name in ['best_calibration.pt','final_calibration.pt','final_refit.pt','history.json']))
            complete=[v for v in events if v['stage']=='epoch_complete']
            self.assertEqual(len(complete),3)
            self.assertEqual(complete[0].get('train_loss'),r['calibration_history'][0]['train_loss'])
            self.assertEqual(complete[0].get('val_accuracy_percent'),50.)
            self.assertEqual(complete[0].get('val_macro_f1_percent'),r['calibration_history'][0]['val_metrics']['macro_f1_percent'])
            self.assertEqual(complete[0].get('best_val_accuracy_percent'),50.)
            self.assertGreater(complete[0].get('estimated_remaining_phase_seconds',0),0)
            self.assertEqual(complete[-1].get('estimated_remaining_phase_seconds'),0)
            self.assertIn('billing',complete[0].get('estimate_scope',''))

    def test_completed_epoch_resume_matches_uninterrupted_model_and_selection(self):
        class Stop(RuntimeError):pass
        def stop(v):
            if v['stage']=='epoch_complete' and v['phase']=='calibration' and v['epoch']==1:raise Stop('simulate interruption after committed epoch')
        c=importlib.import_module('external_baselines.hgtnet.aid_formal_checkpoints')
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            full=Path(a).resolve();partial=Path(b).resolve();expected=self.run_fixture(full)
            with self.assertRaises(Stop):self.run_fixture(partial,callback=stop)
            events=[];actual=self.run_fixture(partial,resume=True,events=events)
            self.assertTrue(any(v['stage']=='resumed' and v['optimizer_steps']==1 for v in events))
            self.assertEqual(actual['selected_epoch'],expected['selected_epoch'])
            self.assertEqual(actual['optimizer_steps'],expected['optimizer_steps'])
            models=[]
            for folder,result in [(full,expected),(partial,actual)]:
                m=Tiny();receipt=result['checkpoints']['final_refit']
                c.load_model(m,folder/'final_refit.pt',expected_sha256=receipt['sha256'],binding=self.binding());models.append(c.state_sha256(m))
            self.assertEqual(models[0],models[1])
            self.assertEqual(actual['test_metrics'],expected['test_metrics'])

    def test_validation_order_failure_prevents_refit_and_test(self):
        with tempfile.TemporaryDirectory() as d:
            events=[];calls=[]
            with self.assertRaisesRegex(ValueError,'order|labels'):
                self.run_fixture(Path(d).resolve(),events=events,calls=calls,bad_val_order=True)
            self.assertFalse(any(split in ('pool','test') for split,epoch in calls))

    def test_validation_failure_preserves_every_completed_training_update(self):
        b=self.binding();b['counts'].update(train=4,pool=6);events=[];calls=[]
        with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(ValueError,'order|labels'):
            engine().run_formal(Tiny,self.loaders(b,events,calls,bad_val_order=True),Path(d).resolve(),
                binding=b,device='cpu',on_progress=events.append)
        successes=[v for v in events if v.get('optimizer_step_completed')]
        self.assertTrue(successes);self.assertEqual(successes[-1]['optimizer_steps'],2)
        self.assertEqual(successes[-1]['stage'],'training_epoch_complete')
        self.assertEqual(successes[-1]['samples_seen'],4)
        self.assertFalse(any(split in ('pool','test') for split,epoch in calls))

    def test_unequal_initialization_rejects_refit_before_pool_or_test(self):
        count=0
        def factory():
            nonlocal count
            count+=1;m=Tiny()
            if count>1:
                with torch.no_grad():m.head.weight.add_(1)
            return m
        with tempfile.TemporaryDirectory() as d:
            events=[];calls=[]
            with self.assertRaisesRegex(ValueError,'initial'):
                self.run_fixture(Path(d).resolve(),events=events,calls=calls,factory=factory)
            self.assertFalse(any(split in ('pool','test') for split,epoch in calls))

    def test_bad_numeric_reload_prevents_first_test_loader(self):
        def wrong(model,*a,**kw):
            with torch.no_grad():model.head.bias.add_(torch.arange(30,dtype=torch.float32))
            return {'loaded':True}
        with tempfile.TemporaryDirectory() as d:
            events=[];calls=[]
            with patch.object(engine(),'load_model',side_effect=wrong),self.assertRaisesRegex(ValueError,'reload'):
                self.run_fixture(Path(d).resolve(),events=events,calls=calls)
            self.assertFalse(any(split=='test' for split,epoch in calls))

    def test_post_update_failure_retains_successful_update_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            events=[]
            with patch.object(engine(),'_finite_parameters',side_effect=[True,False]),self.assertRaisesRegex(ValueError,'nonfinite'):
                self.run_fixture(Path(d).resolve(),events=events)
            observed=[v for v in events if v.get('optimizer_step_completed')]
            self.assertTrue(observed);self.assertEqual(observed[-1]['optimizer_steps'],1)
            self.assertFalse(observed[-1]['post_update_checks_passed'])

    def test_existing_training_artifacts_require_explicit_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d).resolve();(path/'history.json').write_text('user original')
            with self.assertRaises(FileExistsError):self.run_fixture(path)
            self.assertEqual((path/'history.json').read_text(),'user original')

    def test_existing_model_file_cannot_be_overwritten_by_direct_engine_call(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d).resolve();(path/'final_refit.pt').write_bytes(b'user original model')
            with self.assertRaises(FileExistsError):self.run_fixture(path)
            self.assertEqual((path/'final_refit.pt').read_bytes(),b'user original model')

    def test_replacement_sampler_rejected_before_first_update(self):
        b=self.binding();events=[];calls=[];base=self.loaders(b,events,calls)
        def loaders(split,epoch):
            loader=base(split,epoch)
            if split=='train':return DataLoader(loader.dataset,batch_size=2,
                sampler=RandomSampler(loader.dataset,replacement=True,num_samples=2))
            return loader
        with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(ValueError,'sampl|replacement'):
            engine().run_formal(Tiny,loaders,Path(d).resolve(),binding=b,device='cpu',on_progress=events.append)
        self.assertFalse(any(v.get('optimizer_step_completed') for v in events))

    def test_resume_at_phase_transition_restarts_refit_from_equal_initial_state(self):
        class Stop(RuntimeError):pass
        def stop(v):
            if v['stage']=='fresh_refit':raise Stop('interrupt before first refit epoch')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d).resolve();calls=[]
            with self.assertRaises(Stop):self.run_fixture(path,callback=stop,calls=calls)
            self.assertFalse(any(split in ('pool','test') for split,epoch in calls))
            result=self.run_fixture(path,resume=True)
            self.assertEqual(result['initial_state_sha256'],result['refit_initial_state_sha256'])
            self.assertEqual(result['refit_epochs'],1);self.assertTrue(result['completed'])

    def test_recovery_selection_metadata_cannot_change_validation_winner(self):
        class Stop(RuntimeError):pass
        def stop(v):
            if v['stage']=='epoch_complete' and v['phase']=='calibration' and v['epoch']==1:raise Stop()
        c=importlib.import_module('external_baselines.hgtnet.aid_formal_checkpoints')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d).resolve()
            with self.assertRaises(Stop):self.run_fixture(path,callback=stop)
            payload,_=c.load_snapshot(path,self.binding());payload['best_correct']+=1
            c.save_snapshot(path,payload)
            with self.assertRaisesRegex(ValueError,'selection'):
                self.run_fixture(path,resume=True)

    def test_failed_newbest_snapshot_cannot_publish_uncommitted_validation_predictions(self):
        class Stop(RuntimeError):pass
        module=engine();evaluate=module._evaluate;save=module.save_snapshot;validation_calls=0
        def controlled_evaluation(model,loader,binding,split,device):
            nonlocal validation_calls
            packet,reference=evaluate(model,loader,binding,split,device)
            if split=='val':
                validation_calls+=1
                packet['predictions']=[0,1] if validation_calls==2 else [1,0]
                packet['metrics']=module._metrics(packet['labels'],packet['predictions'],binding['class_names'])
                packet['metrics']['scope']='cpu_fixture_validation'
            return packet,reference
        def interrupted_snapshot(output,payload):
            if payload['phase']=='calibration' and payload['epoch']==2:raise Stop('newbest before snapshot commit')
            return save(output,payload)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d).resolve()
            with patch.object(module,'_evaluate',side_effect=controlled_evaluation):
                with patch.object(module,'save_snapshot',side_effect=interrupted_snapshot),self.assertRaises(Stop):
                    self.run_fixture(path)
                uncommitted=json.loads((path/'best_val_predictions.json').read_text())
                self.assertEqual(sum(a==b for a,b in zip(uncommitted['labels'],uncommitted['predictions'])),2)
                result=self.run_fixture(path,resume=True)
            self.assertEqual(result['best_epoch'],1)
            published=json.loads((path/'best_val_predictions.json').read_text())
            self.assertEqual(sum(a==b for a,b in zip(published['labels'],published['predictions'])),0)
            self.assertEqual(published.get('epoch'),1)

    def test_recovery_best_validation_packet_semantics_are_checked_before_training(self):
        class Stop(RuntimeError):pass
        def stop(v):
            if v['stage']=='epoch_complete' and v['phase']=='calibration' and v['epoch']==1:raise Stop()
        c=importlib.import_module('external_baselines.hgtnet.aid_formal_checkpoints')
        for case in ['missing','epoch','paths','labels','predictions','metrics']:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as d:
                path=Path(d).resolve()
                with self.assertRaises(Stop):self.run_fixture(path,callback=stop)
                payload,_=c.load_snapshot(path,self.binding());packet=payload['best_validation_packet']
                if case=='missing':payload.pop('best_validation_packet')
                elif case=='epoch':packet['epoch']=2
                elif case=='paths':packet['paths'].reverse()
                elif case=='labels':packet['labels'].reverse()
                elif case=='predictions':packet['predictions'].pop()
                elif case=='metrics':packet['metrics']['accuracy_percent']=99.
                c.save_snapshot(path,payload);events=[]
                failure=None
                try:self.run_fixture(path,resume=True,events=events)
                except Exception as exc:failure=exc
                self.assertIsInstance(failure,ValueError)
                self.assertIn('validation packet',str(failure))
                self.assertFalse(any(v.get('optimizer_step_completed') for v in events))

    def test_real_binding_cannot_be_executed_as_cpu_fixture(self):
        b=self.binding();b['input_scope']='real_aid'
        with tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError):
            engine().run_formal(Tiny,lambda *a:None,Path(d).resolve(),binding=b,device='cpu')


class AIDTransactionTests(unittest.TestCase):
    binding=FormalEngineTests.binding
    loaders=FormalEngineTests.loaders
    run_fixture=FormalEngineTests.run_fixture
    # These cases catch reevaluation after any durable test-start record.
    def test_complete_test_resume_reuses_packet_and_immutable_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d).resolve();first=self.run_fixture(path);calls=[]
            model_bytes=(path/'final_refit.pt').read_bytes();test_bytes=(path/'test.json').read_bytes()
            transaction=json.loads((path/'test_transaction.json').read_text())
            self.assertEqual(transaction['state'],'complete')
            self.assertEqual(transaction['test_sha256'],first['test_sha256'])
            actual=self.run_fixture(path,resume=True,calls=calls)
            self.assertFalse(any(s=='test' for s,e in calls))
            self.assertEqual((path/'final_refit.pt').read_bytes(),model_bytes)
            self.assertEqual((path/'test.json').read_bytes(),test_bytes)
            self.assertEqual(actual['test_sha256'],first['test_sha256'])

    def test_started_without_packet_refuses_automatic_retest(self):
        module=engine();evaluate=module._evaluate
        def interrupted(model,loader,binding,split,device):
            if split=='test':raise RuntimeError('test interrupted')
            return evaluate(model,loader,binding,split,device)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d).resolve()
            with patch.object(module,'_evaluate',side_effect=interrupted),self.assertRaisesRegex(RuntimeError,'test interrupted'):
                self.run_fixture(path)
            self.assertEqual(json.loads((path/'test_transaction.json').read_text())['state'],'started')
            calls=[]
            with self.assertRaisesRegex(ValueError,'test|Test'):
                self.run_fixture(path,resume=True,calls=calls)
            self.assertFalse(any(s=='test' for s,e in calls))

    def test_packet_committed_before_complete_transaction_recovers_without_retest(self):
        module=engine();atomic=module.atomic_json
        def interrupted(path,value):
            if Path(path).name=='test_transaction.json' and value['state']=='complete':
                raise RuntimeError('receipt commit interrupted')
            return atomic(path,value)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d).resolve()
            with patch.object(module,'atomic_json',side_effect=interrupted),self.assertRaisesRegex(RuntimeError,'receipt commit interrupted'):
                self.run_fixture(path)
            calls=[];packet=(path/'test.json').read_bytes();self.run_fixture(path,resume=True,calls=calls)
            self.assertFalse(any(s=='test' for s,e in calls))
            self.assertEqual((path/'test.json').read_bytes(),packet)
            self.assertEqual(json.loads((path/'test_transaction.json').read_text())['state'],'complete')

    def test_tampered_packet_or_transaction_is_not_retested(self):
        for case in ['predictions','metric','path','binding','checkpoint','sha','missing_transaction']:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as d:
                path=Path(d).resolve();self.run_fixture(path);calls=[]
                packet=json.loads((path/'test.json').read_text());receipt=json.loads((path/'test_transaction.json').read_text())
                if case=='predictions':packet['predictions'][0]=30
                elif case=='metric':packet['metrics']['accuracy_percent']=99
                elif case=='path':packet['paths'].reverse()
                elif case=='binding':receipt['binding_sha256']='0'*64
                elif case=='checkpoint':receipt['refit_checkpoint_sha256']='0'*64
                elif case=='sha':receipt['test_sha256']='0'*64
                if case in ['predictions','metric','path']:engine().atomic_json(path/'test.json',packet)
                else:engine().atomic_json(path/'test_transaction.json',receipt)
                if case=='missing_transaction':(path/'test_transaction.json').unlink()
                with self.assertRaisesRegex(ValueError,'test|Test'):
                    self.run_fixture(path,resume=True,calls=calls)
                self.assertFalse(any(s=='test' for s,e in calls))

    def test_missing_group_rejected_before_optimizer_update(self):
        class HeadOnly(nn.Module):
            def __init__(self):super().__init__();self.head=nn.Linear(3,30)
            def forward(self,x):return self.head(x.mean((2,3)))
        with tempfile.TemporaryDirectory() as d:
            events=[]
            with self.assertRaisesRegex(ValueError,'group'):
                self.run_fixture(Path(d).resolve(),factory=HeadOnly,events=events)
            self.assertFalse(any(e.get('optimizer_step_completed') for e in events))

    def test_sync_failure_after_optimizer_return_preserves_observed_update(self):
        calls=0
        def sync(device):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('post-optimizer sync failed')
        with tempfile.TemporaryDirectory() as d:
            events=[]
            with patch.object(engine(),'_sync',side_effect=sync),self.assertRaisesRegex(RuntimeError,'sync failed'):
                self.run_fixture(Path(d).resolve(),events=events)
            self.assertEqual(events[-1]['optimizer_steps'],1)
            self.assertTrue(events[-1]['optimizer_step_completed'])

    def test_45_class_output_rejected_before_first_update(self):
        def factory():
            model=Tiny();model.head=nn.Linear(3,45);return model
        with tempfile.TemporaryDirectory() as d:
            events=[]
            with self.assertRaises(ValueError):self.run_fixture(Path(d).resolve(),factory=factory,events=events)
            self.assertFalse(any(e.get('optimizer_step_completed') for e in events))

    def test_no_epoch_snapshot_refuses_resume(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(ValueError,'snapshot'):
            self.run_fixture(Path(d).resolve(),resume=True)


if __name__=="__main__":unittest.main()
