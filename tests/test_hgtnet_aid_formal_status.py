"""Consistent recorded-artifact fixtures; no GPU execution or tensor attestation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from external_baselines.hgtnet import aid_formal_entry as e


def write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def fixture(root):
    if not callable(getattr(e,'_completion_evidence_ok',None)):
        raise AssertionError('Formal complete-artifact validator missing')
    recipe=copy.deepcopy(e.PROTO.RECIPE);names=[f'class_{i:02}' for i in range(30)]
    records={split:[dict(path=f'class_00/{split}_{i:05}.jpg',label=0) for i in range(count)]
             for split,count in [('val',1000),('test',5000)]}
    sources=e._source_identity()
    binding=dict(schema='hgtnet-aid-formal-binding-v1',input_scope='real_aid',recipe=recipe,
        recipe_sha256=e.PROTO.recipe_sha256(recipe),counts=dict(e.PROTO.REAL_COUNTS),class_names=names,
        records=records,manifest_sha256=e.DATA.MANIFEST_SHA,config_sha256=e.CONFIG_SHA,
        weights_sha256=e.WEIGHTS_SHA,source_identity=dict(sources,formal_recipe_bytes=e._sha(e.ROOT/'configs/hgtnet_aid_formal_v1.json')))
    gpu='GPU-df386e6c-dbb3-0952-8190-3265a7bd8ae8'
    runtime=dict(python='3.11.8',torch='2.2.2+cu121',torchvision='0.17.2+cu121',numpy='1.26.4',pillow='10.3.0',
        cuda='12.1',cudnn=8902,physical_gpu_uuid=gpu,cuda_visible_devices=gpu,device='cuda:0',cpu_threads=8,
        amp=False,dtype='float32',seed=42,matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,
        gpu=dict(name='NVIDIA GeForce RTX 3060',uuid=gpu,memory_total_mib=12288))
    runtime['cuda_device']=e._check_cuda_device(dict(available=True,device_count=1,name=runtime['gpu']['name'],total_memory_bytes=12487622656),runtime['gpu'])
    def packet(split,epoch,correct):
        count=len(records[split]);labels=[0]*count;predictions=[0]*correct+[1]*(count-correct)
        p=dict(schema='hgtnet-aid-formal-evaluation-v1',phase='validation' if split=='val' else 'test',input_scope='real_aid',
            paths=[r['path'] for r in records[split]],labels=labels,predictions=predictions,
            metrics=e._metrics(labels,predictions,names,'formal_validation' if split=='val' else 'formal_test'),
            loss=1.0,seconds=1.0,manifest_sha256=binding['manifest_sha256'],recipe_sha256=binding['recipe_sha256'])
        if split=='val':p['epoch']=epoch
        return p
    best=packet('val',2,600);last=packet('val',100,600)
    cal=[]
    for epoch in range(1,101):
        p=packet('val',epoch,500 if epoch==1 else 600)
        cal.append(dict(epoch=epoch,train_loss=1.0,train_accuracy_percent=50.0,train_seconds=10.0,
            train_images_per_second=400.0,epoch_optimizer_steps=250,optimizer_steps=epoch*250,
            val_correct=500 if epoch==1 else 600,val_loss=1.0,validation_seconds=1.0,val_metrics=p['metrics'],epoch_precommit_seconds=11.0,epoch_wall_seconds=12.0))
    ref=[dict(epoch=i,train_loss=1.0,train_accuracy_percent=50.0,train_seconds=10.0,train_images_per_second=500.0,
        epoch_optimizer_steps=313,optimizer_steps=25000+313*i,epoch_precommit_seconds=10.0,epoch_wall_seconds=11.0) for i in (1,2)]
    checkpoints={}
    for key in ('best_calibration','final_calibration','final_refit'):
        path=root/(key+'.pt');path.write_bytes(b'recorded-byte-receipt-fixture-'+key.encode())
        receipt=dict(path=str(path.absolute()),bytes=path.stat().st_size,sha256=e._sha(path));checkpoints[key]=receipt
        write(Path(str(path)+'.json'),receipt)
    reload=dict(loaded=True,sha256=checkpoints['final_refit']['sha256'],reload_max_abs_error=0.0,reload_tolerance=1e-6,
        state_sha256='b'*64,metadata=dict(phase='refit',epoch=2,selected_epoch=2,initial_state_sha256='a'*64))
    test=packet('test',2,3000);test.update(selected_epoch=2,refit_checkpoint_sha256=checkpoints['final_refit']['sha256'])
    for file,value in [('binding.json',binding),('runtime.json',runtime),('source_identity.json',sources),
        ('initialization.json',dict(input_scope='real_aid',calibration_state_sha256='a'*64,refit_state_sha256='a'*64,equal=True)),
        ('history.json',dict(input_scope='real_aid',calibration=cal,refit=ref)),('best_val_predictions.json',best),
        ('last_val_predictions.json',last),('checkpoint_reload.json',reload),('test.json',test)]:write(root/file,value)
    transaction=dict(schema='hgtnet-aid-formal-test-transaction-v1',binding_sha256=e._binding_hash(binding),selected_epoch=2,
        refit_checkpoint_sha256=checkpoints['final_refit']['sha256'],state='complete',test_sha256=e._sha(root/'test.json'))
    write(root/'test_transaction.json',transaction)
    for phase,steps in [('calibration',1),('refit',25001)]:
        first=dict(stage='first_optimizer_step',phase=phase,epoch=1,optimizer_steps=steps,samples_seen=16,device='cuda:0',
            training_started=True,formal_training_started=True,optimizer_step_completed=True,optimizer_step_succeeded=True,
            post_update_checks_passed=True,loss=1.0,gradients=dict(passed=True,missing=[],nonfinite=[],groups={k:dict(tensors=n,l2=1.0) for k,n in e.GROUP_TENSORS.items()}),
            updated_groups={k:True for k in e.GROUP_TENSORS})
        write(root/('first_'+phase+'_step.json'),first)
    write(root/'optimizer_progress.json',dict(stage='training_epoch_complete',phase='refit',epoch=2,optimizer_steps=25626,
        training_started=True,formal_training_started=True,samples_seen=5000,optimizer_step_completed=True,
        optimizer_step_succeeded=True,post_update_checks_passed=True))
    checks=dict(source_identity=True,config=True,recipe=True,manifest=True,weights=True,pool_image_count=5000,test_image_count=5000,
        test_authorized=True,pool=dict(checked_count=5000,verified_count=5000,test_image_bytes_read=0,verified_bytes=1),
        test=dict(checked_count=5000,verified_count=5000,test_image_bytes_read=1,verified_bytes=1))
    write(root/'completion_checks.json',checks)
    summary=dict(schema=e.SCHEMA,input_scope='real_aid',completed=True,training_started=True,formal_training_started=True,
        counts=dict(e.PROTO.REAL_COUNTS),recipe=recipe,seed=42,batch_size=16,target_gpu='3060',physical_gpu_uuid=gpu,runtime=runtime,
        source_identity=sources,manifest_sha256=binding['manifest_sha256'],config_sha256=binding['config_sha256'],
        weights_sha256=binding['weights_sha256'],recipe_sha256=binding['recipe_sha256'],selected_epoch=2,best_epoch=2,final_epoch=100,
        refit_epochs=2,optimizer_steps=25626,initial_state_sha256='a'*64,refit_initial_state_sha256='a'*64,
        best_val_metrics=best['metrics'],calibration_history=cal,refit_history=ref,checkpoints=checkpoints,
        checkpoint_reload=reload,test_metrics=test['metrics'],test_path=str((root/'test.json').absolute()),test_sha256=transaction['test_sha256'],
        test_transaction=transaction,parameters=dict(total=74612163,trainable=74612163),parameter_tensors=451,
        completion_checks=checks,wall_seconds=100.0,engine_wall_seconds=99.0,cuda_peak_allocated_bytes=1000,cuda_peak_reserved_bytes=2000)
    write(root/'summary.json',summary);(root/'run.exit').write_text('0\n');(root/'completion.marker').write_text(e.MARKER)
    write(root/'progress.json',dict(stage='complete',completed=True,training_started=True,formal_training_started=True,optimizer_steps=25626))
    return summary


class StatusReceiptTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(dir='/private/tmp');self.addCleanup(t.cleanup);self.root=Path(t.name)

    def test_complete_consistent_receipts_accept_and_compact_metrics(self):
        fixture(self.root);report=e.status(self.root)
        self.assertTrue(report['completed'],report)
        self.assertTrue(report['completion_evidence_valid'])
        self.assertEqual(report['selected_epoch'],2)
        self.assertEqual(report['test_metrics']['accuracy_percent'],60.0)

    def test_earliest_tie_initial_state_and_source_mutations_rejected(self):
        original=fixture(self.root)
        for key,value in [('selected_epoch',3),('refit_initial_state_sha256','c'*64),('source_identity',{})]:
            bad=copy.deepcopy(original);bad[key]=value;write(self.root/'summary.json',bad)
            with self.subTest(key=key):self.assertFalse(e.status(self.root)['completed'])
        write(self.root/'summary.json',original)

    def test_checkpoint_firststep_and_transaction_corruption_rejected(self):
        fixture(self.root);path=self.root/'first_refit_step.json';saved=path.read_bytes();path.unlink()
        self.assertFalse(e.status(self.root)['completed']);path.write_bytes(saved)
        transaction=json.loads((self.root/'test_transaction.json').read_text());transaction['state']='started';write(self.root/'test_transaction.json',transaction)
        self.assertFalse(e.status(self.root)['completed'])
        transaction['state']='complete';write(self.root/'test_transaction.json',transaction)
        (self.root/'final_refit.pt').write_bytes(b'changed')
        self.assertFalse(e.status(self.root)['completed'])

    def test_self_consistent_hashes_cannot_hide_invalid_predictions(self):
        summary=fixture(self.root);test=json.loads((self.root/'test.json').read_text());test['predictions'][0]=30;write(self.root/'test.json',test)
        summary['test_sha256']=e._sha(self.root/'test.json');summary['test_transaction']['test_sha256']=summary['test_sha256']
        write(self.root/'test_transaction.json',summary['test_transaction']);write(self.root/'summary.json',summary)
        self.assertFalse(e.status(self.root)['completed'])
