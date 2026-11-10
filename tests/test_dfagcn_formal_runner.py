"""Formal wrapper file/protocol checks; no simulated CUDA result is a GPU claim."""
import builtins
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import types
import unittest
from unittest import mock
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def api():
    path=ROOT/'tools/run_dfagcn_nwpu_formal.py'
    if not path.exists():raise AssertionError('formal DFAGCN wrapper is not implemented')
    spec=importlib.util.spec_from_file_location('formal_runner_under_test',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def manifest_fixture():
    names=[f'class_{i:02}' for i in range(45)]
    records=[]
    for label,name in enumerate(names):
        for split,count in [('train',112),('val',28),('test',560)]:
            records += [{'path':f'{name}/{split}_{i:03}.jpg','label':label,'split':split,'sha256':'1'*64}
                        for i in range(count)]
    return {'schema_version':1,'dataset':'NWPU-RESISC45','class_names':names,
            'counts':{'train':5040,'val':1260,'test':25200},'records':records}


class FormalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def resources(self,module):
        manifest=self.root/'manifest.json';manifest.write_text(json.dumps(manifest_fixture()))
        weights=self.root/'vgg16-397923af.pth';weights.write_bytes(b'file-identity-fixture-only')
        metadata=self.root/'weights.json';metadata.write_text(json.dumps({'sha256':hashlib.sha256(weights.read_bytes()).hexdigest()}))
        recipe=self.root/'recipe.json';shutil.copyfile(ROOT/'configs/dfagcn_independent_v1.json',recipe)
        archive=self.root/'data.zip'
        with zipfile.ZipFile(archive,'w') as z:z.writestr('placeholder.txt','resource inventory fixture')
        binaries=[]
        for name in ('train','predict'):
            path=self.root/name;path.write_text('#!/bin/sh\nexit 0\n');path.chmod(0o700);binaries.append(path)
        args=module.parser().parse_args(['run','--gpu-uuid',module.GPU.TARGET_UUID,
            '--manifest',str(manifest),'--data-root',str(self.root/'missing-data'),'--archive',str(archive),
            '--weights',str(weights),'--weights-metadata',str(metadata),'--recipe',str(recipe),
            '--train-binary',str(binaries[0]),'--predict-binary',str(binaries[1]),
            '--scratch-root',str(self.root/'scratch'),
            '--output',str(self.root/'formal-run'),'--background-log',str(self.root/'background.log')])
        device={'name':'NVIDIA GeForce RTX 3080','uuid':module.GPU.TARGET_UUID,
                'memory_total_mib':10240,'memory_free_mib':9874,'driver_version':'610.57.04'}
        stack=contextlib.ExitStack()
        stack.enter_context(mock.patch.object(module,'MANIFEST_SHA',hashlib.sha256(manifest.read_bytes()).hexdigest()))
        stack.enter_context(mock.patch.object(module.GPU,'WEIGHTS_SHA',hashlib.sha256(weights.read_bytes()).hexdigest()))
        stack.enter_context(mock.patch.object(module,'dependency_inventory',return_value={
            'python':'3.11.8',**{k:{'located':True,'version':v,'import_tested':False} for k,v in module.VERSIONS.items()}}))
        stack.enter_context(mock.patch.object(module.GPU,'gpu_inventory',return_value={'gpus':[device],'processes':[],'error':None}))
        stack.enter_context(mock.patch.object(module.shutil,'disk_usage',return_value=types.SimpleNamespace(total=100*2**30,used=20*2**30,free=80*2**30)))
        self.addCleanup(stack.close)
        return args

    def test_full_pool_protocol_and_fixed_recipe(self):
        module=api();selected=module.select_full(manifest_fixture())
        self.assertEqual({name:len(rows) for name,rows in selected.items()},
                         {'train':5040,'val':1260,'pool':6300,'test':25200})
        self.assertFalse(any(row['split']=='test' for row in selected['pool']))
        self.assertEqual(module.recipe_bytes(ROOT/'configs/dfagcn_independent_v1.json')['epochs'],100)
        changed=self.root/'recipe.json';value=json.loads((ROOT/'configs/dfagcn_independent_v1.json').read_text())
        value['learning_rate']=0.01;changed.write_text(json.dumps(value))
        with self.assertRaises(ValueError):module.recipe_bytes(changed)

    def test_inspect_is_stdlib_only_and_prints_exclusive_background_command(self):
        module=api();args=self.resources(module)
        original=builtins.__import__
        def guard(name,*a,**kw):
            if name.split('.')[0] in {'torch','torchvision','numpy','PIL'}:raise AssertionError(name)
            return original(name,*a,**kw)
        with mock.patch.object(builtins,'__import__',side_effect=guard):report=module.inspect_resources(args)
        self.assertTrue(report['ready'],report)
        self.assertEqual(report['counts'],{'train':5040,'val':1260,'pool':6300,'test':25200})
        self.assertEqual(report['disk']['minimum_gate_bytes'],40*2**30)
        self.assertEqual(report['disk']['feature_cache_bytes'],10102579200)
        self.assertIn('set -C',report['nohup_command'])
        self.assertIn('exec nohup',report['nohup_command'])
        self.assertFalse(args.output.exists())
        args.background_log.write_text('old log')
        second=module.inspect_resources(args)
        self.assertTrue(second['ready'])
        self.assertNotIn('nohup_command',second)
        self.assertEqual(args.background_log.read_text(),'old log')

    def test_inspect_rejects_low_disk_busy_gpu_versions_and_existing_output(self):
        module=api();args=self.resources(module)
        for kind in ('space','busy','version','output','uuid','threads'):
            with contextlib.ExitStack() as stack:
                check=copy.copy(args)
                if kind=='space':stack.enter_context(mock.patch.object(module.shutil,'disk_usage',return_value=types.SimpleNamespace(total=50*2**30,used=20*2**30,free=30*2**30)))
                elif kind=='busy':stack.enter_context(mock.patch.object(module.GPU,'gpu_inventory',return_value={'gpus':[],'processes':[{'gpu_uuid':module.GPU.TARGET_UUID,'pid':123}],'error':None}))
                elif kind=='version':stack.enter_context(mock.patch.object(module,'dependency_inventory',return_value={'python':'3.11.8','torch':{'located':True,'version':'2.2.2','import_tested':False}}))
                elif kind=='output':check.output.mkdir()
                elif kind=='uuid':check.gpu_uuid='GPU-other'
                else:check.cpu_threads=1
                with self.subTest(kind=kind):self.assertFalse(module.inspect_resources(check)['ready'])
                if kind=='output':check.output.rmdir()

    def test_generated_nohup_command_runs_and_cannot_replace_existing_log(self):
        module=api();args=self.resources(module)
        # Substitute only the launched executable: actual shell/redirection are
        # the boundary under test, and /bin/echo cannot train or initialize CUDA.
        with mock.patch.object(module.sys,'executable','/bin/echo'),mock.patch.dict(os.environ,{'DFAGCN_DELIVERY_ENTRY':'FORMAL_BACKGROUND_BOUNDARY'}):
            command=module.inspect_resources(args)['nohup_command']
        def launch_and_wait():
            run=subprocess.run(['/bin/sh','-c',command],capture_output=True,text=True,timeout=5)
            self.assertEqual(run.returncode,0,run.stderr)
            for unused in range(100):
                if args.background_log.exists() and args.background_log.stat().st_size:break
                time.sleep(0.01)
        launch_and_wait()
        self.assertIn('FORMAL_BACKGROUND_BOUNDARY run --gpu-uuid',args.background_log.read_text())
        args.background_log.write_text('preserve prior launch evidence\n')
        launch_and_wait()
        time.sleep(0.1)
        self.assertEqual(args.background_log.read_text(),'preserve prior launch evidence\n')
        self.assertFalse(args.output.exists())

    def test_run_requires_explicit_uuid_and_status_is_read_only(self):
        module=api()
        with mock.patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit):module.parser().parse_args(['run'])
        report=module.status(self.root/'absent')
        self.assertEqual(report['state'],'not_started')
        self.assertFalse((self.root/'absent').exists())

    def test_new_bundle_reuses_complete_previous_native_build_only(self):
        module=api();module.ROOT=self.root/'new_fix_bundle'
        previous=self.root/'dfagcn_nwpu_formal_3080_20261030/liblinear/bin'
        previous.mkdir(parents=True)
        for name in ('train','predict'):(previous/name).write_bytes(b'native-file-fixture')
        args=module.parser().parse_args(['inspect'])
        self.assertEqual(args.train_binary,previous/'train')
        self.assertEqual(args.predict_binary,previous/'predict')
        (previous/'predict').unlink()
        incomplete=module.parser().parse_args(['inspect'])
        self.assertEqual(incomplete.train_binary,module.ROOT/'liblinear/bin/train')
        self.assertEqual(incomplete.predict_binary,module.ROOT/'liblinear/bin/predict')

    def test_actual_float32_cache_slices_preserve_pool_order_without_vgg(self):
        module=api()
        import torch
        from external_baselines.dfagcn.cache import FeatureCache,write_feature_cache
        from external_baselines.dfagcn.identity import build_cache_identity
        rows=[{'path':f'class_00/image_{i}.jpg','sha256':str(i+1)*64} for i in range(4)]
        def identity(records):
            return build_cache_identity(manifest_sha256='1'*64,class_names=['class_00','class_01'],
                samples=records,weights_sha256='2'*64,preprocessing=module.GPU.PREPROCESSING,
                feature_spec={'pooling_hw':[7,7],'conv_activation':'pre_relu','fc_activation':'post_relu'},
                source_sha256='3'*64,stage='frozen_pretrained')
        values={name:torch.stack([torch.full((49,width),float(i)) for i in range(4)])
                for name,width in [('conv3',256),('conv4',512),('conv5',512)]}
        values['fc']=torch.stack([torch.full((4096,),float(i)) for i in range(4)])
        write_feature_cache(self.root/'pool',identity(rows),[values])
        pool=FeatureCache(self.root/'pool',expected_identity=identity(rows))
        selected={'pool':rows,'train':[rows[i] for i in (0,2,3)],'val':[rows[1]]}
        module.derive_split_caches(pool,selected,identity,self.root)
        train=FeatureCache(self.root/'train_cache',expected_identity=identity(selected['train']))
        val=FeatureCache(self.root/'val_cache',expected_identity=identity(selected['val']))
        self.assertEqual([float(train[i]['fc'][0]) for i in range(3)],[0.,2.,3.])
        self.assertEqual(float(val[0]['conv5'][0,0]),1.)
        self.assertEqual(float(pool[1]['fc'][0]),1.)

    def test_actual_test_zip_recovery_counts_and_refuses_overwrite(self):
        module=api();archive=self.root/'images.zip';records=[]
        with zipfile.ZipFile(archive,'w') as z:
            for i in range(3):
                raw=f'test-image-{i}'.encode();path=f'class_00/test_{i}.jpg'
                z.writestr('NWPU-RESISC45/'+path,raw)
                records.append({'path':path,'sha256':hashlib.sha256(raw).hexdigest(),'split':'test'})
        destination=self.root/'restored'
        module.recover_test_images(archive,records,destination,expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        report=json.loads((destination/'recovery.json').read_text())
        self.assertEqual(report['test_images_extracted'],3)
        self.assertEqual(report['selected_images'],3)
        self.assertEqual((destination/'class_00/test_2.jpg').read_bytes(),b'test-image-2')
        with self.assertRaises(FileExistsError):module.recover_test_images(archive,records,destination,expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())

    def test_test_recovery_rejects_traversal_wrong_sha_and_mixed_split(self):
        module=api();archive=self.root/'unsafe.zip'
        with zipfile.ZipFile(archive,'w') as z:z.writestr('../outside.jpg',b'bad')
        rows=[{'path':'class_00/test.jpg','sha256':'1'*64,'split':'test'}]
        for digest,records in [('0'*64,rows),(hashlib.sha256(archive.read_bytes()).hexdigest(),rows),
                               (hashlib.sha256(archive.read_bytes()).hexdigest(),[dict(rows[0],split='train')])]:
            with self.assertRaises((ValueError,FileNotFoundError)):module.recover_test_images(archive,records,self.root/'destination',expected_sha256=digest)
        self.assertFalse((self.root/'outside.jpg').exists())
        self.assertFalse((self.root/'destination').exists())

    def test_failure_saves_log_exit_and_cannot_overwrite_or_restart(self):
        module=api();args=self.resources(module)
        def failed(*unused):
            print('BOUNDARY_FIXTURE_NO_CUDA')
            raise RuntimeError('controlled unavailable CUDA boundary')
        with mock.patch.object(module,'execute_pipeline',side_effect=failed),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):module.run_formal(args)
        self.assertEqual(json.loads((args.output/'run.exit').read_text())['exit_code'],1)
        self.assertIn('BOUNDARY_FIXTURE_NO_CUDA',(args.output/'run.log').read_text())
        self.assertTrue((args.output/'failure.json').is_file())
        previous=(args.output/'run.log').read_bytes()
        with self.assertRaises(ValueError):module.run_formal(args)
        self.assertEqual((args.output/'run.log').read_bytes(),previous)
        self.assertEqual(module.status(args.output)['state'],'failed')

    def test_malformed_batch_marker_cannot_establish_training_start(self):
        module=api();args=self.resources(module)
        def failed(*unused):
            print('DFAGCN_TRAIN_BATCH {"phase":"calibration","epoch":1,"device":"cuda:0"}')
            raise RuntimeError('controlled boundary after a reported batch')
        with mock.patch.object(module,'execute_pipeline',side_effect=failed),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):module.run_formal(args)
        self.assertIsNone(json.loads((args.output/'failure.json').read_text())['formal_training_started'])

    def test_formal_stage_order_and_receipts_with_only_cuda_boundary_substituted(self):
        module=api();args=self.resources(module)
        import torch
        from external_baselines.dfagcn import extraction,pipeline
        runtime={'python':'3.11.8 (boundary fixture)','torch':'2.2.2+cu121','torchvision':'0.17.2+cu121',
            'numpy':'1.26.4','pillow':'10.3.0','cuda':'12.1','cudnn':8902,'device':'cuda:0',
            'physical_gpu_uuid':module.GPU.TARGET_UUID,'cpu_threads':8,'seed':42,'dtype':'float32',
            'amp':False,'matmul_tf32':False,'cudnn_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
        class Adapter(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
            def forward(self,left,right):return {'fc':torch.ones(1,1)}
        class Cache:
            key='boundary-fixture';content_sha256='1'*64
            def __init__(self,count):self.count=count
            def __len__(self):return self.count
        def extract(adapter,root,rows,identity,output,**kwargs):
            if len(rows)==25200:self.assertTrue((args.output/'refit/refit.json').is_file())
            self.assertEqual(kwargs['batch_size'],1)
            output.mkdir();(output/'index.json').write_text(json.dumps({'fixture_samples':len(rows)}))
            # Actual adapter hooks exercise progress accounting; no VGG or CUDA is claimed.
            for unused in rows:adapter(torch.zeros(1,1),torch.zeros(1,1))
            return Cache(len(rows))
        def splits(pool,selected,identity,output):
            for split in ('train','val'):
                (output/(split+'_cache')).mkdir()
                (output/(split+'_cache/index.json')).write_text(json.dumps({'fixture_samples':len(selected[split])}))
        def calibrate(*pos,**kw):
            self.assertEqual(kw['mode'],'formal');self.assertNotIn('train_per_class',kw)
            self.assertEqual(pos[1],args.scratch_root/'train_cache')
            self.assertEqual(pos[2],args.scratch_root/'val_cache')
            self.assertEqual(pos[4],args.output/'calibration')
            output=pos[4];output.mkdir()
            for name in ('best.pt','final.pt'):(output/name).write_bytes(b'CUDA-boundary-artifact-fixture')
            (output/'selection.json').write_text('{"fixture":true}')
            return {'mode':'formal','fit_scope':'train','scores':[{}]*9,'selected_epoch':7,'selected_C':1}
        def refit(*pos,**kw):
            self.assertEqual(pos[1],args.scratch_root/'pool_cache')
            self.assertEqual(pos[3],args.output/'refit')
            output=pos[3];output.mkdir();(output/'linear').mkdir()
            for name in ('model.pt','final.pt'):(output/name).write_bytes(b'CUDA-boundary-artifact-fixture')
            (output/'linear/model.txt').write_text('fixture')
            (output/'refit.json').write_text('{"fixture":true}')
            return {'fit_scope':'train_plus_val','fit_samples':6300,'completed':True,'reload_max_abs_error':0.}
        def evaluate(*pos,**kw):
            self.assertEqual(pos[1],args.scratch_root/'test_cache')
            self.assertEqual(pos[2],args.output/'refit')
            self.assertEqual(pos[3],args.output/'test')
            output=pos[3];output.mkdir();(output/'test.json').write_text('{"fixture":true}')
            return {'completed':True,'sample_paths':['fixture']*25200,
                    'metrics':{'oa_percent':80.,'macro_f1_percent':79.},'cached_pipeline_elapsed_seconds':1.}
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(module.GPU,'_runtime_and_torch',return_value=(torch,runtime)))
            stack.enter_context(mock.patch.object(module.SMOKE,'validate_selected',return_value=True))
            for name in ('synchronize','empty_cache','reset_peak_memory_stats'):
                stack.enter_context(mock.patch.object(torch.cuda,name,return_value=None))
            for name in ('max_memory_allocated','max_memory_reserved'):
                stack.enter_context(mock.patch.object(torch.cuda,name,return_value=0))
            stack.enter_context(mock.patch.object(extraction,'load_vgg_adapter',return_value=Adapter()))
            stack.enter_context(mock.patch.object(extraction,'extract_cache',side_effect=extract))
            stack.enter_context(mock.patch.object(module,'derive_split_caches',side_effect=splits))
            stack.enter_context(mock.patch.object(pipeline,'calibrate',side_effect=calibrate))
            stack.enter_context(mock.patch.object(pipeline,'refit',side_effect=refit))
            stack.enter_context(mock.patch.object(pipeline,'evaluate',side_effect=evaluate))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result=module.run_formal(args)
        self.assertEqual(result['counts'],{'train':5040,'val':1260,'pool':6300,'test':25200})
        for split,samples in [('pool',6300),('train',5040),('val',1260),('test',25200)]:
            self.assertTrue((args.scratch_root/(split+'_cache/index.json')).is_file())
            self.assertFalse((args.output/(split+'_cache')).exists())
            self.assertEqual(json.loads((args.output/'cache_manifests'/(split+'.json')).read_text())['fixture_samples'],samples)
        receipt=json.loads((args.output/'phases/test.json').read_text())
        self.assertIn('source_identity_sha256',receipt.get('identity',{}),
                      'Each stage needs a binding to the saved run identity')
        self.assertEqual(receipt['identity']['source_identity_sha256'],
                         hashlib.sha256((args.output/'source_identity.json').read_bytes()).hexdigest())
        self.assertEqual(receipt['identity']['runtime_sha256'],
                         hashlib.sha256((args.output/'runtime.json').read_bytes()).hexdigest())
        self.assertEqual(receipt['identity']['physical_gpu_uuid'],'GPU-70be4364-0641-e89d-828e-78727c6001ec')
        self.assertEqual(module.status(args.output)['state'],'complete')
        self.assertEqual(json.loads((args.output/'run.exit').read_text())['exit_code'],0)
        self.assertLess((args.output/'summary.json').stat().st_size,15000)

    def test_compact_status_excludes_predictions_and_confusion_matrix(self):
        module=api();output=self.root/'run';output.mkdir()
        (output/'summary.json').write_text(json.dumps({'completed':True,'selected_epoch':7,'selected_C':1,
            'test_metrics':{'oa_percent':80.,'macro_f1_percent':79.,'predictions':list(range(10000)),'confusion_matrix':[[1]*45]*45}}))
        (output/'run.exit').write_text('{"exit_code":0}')
        report=module.status(output)
        self.assertEqual(report['state'],'complete')
        self.assertLess(len(json.dumps(report)),1500)
        self.assertNotIn('confusion_matrix',json.dumps(report))
        self.assertNotIn('predictions',json.dumps(report))

    def test_later_failure_preserves_observed_training_start(self):
        module=api();args=self.resources(module)
        def trained_then_failed(*unused):
            print('DFAGCN_TRAIN_BATCH '+json.dumps({'phase':'calibration','epoch':1,
                'batch_samples':100,'device':'cuda:0','loss':1.25}),flush=True)
            raise RuntimeError('controlled failure after observed training')
        with mock.patch.object(module,'execute_pipeline',side_effect=trained_then_failed),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):module.run_formal(args)
        failure=json.loads((args.output/'failure.json').read_text())
        self.assertIs(failure['formal_training_started'],True,
                      'A later failure must not erase evidence of actual training')

    def test_failure_before_first_calibration_batch_keeps_start_unknown(self):
        module=api();args=self.resources(module)
        def first_batch_failed(*unused):
            (args.output/'progress.json').write_text(json.dumps({'phase':'calibration','state':'running'}))
            raise RuntimeError('controlled failure before first batch report')
        with mock.patch.object(module,'execute_pipeline',side_effect=first_batch_failed),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):module.run_formal(args)
        self.assertIsNone(json.loads((args.output/'failure.json').read_text())['formal_training_started'])

    def test_runtime_rejects_cpu_or_changed_cuda_software_recipe(self):
        module=api()
        runtime={'python':'3.11.8 (example)','torch':'2.2.2+cu121','torchvision':'0.17.2+cu121',
            'numpy':'1.26.4','pillow':'10.3.0','cuda':'12.1','cudnn':8902,'device':'cuda:0',
            'physical_gpu_uuid':module.GPU.TARGET_UUID,'cpu_threads':8,'seed':42,'dtype':'float32',
            'amp':False,'matmul_tf32':False,'cudnn_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
        module.require_runtime(runtime,expected_gpu_uuid=module.GPU.TARGET_UUID)
        for key,value in [('device','cpu'),('torch','2.3.0'),('amp',True),('physical_gpu_uuid','GPU-other')]:
            changed=dict(runtime);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):module.require_runtime(changed,expected_gpu_uuid=module.GPU.TARGET_UUID)


if __name__=='__main__':unittest.main()
