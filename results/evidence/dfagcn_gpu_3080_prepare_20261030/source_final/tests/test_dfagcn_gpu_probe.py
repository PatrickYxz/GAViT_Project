"""Bounded GPU migration checks; CUDA hardware is not simulated as executed."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


ROOT=Path(__file__).resolve().parents[1]


def api():
    path=ROOT/'tools/probe_dfagcn_gpu.py'
    if not path.exists():
        raise AssertionError('3080 bounded GPU probe is not implemented')
    spec=importlib.util.spec_from_file_location('dfagcn_gpu_probe_under_test',path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def manifest_fixture():
    names=[f'class_{i:02}' for i in range(45)]
    records=[]
    for label,name in enumerate(names):
        for split,count in [('train',112),('val',28),('test',560)]:
            for i in range(count):
                records.append({'path':f'{name}/{split}_{i:03}.jpg','label':label,
                                'split':split,'sha256':'1'*64})
    return {'schema_version':1,'dataset':'NWPU-RESISC45','class_names':names,
            'counts':{'train':5040,'val':1260,'test':25200},'records':records}


class GPUProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_exact_3080_uuid_capacity_and_idle_gate(self):
        module=api()
        device={'name':'NVIDIA GeForce RTX 3080','uuid':module.TARGET_UUID,
                'memory_total_mib':10240,'memory_free_mib':9874,'driver_version':'610.57.04'}
        inventory={'gpus':[device],'processes':[],'error':None}
        self.assertEqual(module.require_gpu(module.TARGET_UUID,inventory),device)
        for kind in ('uuid','model','capacity','busy'):
            changed=copy.deepcopy(inventory)
            if kind=='uuid':changed['gpus'][0]['uuid']='GPU-other'
            elif kind=='model':changed['gpus'][0]['name']='NVIDIA GeForce RTX 4090'
            elif kind=='capacity':changed['gpus'][0]['memory_total_mib']=12288
            else:changed['processes']=[{'gpu_uuid':module.TARGET_UUID,'pid':123}]
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                module.require_gpu(module.TARGET_UUID,changed)

    def test_inventory_parses_driver_and_free_memory_at_external_io_boundary(self):
        module=api()
        responses=[types.SimpleNamespace(returncode=0,stdout=f'NVIDIA GeForce RTX 3080, {module.TARGET_UUID}, 10240, 9874, 610.57.04\n',stderr=''),
                   types.SimpleNamespace(returncode=0,stdout='',stderr='')]
        with mock.patch.object(module.subprocess,'run',side_effect=responses):
            inventory=module.gpu_inventory()
        self.assertEqual(inventory['gpus'][0]['memory_free_mib'],9874)
        self.assertEqual(inventory['gpus'][0]['driver_version'],'610.57.04')
        self.assertIsNone(inventory['error'])

    def test_inventory_rejects_malformed_nonempty_process_rows(self):
        module=api()
        responses=[types.SimpleNamespace(returncode=0,stdout=f'NVIDIA GeForce RTX 3080, {module.TARGET_UUID}, 10240, 9874, 610.57.04\n',stderr=''),
                   types.SimpleNamespace(returncode=0,stdout='123, GPU-other, missing-memory\n',stderr='')]
        with mock.patch.object(module.subprocess,'run',side_effect=responses):
            inventory=module.gpu_inventory()
        self.assertIsNotNone(inventory['error'])
        with self.assertRaises(ValueError):module.require_gpu(module.TARGET_UUID,inventory)

    def test_fresh_cuda_required_before_environment_mutation(self):
        module=api()
        fake=types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda:True))
        with mock.patch.dict('sys.modules',{'torch':fake}),mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-old'}):
            with self.assertRaisesRegex(ValueError,'fresh'):
                module.select_cuda_uuid(module.TARGET_UUID)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'GPU-old')

    def test_frozen_manifest_and_pool_only_select_450_train_validation_images(self):
        module=api()
        path=self.root/'manifest.json'
        path.write_text(json.dumps(manifest_fixture()))
        manifest=module.load_manifest(path,expected_sha256=module.sha256_file(path))
        selected=module.select_pool(manifest)
        self.assertEqual(len(selected),450)
        self.assertEqual(sum(r['split']=='train' for r in selected),360)
        self.assertEqual(sum(r['split']=='val' for r in selected),90)
        self.assertFalse(any(r['split']=='test' for r in selected))
        with self.assertRaisesRegex(ValueError,'SHA'):
            module.load_manifest(path,expected_sha256='0'*64)

    def test_expected_cache_header_rejects_changed_order_and_declared_shapes(self):
        module=api()
        identity=module.expected_cache_identity(manifest_fixture(),module.WEIGHTS_SHA)
        self.assertEqual(len(identity['samples']),450)
        from external_baselines.dfagcn.identity import cache_key
        folder=self.root/'cache';folder.mkdir()
        arrays={}
        for name,width in [('conv3',256),('conv4',512),('conv5',512),('fc',4096)]:
            (folder/(name+'.npy')).write_bytes(b'header-only-fixture')
            arrays[name]={'file':name+'.npy','shape':[450,width] if name=='fc' else [450,49,width],
                          'dtype':'float32','sha256':'1'*64}
        index={'schema':'dfagcn-pooled-cache-v1','identity':identity,'key':cache_key(identity),'arrays':arrays}
        path=folder/'index.json';path.write_text(json.dumps(index))
        result=module.inspect_cache(folder,identity)
        self.assertFalse(result['array_bytes_verified'])
        changed=copy.deepcopy(index)
        changed['identity']['samples'].reverse()
        changed['key']=cache_key(changed['identity'])
        path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):module.inspect_cache(folder,identity)
        changed=copy.deepcopy(index);changed['arrays']['conv3']['shape']=[450,48,256]
        path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):module.inspect_cache(folder,identity)

    def test_weight_metadata_conflicts_and_incomplete_digest_are_rejected(self):
        module=api()
        path=self.root/'weights.json'
        path.write_text(json.dumps({'sha256':module.WEIGHTS_SHA}))
        self.assertEqual(module.read_weights_sha(path,None),module.WEIGHTS_SHA)
        with self.assertRaises(ValueError):module.read_weights_sha(path,'397923af'+'0'*56)
        path.write_text(json.dumps({'sha256':'397923af'}))
        with self.assertRaises(ValueError):module.read_weights_sha(path,None)

    def test_missing_resource_inspect_is_read_only_and_does_not_train(self):
        module=api()
        args=module.parser().parse_args(['inspect','--manifest',str(self.root/'missing.json'),
            '--weights',str(self.root/'missing.pth'),'--cache',str(self.root/'missingcache'),
            '--weights-metadata',str(self.root/'missingmeta.json'),'--output',str(self.root/'probe')])
        with mock.patch.object(module,'gpu_inventory',return_value={'gpus':[],'processes':[],'error':'absent'}):
            report=module.inspect_resources(args)
        self.assertFalse(report['ready'])
        self.assertFalse(report['formal_training_started'])
        self.assertFalse((self.root/'probe').exists())
        self.assertIn('manifest',report['blockers'])

    def test_run_requires_explicit_uuid(self):
        module=api()
        with mock.patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit) as caught:
            module.parser().parse_args(['run'])
        self.assertEqual(caught.exception.code,2)

    def test_real_full_recipe_batch100_updates_only_disposable_graph_model_on_cpu(self):
        module=api()
        import torch
        from external_baselines.dfagcn.config import load_recipe
        recipe=load_recipe(ROOT/'configs/dfagcn_independent_v1.json')
        previous=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,previous)
        generator=torch.Generator().manual_seed(11)
        values={name:torch.randn(100,49,width,generator=generator) for name,width in [('conv3',256),('conv4',512),('conv5',512)]}
        values['fc']=torch.randn(100,4096,generator=generator)
        cache=[{name:tensor[i] for name,tensor in values.items()} for i in range(100)]
        before={name:tensor.clone() for name,tensor in values.items()}
        result=module.check_graph_batch(cache,[i%45 for i in range(100)],recipe,'cpu')
        self.assertEqual(result['batch_samples'],100)
        self.assertEqual(result['device'],'cpu')
        self.assertTrue(result['gradients_finite'])
        self.assertTrue(result['parameters_finite'])
        self.assertTrue(result['disposable_parameters_changed'])
        self.assertEqual(result['optimizer_steps'],1)
        self.assertFalse(result['formal_training_started'])
        for name,tensor in values.items():torch.testing.assert_close(tensor,before[name])

    def test_graph_probe_rejects_insufficient_cache_or_bad_labels(self):
        module=api()
        from external_baselines.dfagcn.config import load_recipe
        recipe=load_recipe(ROOT/'configs/dfagcn_independent_v1.json')
        for cache,labels in [([{}],[0]),([{}]*100,[0]),([{}]*100,[45]*100)]:
            with self.subTest(count=len(cache)),self.assertRaises(ValueError):
                module.check_graph_batch(cache,labels,recipe,'cpu')


if __name__=='__main__':unittest.main()
