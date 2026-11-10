"""Portable campaign contract tests; CPU/file fixtures are not GPU evidence."""
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
def api():
    spec=importlib.util.spec_from_file_location('dfagcn_campaign_test',ROOT/'tools/run_dfagcn_campaign.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.api=api();self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def test_consecutive_seeds_and_frozen_recipe_only_change_seed(self):
        frozen=json.loads((ROOT/'configs/dfagcn_independent_v1.json').read_text())
        for seed in (43,44,45):
            recipe=self.api.recipe_for_seed(ROOT/'configs/dfagcn_independent_v1.json',seed)
            expected=copy.deepcopy(frozen);expected['seed']=seed
            self.assertEqual(recipe,expected)
        for seed in (42,46,True):
            with self.assertRaises(ValueError):self.api.recipe_for_seed(ROOT/'configs/dfagcn_independent_v1.json',seed)
        bad=self.root/'recipe.json';bad.write_text(json.dumps(frozen))
        with self.assertRaises(ValueError):self.api.recipe_for_seed(bad,43)
    def test_uuid_selected_gpu_any_model_and_1gib_memory_gap(self):
        gpu={'uuid':'GPU-selected','name':'NVIDIA GeForce RTX 4090','memory_total_mib':24576,'memory_free_mib':24000}
        inv={'gpus':[gpu],'processes':[],'error':None}
        self.assertEqual(self.api.require_gpu('GPU-selected',inv),gpu)
        self.api.require_cuda_memory(24576,23552*2**20)
        for actual in (23551*2**20,253):
            with self.assertRaises(ValueError):self.api.require_cuda_memory(24576,actual)
        inv['processes']=[{'gpu_uuid':'GPU-selected','pid':12}]
        with self.assertRaises(ValueError):self.api.require_gpu('GPU-selected',inv)
    def test_source_parity_is_explicit_and_byte_verified(self):
        parity=self.api.source_parity(ROOT/'output/dfagcn_nwpu_formal_3080_20261030_fix1')
        self.assertTrue(parity['matched']);self.assertGreaterEqual(len(parity['files']),17)
    def test_svm_io_relocation_keeps_original_features_model_receipts_and_failure(self):
        # A fake boundary checks routing, not fitting. No algorithm is substituted in production.
        events=[]
        def train(features,labels,out,**kw):
            events.append((features,labels,kw));out=Path(out);out.mkdir(parents=True)
            (out/'train.svm').write_bytes(b'large-text-fixture');(out/'model.txt').write_bytes(b'model')
            (out/'metadata.json').write_text('{}');return out/'model.txt'
        def predict(features,model,out,**kw):
            out=Path(out);out.mkdir(parents=True);(out/'predict.svm').write_bytes(b'test-fixture')
            (out/'predictions.txt').write_text('1\n');return [1]
        pipeline=types.SimpleNamespace(train_linear=train,predict_linear=predict)
        output=self.root/'work';output.mkdir();scratch=self.root/'scratch';scratch.mkdir()
        features=object();labels=[0,1]
        with self.api.scratch_linear_io(pipeline,output,scratch):
            model=pipeline.train_linear(features,labels,output/'refit/linear',C=10,solver=2,bias=1,train_binary='x')
            self.assertEqual(model,(output/'refit/linear/model.txt').resolve())
            self.assertEqual(pipeline.predict_linear(features,model,output/'test/predictions',predict_binary='y'),[1])
        self.assertIs(events[0][0],features);self.assertEqual(events[0][1],labels)
        self.assertEqual(events[0][2]['C'],10)
        self.assertFalse((output/'refit/linear/train.svm').exists())
        self.assertTrue((scratch/'svm/refit/linear/train.svm').is_file())
        self.assertTrue((output/'refit/linear/scratch_receipt.json').is_file())
        self.assertIs(pipeline.train_linear,train)
    def test_completion_cannot_be_inferred_from_summary_exit_alone(self):
        out=self.root/'out';out.mkdir()
        (out/'summary.json').write_text('{"completed":true}')
        (out/'run.exit').write_text('{"exit_code":0}')
        report=self.api.status(out)
        self.assertNotEqual(report['state'],'complete');self.assertFalse(report['completion_evidence_valid'])
    def test_cache_identity_is_seed_independent_but_all_input_identities_matter(self):
        manifest={'class_names':['a','b']};rows=[{'path':'a/1.jpg','sha256':'1'*64}]
        one=self.api.expected_cache_identity(manifest,rows)
        self.assertEqual(one,self.api.expected_cache_identity(manifest,rows))
        self.assertNotIn('seed',json.dumps(one));self.assertNotIn('training_config": {',json.dumps(one))
        changed=self.api.expected_cache_identity(manifest,[{'path':'a/1.jpg','sha256':'2'*64}])
        self.assertNotEqual(one,changed)
    def test_status_has_no_ml_imports(self):
        real=builtins.__import__
        def guarded(name,*a,**kw):
            if name.split('.')[0] in {'torch','numpy','torchvision','PIL'}:raise AssertionError(name)
            return real(name,*a,**kw)
        with mock.patch.object(builtins,'__import__',side_effect=guarded):
            self.assertEqual(self.api.status(self.root/'missing')['state'],'not_started')

class ResourceAndDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.api=api();self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
    def test_inspect_is_stdlib_and_preserves_outputs_with_actual_storage_gates(self):
        import test_dfagcn_formal_runner as legacy
        manifest=self.root/'manifest.json';manifest.write_text(json.dumps(legacy.manifest_fixture()))
        weights=self.root/'vgg16-397923af.pth';weights.write_bytes(b'weight-fixture')
        archive=self.root/'input.zip'
        import zipfile
        with zipfile.ZipFile(archive,'w') as stream:stream.writestr('small.jpg',b'12345')
        for name in ('train','predict'):
            binary=self.root/name;binary.write_bytes(b'\x7fELF'+bytes(14)+b'\x3e\x00');binary.chmod(0o755)
        args=types.SimpleNamespace(seed=43,output=self.root/'out',scratch_root=self.root/'scratch',cache=self.root/'cache',
            manifest=manifest,data_root=self.root/'data',archive=archive,weights=weights,weights_metadata=None,
            weights_sha256=self.api.WEIGHTS_SHA,recipe=ROOT/'configs/dfagcn_independent_v1.json',
            train_binary=self.root/'train',predict_binary=self.root/'predict',cpu_threads=8,gpu_uuid='GPU-selected',background_log=self.root/'background.log')
        versions={'python':'3.11.8',**{k:{'located':True,'version':v,'import_tested':False} for k,v in self.api.VERSIONS.items()}}
        gpu={'uuid':'GPU-selected','name':'NVIDIA RTX 6000 Ada Generation','memory_total_mib':49152,'memory_free_mib':48000}
        inventory={'gpus':[gpu],'processes':[],'error':None}
        def disk(path,*,minimum_gate_bytes=0):
            device=2 if Path(path)==args.output.parent else 1
            return {'free_bytes':200*2**30,'minimum_gate_bytes':minimum_gate_bytes,'device_id':device}
        real_import=builtins.__import__;real_sha=self.api.GPU.sha256_file
        def guarded(name,*a,**kw):
            if name.split('.')[0] in {'torch','numpy','torchvision','PIL'}:raise AssertionError('ML import '+name)
            return real_import(name,*a,**kw)
        with mock.patch.object(self.api,'MANIFEST_SHA',hashlib.sha256(manifest.read_bytes()).hexdigest()), \
             mock.patch.object(self.api.SMOKE,'validate_selected',side_effect=ValueError('fresh instance')), \
             mock.patch.object(self.api.GPU,'sha256_file',side_effect=lambda p:self.api.WEIGHTS_SHA if Path(p)==weights else real_sha(p)), \
             mock.patch.object(self.api,'dependency_inventory',return_value=versions), \
             mock.patch.object(self.api.GPU,'gpu_inventory',return_value=inventory), \
             mock.patch.object(self.api,'_disk',side_effect=disk), \
             mock.patch.object(builtins,'__import__',side_effect=guarded):
            report=self.api.inspect_resources(args)
            self.assertTrue(report['ready'],report)
            self.assertEqual(report['persistent_disk']['minimum_gate_bytes'],2**30)
            self.assertEqual(report['storage_budget']['recovered_image_bytes'],5)
            self.assertEqual(report['filesystem_capacity_gates']['1']['required'],
                self.api.SVM_TEXT_UPPER_BYTES+5+self.api.FEATURE_CACHE_BYTES+2*2**30)
            self.assertIn('--seed 43',report['run_command'])
            args.output.mkdir();(args.output/'evidence').write_bytes(b'keep')
            blocked=self.api.inspect_resources(args)
            self.assertFalse(blocked['ready']);self.assertIn('output_already_exists',blocked['blockers'])
            self.assertEqual((args.output/'evidence').read_bytes(),b'keep')
        self.assertFalse(args.scratch_root.exists())
    def test_scratch_rejects_persistent_work(self):
        with self.assertRaises(ValueError):self.api.require_local_storage('/home/featurize/work/large_cache')
    def test_original_native_failure_is_retained_and_rethrows(self):
        def failed(features,labels,out,**kw):
            out=Path(out);out.mkdir(parents=True);(out/'stderr.log').write_text('native failed')
            raise RuntimeError('native failed')
        pipeline=types.SimpleNamespace(train_linear=failed,predict_linear=lambda:None)
        output=self.root/'output';output.mkdir();scratch=self.root/'scratch';scratch.mkdir()
        with self.api.scratch_linear_io(pipeline,output,scratch):
            with self.assertRaisesRegex(RuntimeError,'native failed'):
                pipeline.train_linear(object(),[0,1],output/'refit/linear',C=1)
        self.assertEqual((output/'refit/linear/stderr.log').read_text(),'native failed')
        self.assertTrue((output/'refit/linear/scratch_receipt.json').is_file())
        self.assertIs(pipeline.train_linear,failed)
    def test_complete_requires_seals_full_test_reload_and_immutable_artifacts(self):
        out=self.root/'output';out.mkdir()
        summary={'completed':True,'seed':43,'selected_epoch':2,'selected_C':10,'test_metrics':{'oa_percent':50,'macro_f1_percent':49}}
        sources=self.api._source_identity()
        files={'summary.json':summary,'source_identity.json':sources,'runtime.json':{},'recipe.json':{},'recipe_base.json':{},'frozen_cache_receipt.json':{}}
        records={
            'calibration/selection.json':{'completed':True,'fit_scope':'train','scores':[{}]*9,'selected_epoch':2,'selected_C':10},
            'refit/refit.json':{'completed':True,'fit_scope':'train_plus_val','fit_samples':6300,'reload_max_abs_error':0},
            'test/test.json':{'completed':True,'sample_paths':[str(i) for i in range(25200)],'predictions':[0]*25200}}
        for name,value in records.items():value['record_sha256']=self.api.GPU._digest(value);files[name]=value
        for name,value in files.items():
            path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        for name in ('calibration/best.pt','calibration/final.pt','refit/model.pt','refit/final.pt','refit/linear/model.txt'):
            path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'checkpoint-fixture')
        self.api.seal_completion(out);(out/'run.exit').write_text('{"exit_code":0}')
        self.assertEqual(self.api.status(out)['state'],'complete')
        (out/'refit/model.pt').write_bytes(b'changed')
        self.assertFalse(self.api.status(out)['completion_evidence_valid'])
    def test_cache_identity_rejects_post_fit_and_unsealed_cache(self):
        cache=self.root/'cache';cache.mkdir();(cache/'cache.ready.json').write_text('{"completed":true}')
        with self.assertRaises(ValueError):self.api.inspect_frozen_cache(cache,{'pool':{}})
    def test_delivery_extract_reuse_tamper_and_mac_binary_rejection(self):
        spec=importlib.util.spec_from_file_location('dfagcn_campaign_package_test',ROOT/'tools/package_dfagcn_campaign.py')
        package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)
        bundle=self.root/'campaign.pyz';record=package.build_bundle(bundle)
        self.assertFalse(record['manifest']['compiled_binaries_included'])
        workspace,manifest=package.install_bundle(bundle)
        self.assertEqual(package.install_bundle(bundle)[1],manifest)
        (workspace/'configs/dfagcn_independent_v1.json').write_bytes(b'changed')
        with self.assertRaises(ValueError):package.install_bundle(bundle)
        binaries=self.root/'bin';binaries.mkdir()
        for name in ('train','predict'):(binaries/name).write_bytes(b'fake-Mach-O')
        with self.assertRaisesRegex(ValueError,'Linux x86_64 ELF'):
            package.build_bundle(self.root/'mac.pyz',linux_bin_dir=binaries)

if __name__=='__main__':unittest.main(verbosity=2)
