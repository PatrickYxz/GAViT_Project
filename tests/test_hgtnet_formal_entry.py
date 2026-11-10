"""Formal entry ownership, resource, artifact and blind-test orchestration gates."""
import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

def api():
    try:return importlib.import_module('external_baselines.hgtnet.formal_entry')
    except ModuleNotFoundError as exc:
        if exc.name=='external_baselines.hgtnet.formal_entry':raise AssertionError('formal entry API missing') from exc
        raise

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class FormalEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage=tempfile.TemporaryDirectory();cls.base=Path(cls.storage.name).resolve()
        cls.names=[f'class_{i:02d}' for i in range(45)];rows=[]
        for label,name in enumerate(cls.names):
            folder=cls.base/'images'/name;folder.mkdir(parents=True)
            for split,count in [('train',112),('val',28),('test',560)]:
                for index in range(count):
                    relative=f'{name}/{split}_{index:03d}.jpg';raw=f'{label}:{split}:{index}'.encode()
                    rows.append(dict(path=relative,label=label,split=split,sha256=hashlib.sha256(raw).hexdigest()))
                    (cls.base/'images'/relative).write_bytes(raw)
        cls.manifest=json.dumps(dict(schema_version=2,dataset='NWPU-RESISC45',class_names=cls.names,
            counts=dict(train=5040,val=1260,test=25200),records=rows))
    @classmethod
    def tearDownClass(cls):cls.storage.cleanup()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name).resolve()
    def fixture(self,m):
        manifest=self.folder/'manifest.json';manifest.write_text(self.manifest)
        weights=self.folder/'weights.pth';weights.write_bytes(b'official candidate fixture')
        archive=self.folder/'archive.zip';archive.write_bytes(b'pinned archive fixture')
        return m.parser().parse_args(['inspect','--manifest',str(manifest),'--weights',str(weights),
            '--archive',str(archive),'--data-root',str(self.base/'images'),'--output',str(self.folder/'output'),
            '--scratch-root',str(self.folder/'scratch')])
    @contextlib.contextmanager
    def gates(self,m,a):
        gpu=dict(name='NVIDIA GeForce RTX 3090',uuid='GPU-fixture-current',memory_total_mib=24576,
            memory_free_mib=24123,driver_version='610.57.04')
        identities=dict(m.PROTO.IDENTITIES,manifest_sha256=sha(a.manifest),weights_sha256=sha(a.weights))
        with contextlib.ExitStack() as stack:
            for target,name,value in [(m.DATA,'MANIFEST_SHA',sha(a.manifest)),(m.DATA,'ARCHIVE_SHA',sha(a.archive)),
                (m.COMMON,'WEIGHTS_SHA',sha(a.weights)),(m.COMMON,'WEIGHTS_BYTES',a.weights.stat().st_size)]:
                stack.enter_context(patch.object(target,name,value))
            stack.enter_context(patch.dict(m.PROTO.IDENTITIES,identities))
            stack.enter_context(patch.object(m.COMMON,'_source_identity',return_value={'fixture_source.py':'a'*64}))
            stack.enter_context(patch.object(m.RES,'dependency_inventory',return_value={}))
            stack.enter_context(patch.object(m.RES,'dependency_blockers',return_value=[]))
            stack.enter_context(patch.object(m.RES,'inventory_gpu',return_value=dict(gpus=[gpu],processes=[],error=None)))
            yield
    def runtime(self,m):return patch.object(m.COMMON,'_runtime_and_torch',return_value=(object(),dict(device='cuda:0')))
    def first(self,progress):progress(dict(stage='first_optimizer_step',optimizer_step_succeeded=True,
        optimizer_steps=1,training_started=True,formal_training_started=True))
    def successful(self,a,torch,runtime,inputs,binding,progress,resume):
        self.first(progress);progress(dict(stage='before_test',optimizer_steps=2,training_started=True,formal_training_started=True))
        a._test_root=a.data_root
        return dict(completed=True,training_started=True,formal_training_started=True,optimizer_steps=2,
            selected_epoch=1,counts=inputs['counts'],test=dict(accuracy_percent=1.0,macro_f1_percent=0.5))
    def seed_failure(self,m,a):
        a.gpu_uuid='GPU-fixture-current'
        def fail(a,t,r,i,b,p,resume):self.first(p);raise RuntimeError('fixture later failure')
        with self.runtime(m),patch.object(m,'_run_model',side_effect=fail):self.assertEqual(m.run_formal_entry(a),1)
        recovery=a.output/'recovery';recovery.mkdir();p=recovery/'slot1.pt';p.write_bytes(b'compatible epoch snapshot fixture')
        binding=json.loads((a.output/'binding.json').read_text())
        receipt=dict(sha256=sha(p),generation=1,phase='calibration',epoch=1,
            binding_sha256=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest())
        Path(str(p)+'.json').write_text(json.dumps(receipt))

    def test_formal_entry_api_and_fullcounts_readiness(self):
        m=api();a=self.fixture(m)
        with self.gates(m,a):report=m.inspect_resources(a)
        self.assertTrue(report['ready'],report);self.assertEqual(report['counts'],dict(train=5040,val=1260,pool=6300,test=25200))
        self.assertIn('--recipe',report['run_command']);self.assertIn('--gpu-uuid GPU-fixture-current',report['run_command'])
        self.assertFalse(a.output.exists());self.assertFalse(a.scratch_root.exists())
        self.assertEqual(report['disk']['output']['required_bytes'],6*2**30)
        self.assertEqual(report['disk']['scratch']['required_bytes'],2*2**30)
    def test_run_requires_uuid_and_independent_new_paths(self):
        m=api();a=self.fixture(m)
        with self.assertRaises(ValueError):m.run_formal_entry(a)
        a.gpu_uuid='GPU-fixture-current';a.output.mkdir();(a.output/'keep').write_text('keep')
        with self.gates(m,a),self.assertRaises(ValueError):m.run_formal_entry(a)
        self.assertEqual((a.output/'keep').read_text(),'keep')
    def test_runtime_failure_records_no_training_and_no_scratch(self):
        m=api();a=self.fixture(m);a.gpu_uuid='GPU-fixture-current'
        with self.gates(m,a),patch.object(m.COMMON,'_runtime_and_torch',side_effect=ValueError('CUDA mismatch')):
            self.assertEqual(m.run_formal_entry(a),1)
        self.assertFalse(a.scratch_root.exists());self.assertFalse(m.status(a.output)['training_started'])
        self.assertEqual((a.output/'run.exit').read_text(),'1\n')
    def test_success_requires_engine_and_final_inputs_then_marker(self):
        m=api();a=self.fixture(m);a.gpu_uuid='GPU-fixture-current'
        with self.gates(m,a),self.runtime(m),patch.object(m,'_run_model',side_effect=self.successful):
            self.assertEqual(m.run_formal_entry(a),0)
        result=m.status(a.output);self.assertTrue(result['completed']);self.assertEqual(result['state'],'complete')
        self.assertEqual(result['selected_epoch'],1);self.assertEqual(result['exit_code'],0)
        self.assertEqual(json.loads((a.output/'binding.json').read_text())['counts']['test'],25200)
    def test_before_test_input_mutation_is_rejected_and_first_step_retained(self):
        m=api();a=self.fixture(m);a.gpu_uuid='GPU-fixture-current';reached=[]
        def mutate(a,t,r,i,b,p,resume):
            self.first(p);a.weights.write_bytes(b'mutated after calibration')
            p(dict(stage='before_test',optimizer_steps=1));reached.append('test')
        with self.gates(m,a),self.runtime(m),patch.object(m,'_run_model',side_effect=mutate):
            self.assertEqual(m.run_formal_entry(a),1)
        self.assertEqual(reached,[]);self.assertTrue((a.output/'first_step.json').exists())
        self.assertEqual(m.status(a.output)['optimizer_steps'],1);self.assertFalse(m.status(a.output)['completed'])
    def test_failure_after_update_preserves_training_and_never_completes(self):
        m=api();a=self.fixture(m)
        with self.gates(m,a):self.seed_failure(m,a)
        report=m.status(a.output);self.assertEqual(report['state'],'failed');self.assertTrue(report['training_started'])
        self.assertEqual(report['optimizer_steps'],1)
    def test_resume_requires_owned_matching_binding_and_valid_snapshot_sha(self):
        m=api();a=self.fixture(m);a.resume=True
        with self.gates(m,a):self.assertIn('resume_ownership',m.inspect_resources(a)['blockers'])
        a.resume=False
        with self.gates(m,a):
            self.seed_failure(m,a);a.resume=True;self.assertTrue(m.inspect_resources(a)['ready'])
            (a.output/'recovery/slot1.pt').write_bytes(b'tampered')
            self.assertIn('resume_ownership',m.inspect_resources(a)['blockers'])
    def test_resume_preserves_prior_attempt_and_changes_only_owned_results(self):
        m=api();a=self.fixture(m)
        with self.gates(m,a):
            self.seed_failure(m,a)
            with self.runtime(m),patch.object(m,'_run_model',side_effect=self.successful):
                self.assertEqual(m.run_formal_entry(a,resume=True),0)
        self.assertTrue(m.status(a.output)['completed']);self.assertTrue((a.output/'attempts/0002/previous_run.exit').is_file())
        self.assertEqual((a.output/'attempts/0002/previous_run.exit').read_text(),'1\n')
        self.assertTrue((a.output/'attempts/0001/run.log').is_file())
    def test_resume_rejects_changed_scratch_ownership(self):
        m=api();a=self.fixture(m)
        with self.gates(m,a):
            self.seed_failure(m,a);a.resume=True
            (a.scratch_root/'.formal_owner.json').write_text('{}')
            self.assertIn('resume_ownership',m.inspect_resources(a)['blockers'])

    def test_resume_after_partial_pool_recovery_uses_new_attempt_destination(self):
        m=api();a=self.fixture(m)
        with self.gates(m,a):
            self.seed_failure(m,a)
            partial=a.scratch_root/'pool';partial.mkdir();sentinel=partial/'partial.jpg';sentinel.write_bytes(b'keep partial bytes')
            a.data_root=self.folder/'missing-after-instance-change';calls=[]
            def recover(archive,records,destination,*,expected_sha256,resume):
                calls.append((Path(destination),resume))
                self.assertEqual(Path(destination),a.scratch_root/'pool_attempt_0002')
                self.assertFalse(resume)
                return self.base/'images'
            def execute(*args):
                result=self.successful(*args);a._test_root=self.base/'images';return result
            with self.runtime(m),patch.object(m.DATA,'recover_selected',side_effect=recover),patch.object(m,'_run_model',side_effect=execute):
                self.assertEqual(m.run_formal_entry(a,resume=True),0)
        self.assertEqual(calls,[(a.scratch_root/'pool_attempt_0002',False)])
        self.assertEqual(sentinel.read_bytes(),b'keep partial bytes')
        self.assertFalse((partial/'recovery.json').exists())
    def test_inspect_and_status_never_import_ml(self):
        api();script="""import builtins,sys
original=builtins.__import__
def guarded(name,*a,**kw):
 if name.split('.')[0] in {'torch','torchvision','numpy','PIL'}:raise AssertionError(name)
 return original(name,*a,**kw)
builtins.__import__=guarded
from external_baselines.hgtnet.formal_entry import main
raise SystemExit(main([sys.argv[1],'--output',sys.argv[2]]))
"""
        for action in ['inspect','status']:
            r=subprocess.run([sys.executable,'-c',script,action,str(self.folder/'absent')],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
    def test_status_requires_exact_completion_marker_and_exit_zero(self):
        m=api();a=self.fixture(m);o=a.output;self.assertEqual(m.status(o)['state'],'not_started');o.mkdir()
        (o/'summary.json').write_text(json.dumps(dict(completed=True,selected_epoch=7)))
        (o/'run.exit').write_text('0\n');(o/'completion.marker').write_text('foreign marker\n')
        self.assertFalse(m.status(o)['completed']);(o/'completion.marker').write_text('HGTNET_NWPU_FORMAL_COMPLETE\n')
        self.assertTrue(m.status(o)['completed'])
    def test_command_honors_delivery_entry_and_resume_action(self):
        m=api();a=self.fixture(m);a.resume=True
        with patch.dict('os.environ',dict(HGTNET_DELIVERY_ENTRY='/tmp/formal package.pyz')):
            command=shlex.split(m.run_command(a,'GPU-current'))
        self.assertEqual(command[1],'/tmp/formal package.pyz');self.assertEqual(command[2],'resume')
        self.assertIn('--recipe',command)

    def test_short_status_keeps_metrics_without_large_prediction_or_confusion_arrays(self):
        m=api();a=self.fixture(m);a.output.mkdir()
        (a.output/'summary.json').write_text(json.dumps(dict(completed=True,
            best_val_metrics=dict(accuracy_percent=71.0,macro_f1_percent=70.5,confusion_matrix=[[1]*45]*45),
            test=dict(accuracy_percent=70.0,macro_f1_percent=69.0,confusion_matrix=[[1]*45]*45,predictions=[1]*25200),
            test_metrics=dict(accuracy_percent=70.0,macro_f1_percent=69.0,confusion_matrix=[[1]*45]*45))))
        report=m.status(a.output);self.assertEqual(report['test_metrics']['accuracy_percent'],70.0)
        self.assertEqual(report['best_val_metrics']['accuracy_percent'],71.0)
        self.assertNotIn('predictions',json.dumps(report));self.assertNotIn('confusion_matrix',json.dumps(report))

if __name__=='__main__':unittest.main()
