"""Real stdlib inputs/paths/ZIP receipts; GPU execution is the only fake boundary."""
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
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def api():
    try:
        return importlib.import_module('external_baselines.hgtnet.batch_probe')
    except ModuleNotFoundError as exc:
        if exc.name == 'external_baselines.hgtnet.batch_probe':
            raise AssertionError('batch16 CLI API missing') from exc
        raise


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def fixture(self, m, *, images=True):
        rows=[]; names=[f'class_{i:02d}' for i in range(45)]
        for label,name in enumerate(names):
            for split,count in [('train',112),('val',28),('test',560)]:
                for index in range(count):
                    raw=f'{label}:{split}:{index}'.encode()
                    relative=f'{name}/{split}_{index:03d}.jpg'
                    rows.append(dict(path=relative,label=label,split=split,sha256=hashlib.sha256(raw).hexdigest()))
                    if images and split=='train' and label<16 and index==0:
                        target=self.folder/'images'/relative; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
        manifest=self.folder/'manifest.json'
        manifest.write_text(json.dumps(dict(schema_version=2,dataset='NWPU-RESISC45',class_names=names,
            counts=dict(train=5040,val=1260,test=25200),records=rows)))
        weights=self.folder/'weights.pth'; weights.write_bytes(b'candidate fixture')
        args=m.parser().parse_args(['inspect','--manifest',str(manifest),'--data-root',str(self.folder/'images'),
            '--weights',str(weights),'--output',str(self.folder/'output'),'--scratch-root',str(self.folder/'scratch'),
            '--archive',str(self.folder/'source.zip')])
        return args

    @contextlib.contextmanager
    def gates(self,m,args):
        gpu=dict(name='NVIDIA GeForce RTX 3090',uuid='GPU-fixture-current',memory_total_mib=24576,
                 memory_free_mib=24123,driver_version='610.57.04')
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(m.DATA,'MANIFEST_SHA',sha(args.manifest)))
            stack.enter_context(patch.object(m.COMMON,'WEIGHTS_SHA',sha(args.weights)))
            stack.enter_context(patch.object(m.COMMON,'WEIGHTS_BYTES',args.weights.stat().st_size))
            stack.enter_context(patch.object(m,'RESTORED_CACHE',self.folder/'absent-cache'))
            stack.enter_context(patch.object(m.RES,'dependency_inventory',return_value={}))
            stack.enter_context(patch.object(m.RES,'dependency_blockers',return_value=[]))
            stack.enter_context(patch.object(m.RES,'inventory_gpu',return_value=dict(gpus=[gpu],processes=[],error=None)))
            yield

    def test_missing_api_red_then_selects_one_sorted_train_image_per_label_0_to_15(self):
        m=api(); args=self.fixture(m)
        with self.gates(m,args):
            inputs=m.DATA.prepare_inputs(args.manifest,expected_sha256=m.DATA.MANIFEST_SHA)
            inputs['selected']['train'].reverse()
            records=m.select_probe_records(inputs)
        self.assertEqual([r['label'] for r in records],list(range(16)))
        self.assertEqual([r['path'] for r in records],[f'class_{i:02d}/train_000.jpg' for i in range(16)])
        inputs['selected']['train']=[r for r in inputs['selected']['train'] if r['label']!=7]
        with self.assertRaises(ValueError):m.select_probe_records(inputs)

    def test_inspect_never_requires_unselected_train_val_or_test_bytes(self):
        m=api(); args=self.fixture(m)
        with self.gates(m,args):report=m.inspect_resources(args)
        self.assertTrue(report['ready'],report)
        self.assertEqual(report['expected_unique_images'],16)
        self.assertEqual(report['expected_processed_images'],64)
        self.assertEqual(report['expected_optimizer_steps'],4)
        self.assertEqual(report['counts'],dict(train=16,val=0,test=0))
        self.assertFalse(args.output.exists()); self.assertFalse(args.scratch_root.exists())
        self.assertIn('--gpu-uuid GPU-fixture-current',report['run_command'])
        self.assertIn('--target-gpu 3090',report['run_command'])

    def test_valid_restored_cache_wins_and_bad_hash_falls_back_to_data_root(self):
        m=api(); args=self.fixture(m)
        with self.gates(m,args),patch.object(m,'RESTORED_CACHE',args.data_root):
            self.assertEqual(m.inspect_resources(args)['data_root_used'],str(args.data_root))
        cache=self.folder/'bad-cache';(cache/'class_00').mkdir(parents=True)
        (cache/'class_00/train_000.jpg').write_bytes(b'changed cache')
        with self.gates(m,args),patch.object(m,'RESTORED_CACHE',cache):
            report=m.inspect_resources(args)
        self.assertTrue(report['ready']);self.assertEqual(report['data_root_used'],str(args.data_root))
        (args.data_root/'class_00/train_000.jpg').write_bytes(b'changed')
        with self.gates(m,args):
            self.assertFalse(m.inspect_resources(args)['ready'])

    def test_inspect_rejects_changed_manifest_even_if_selected_bytes_still_exist(self):
        m=api();args=self.fixture(m)
        with self.gates(m,args):
            args.manifest.write_text(args.manifest.read_text()+'\n')
            report=m.inspect_resources(args)
        self.assertFalse(report['ready']);self.assertIn('manifest',report['blockers'])

    def test_zip_recovery_extracts_only_16_train_images_and_reports_verified_receipt(self):
        m=api(); args=self.fixture(m,images=False)
        with zipfile.ZipFile(args.archive,'w') as stream:
            for label in range(16):stream.writestr(f'class_{label:02d}/train_000.jpg',f'{label}:train:0'.encode())
            stream.writestr('class_00/val_000.jpg',b'do not decode')
            stream.writestr('class_00/test_000.jpg',b'do not decode')
        def execute(torch,cfg,root,records,weights,output,progress):
            self.assertEqual(len(list(root.rglob('*.jpg'))),16)
            progress(dict(stage='optimizer_step',optimizer_step_succeeded=True,optimizer_steps=1))
            return dict(completed=True,optimizer_steps=4,training_started=True)
        args.gpu_uuid='GPU-fixture-current'
        with self.gates(m,args),patch.object(m.DATA,'ARCHIVE_SHA',sha(args.archive)),\
             patch.object(m.COMMON,'_runtime_and_torch',return_value=(object(),dict(device='cuda:0'))),\
             patch.object(m,'_run_model',side_effect=execute):
            self.assertEqual(m.run_probe(args),0)
        summary=json.loads((args.output/'summary.json').read_text())
        self.assertTrue(summary['archive_sha_verified']); self.assertTrue(summary['completed'])
        self.assertEqual(summary['data_source'],'persistent_zip_verified_selected_recovery')
        self.assertEqual(json.loads((args.scratch_root/'images/recovery.json').read_text())['selected_images'],16)
        self.assertEqual(m.status(args.output)['state'],'complete')

    def test_failure_after_first_optimizer_keeps_evidence_and_never_completes(self):
        m=api(); args=self.fixture(m); args.gpu_uuid='GPU-fixture-current'
        def fail(torch,cfg,root,records,weights,output,progress):
            progress(dict(stage='optimizer_step',optimizer_step_succeeded=True,optimizer_steps=1))
            raise RuntimeError('later allocation failed')
        with self.gates(m,args),patch.object(m.COMMON,'_runtime_and_torch',return_value=(object(),{})),\
             patch.object(m,'_run_model',side_effect=fail):
            self.assertEqual(m.run_probe(args),1)
        summary=json.loads((args.output/'summary.json').read_text())
        self.assertFalse(summary['completed']);self.assertTrue(summary['training_started'])
        self.assertEqual(summary['optimizer_steps'],1)
        self.assertTrue((args.output/'first_step.json').exists())
        self.assertEqual(m.status(args.output)['state'],'failed')
        self.assertEqual((args.output/'run.exit').read_text(),'1\n')

    def test_run_requires_uuid_and_new_paths_before_writing_anything(self):
        m=api(); args=self.fixture(m)
        with self.assertRaises(ValueError):m.run_probe(args)
        self.assertFalse(args.output.exists())
        args.gpu_uuid='GPU-fixture-current';args.output.mkdir();(args.output/'keep').write_text('old')
        with self.gates(m,args),self.assertRaises(ValueError):m.run_probe(args)
        self.assertEqual((args.output/'keep').read_text(),'old')

    def test_actual_runtime_failure_records_exit_one_without_training_or_scratch(self):
        m=api();args=self.fixture(m);args.gpu_uuid='GPU-fixture-current'
        with self.gates(m,args),patch.object(m.COMMON,'_runtime_and_torch',side_effect=ValueError('CUDA mismatch')):
            self.assertEqual(m.run_probe(args),1)
        summary=json.loads((args.output/'summary.json').read_text())
        self.assertFalse(summary['completed']);self.assertFalse(summary['training_started'])
        self.assertEqual(summary['optimizer_steps'],0);self.assertFalse(args.scratch_root.exists())
        self.assertEqual((args.output/'run.exit').read_text(),'1\n')

    def test_final_image_change_rejects_engine_success_and_keeps_four_update_receipt(self):
        m=api();args=self.fixture(m);args.gpu_uuid='GPU-fixture-current'
        def mutate(torch,cfg,root,records,weights,output,progress):
            progress(dict(stage='optimizer_step',optimizer_step_succeeded=True,optimizer_steps=4))
            (root/records[0]['path']).write_bytes(b'changed during probe')
            return dict(completed=True,optimizer_steps=4)
        with self.gates(m,args),patch.object(m.COMMON,'_runtime_and_torch',return_value=(object(),{})),\
             patch.object(m,'_run_model',side_effect=mutate):
            self.assertEqual(m.run_probe(args),1)
        summary=json.loads((args.output/'summary.json').read_text())
        self.assertFalse(summary['completed']);self.assertEqual(summary['optimizer_steps'],4)
        self.assertTrue(summary['training_started']);self.assertFalse((args.output/'completion.marker').exists())

    def test_status_requires_exit_zero_and_completion_marker_and_has_no_ml_import(self):
        m=api(); output=self.folder/'absent';self.assertEqual(m.status(output)['state'],'not_started')
        output.mkdir();(output/'summary.json').write_text(json.dumps(dict(completed=True)))
        (output/'run.exit').write_text('0\n'); self.assertFalse(m.status(output)['completed'])
        (output/'completion.marker').write_text('HGTNET_BATCH16_PROBE_COMPLETE\n')
        self.assertTrue(m.status(output)['completed'])
        script="""import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','torchvision','numpy','PIL'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from external_baselines.hgtnet.batch_probe import main
raise SystemExit(main(['status','--output',sys.argv[1]]))
"""
        result=subprocess.run([sys.executable,'-c',script,str(output)],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_inspect_imports_no_ml_even_when_all_resources_are_missing(self):
        api()
        script="""import builtins,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','torchvision','numpy','PIL'}:raise AssertionError(name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
from external_baselines.hgtnet.batch_probe import main
raise SystemExit(main(['inspect','--manifest',sys.argv[1],'--weights',sys.argv[1]]))
"""
        result=subprocess.run([sys.executable,'-c',script,str(self.folder/'absent')],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout);self.assertFalse(report['ready'])
        self.assertIn('manifest',report['blockers']);self.assertIn('weights',report['blockers'])

    def test_short_status_exposes_measured_capacity_scope_without_accuracy(self):
        m=api();output=self.folder/'output';output.mkdir()
        (output/'summary.json').write_text(json.dumps(dict(completed=True,unique_images=16,processed_images=64,
            median_step_seconds=0.5,compute_images_per_second=32.0,probe_scope='bounded compute fixture',
            step_seconds=[0.4,0.5,0.6],cuda_peak_allocated_bytes=123,cuda_peak_reserved_bytes=456)))
        (output/'run.exit').write_text('0\n');(output/'completion.marker').write_text('HGTNET_BATCH16_PROBE_COMPLETE\n')
        report=m.status(output)
        self.assertEqual(report.get('median_step_seconds'),0.5)
        self.assertEqual(report.get('compute_images_per_second'),32.0)
        self.assertEqual(report.get('unique_images'),16);self.assertEqual(report.get('processed_images'),64)
        self.assertEqual(report.get('probe_scope'),'bounded compute fixture')
        self.assertNotIn('accuracy',report)

    def test_command_honors_delivery_entry_and_exact_target_profile(self):
        m=api();args=self.fixture(m)
        with patch.dict('os.environ',dict(HGTNET_DELIVERY_ENTRY='/tmp/new package.pyz')):
            command=shlex.split(m.run_command(args,'GPU-current'))
        self.assertEqual(command[1],'/tmp/new package.pyz')
        self.assertEqual(command[command.index('--target-gpu')+1],'3090')
        with self.assertRaises(SystemExit):m.parser().parse_args(['inspect','--target-gpu','3080'])


if __name__=='__main__':unittest.main()
