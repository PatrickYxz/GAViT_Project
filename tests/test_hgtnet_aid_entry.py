"""Stdlib gates and evidence semantics of the independent AID smoke entry."""
import importlib
import hashlib
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def api():
    try:
        return importlib.import_module('external_baselines.hgtnet.aid_entry')
    except ModuleNotFoundError as exc:
        raise AssertionError('independent AID entry has not been implemented') from exc


def gpu(uuid='GPU-current'):
    return dict(uuid=uuid, name='NVIDIA GeForce RTX 3060', memory_total_mib=12288,
                memory_free_mib=12000, driver_version='535.1')


def complete_fixture(output):
    """Consistent recorded real-AID evidence; opaque checkpoint bytes stay a fixture."""
    output.mkdir()
    cfg=json.loads((ROOT/'configs/hgtnet_aid_independent_v0.json').read_text())
    names=sorted(api().DATA.inspector.AID_CLASSES)
    paths={split:[f'{name}/{split}{index:02}.jpg' for name in names for index in range(count)]
           for split,count in [('train',8),('val',2)]}
    labels={split:[label for label in range(30) for _ in range(count)] for split,count in [('train',8),('val',2)]}
    runtime=dict(python='3.11.9',torch='2.2.2+cu121',torchvision='0.17.2+cu121',numpy='1.26.4',pillow='10.3.0',
                 cuda='12.1',cudnn=8902,physical_gpu_uuid='GPU-current',cuda_visible_devices='GPU-current',
                 device='cuda:0',cpu_threads=8,amp=False,dtype='float32',seed=42,matmul_tf32=False,
                 cudnn_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,gpu=gpu(),
                 cuda_device=dict(available=True,device_count=1,name=gpu()['name'],total_memory_bytes=12288*1024**2))
    groups={name:dict(tensors=1,l2=1.0) for name in ['patch_embed','transformer','dual_attention','graph','head']}
    gradients=dict(passed=True,missing=[],nonfinite=[],groups=groups)
    first=dict(stage='first_optimizer_step',training_started=True,optimizer_steps=1,samples_seen=2,
               optimizer_step_succeeded=True,diagnostic_passed=True,device='cuda:0',gradients=gradients,
               updated_groups={name:True for name in groups},loss=1.0,elapsed_seconds=1.0)
    metrics=dict(accuracy_percent=100.0,macro_f1_percent=100.0,scope='smoke_validation_only',
                 confusion_matrix=[[2 if i==j else 0 for j in range(30)] for i in range(30)],
                 per_class=[dict(**{'class':name},support=2,correct=2,accuracy_percent=100.0,f1_percent=100.0) for name in names])
    history=dict(epoch=1,optimizer_steps=120,train_loss=1.0,val_loss=1.0,train_accuracy_percent=50.0,
                 train_seconds=120.0,validation_seconds=10.0,train_images_per_second=2.0)
    protocol=dict(seed=42,epochs=1,batch_size=2,optimizer='Adam',lr=1e-4,loss='CrossEntropyLoss',
                  augmentation=False,amp=False,tf32=False,preprocessing='RGB_resize224_bilinear_ImageNet_normalize')
    sources=api()._source_identity()
    binding=dict(schema='hgtnet-aid-smoke-binding-v1',input_scope='real_aid',config=cfg,config_sha256=api().CONFIG_SHA,
                 class_names=names,counts=dict(train=240,val=60,test=0),runtime=runtime,source_identity=sources,
                 manifest_sha256=api().DATA.MANIFEST_SHA,weights_sha256=api().WEIGHTS_SHA,
                 train_paths=paths['train'],train_labels=labels['train'],val_paths=paths['val'],val_labels=labels['val'])
    checks=dict(source_identity=True,config=True,manifest=True,weights=True,selected_images=True,
                selected_image_count=300,test_image_bytes_read=0,checkpoint_sha256=True,checkpoint_reload=True)
    (output/'smoke_epoch1.pt').write_bytes(b'opaque-checkpoint-unit-fixture')
    summary=dict(schema=api().SCHEMA,completed=True,training_started=True,formal_training_started=False,
                 input_scope='real_aid',counts=dict(train=240,val=60,test=0),optimizer_steps=120,best_epoch=1,final_epoch=1,
                 config_sha256=api().CONFIG_SHA,manifest_sha256=api().DATA.MANIFEST_SHA,weights_sha256=api().WEIGHTS_SHA,
                 source_identity=sources,runtime=runtime,physical_gpu_uuid='GPU-current',protocol=protocol,
                 history=history,val_metrics=metrics,first_batch_gradients=gradients,completion_checks=checks,
                 selected_pool_count=300,data_source='verified_images',
                 checkpoint=dict(path=str(output/'smoke_epoch1.pt'),sha256=hashlib.sha256(b'opaque-checkpoint-unit-fixture').hexdigest(),
                                 bytes=len(b'opaque-checkpoint-unit-fixture'),best_epoch=1,final_epoch=1,synthetic_inputs=False,
                                 reload_max_abs_error=0.0,reload_tolerance=1e-6))
    artifacts={'summary.json':summary,'binding.json':binding,'runtime.json':runtime,'source_identity.json':sources,
               'history.json':[history],'first_step.json':first,
               'optimizer_progress.json':dict(stage='training',training_started=True,optimizer_steps=120,samples_seen=240),
               'val_predictions.json':dict(scope='smoke_validation_only',paths=paths['val'],labels=labels['val'],
                                           predictions=labels['val'],metrics=metrics),
               'progress.json':dict(stage='complete',completed=True,training_started=True,optimizer_steps=120)}
    for name,value in artifacts.items():(output/name).write_text(json.dumps(value))
    (output/'run.exit').write_text('0\n')
    return summary


class EntryTests(unittest.TestCase):
    def test_30_class_config_changes_only_num_classes(self):
        old = json.loads((ROOT/'configs/hgtnet_independent_v0.json').read_text())
        new_path = ROOT/'configs/hgtnet_aid_independent_v0.json'
        self.assertTrue(new_path.is_file(), 'separate AID config is missing')
        new = json.loads(new_path.read_text())
        self.assertEqual(new, dict(old, num_classes=30))
        from external_baselines.hgtnet.config import config_sha256
        self.assertEqual(api().CONFIG_SHA, config_sha256(new))

    def test_defaults_and_command_use_distinct_AID_install_and_absolute_arguments(self):
        args = api().parser().parse_args(['inspect'])
        self.assertEqual(args.output, ROOT/'runs/aid_smoke_seed42_01')
        self.assertEqual(str(args.manifest), '/home/featurize/work/aid_protocol_20260920_03.json')
        self.assertEqual(str(args.data_root), '/home/featurize/data/AID/AID')
        self.assertEqual(str(args.weights), '/home/featurize/work/swin_b-68c6b09e.pth')
        args.output = Path('output with spaces')
        with patch.dict(os.environ, {'HGTNET_AID_DELIVERY_ENTRY': '/tmp/a bundle.pyz'}):
            command = shlex.split(api().run_command(args, 'GPU-current'))
        self.assertEqual(command[:3], [sys.executable, '/tmp/a bundle.pyz', 'run'])
        self.assertEqual(command[command.index('--gpu-uuid')+1], 'GPU-current')
        self.assertTrue(Path(command[command.index('--output')+1]).is_absolute())
        self.assertNotIn('--archive', command)

    def test_GPU_gate_selects_only_unique_idle_exact3060_and_requested_UUID(self):
        inventory = dict(error=None, gpus=[gpu()], processes=[])
        self.assertEqual(api().select_gpu(None, inventory), gpu())
        self.assertEqual(api().select_gpu('GPU-current', inventory), gpu())
        variants = [dict(inventory, error='query failed'),
                    dict(inventory, gpus=[gpu(), gpu('GPU-other')]),
                    dict(inventory, processes=[dict(gpu_uuid='GPU-current', pid=1, name='busy', used_memory_mib=1)]),
                    dict(inventory, gpus=[dict(gpu(), memory_total_mib=12000)]),
                    dict(inventory, gpus=[dict(gpu(), name='NVIDIA GeForce RTX 3090')])]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                api().select_gpu(None, value)
        with self.assertRaises(ValueError): api().select_gpu('GPU-wrong', inventory)

    def test_CUDA_gate_keeps_physical_capacity_and_bounded_interface_gap(self):
        probe = dict(available=True, device_count=1, name=gpu()['name'],
                     total_memory_bytes=12288*1024**2-100*1024**2)
        value = api()._check_cuda_device(probe, gpu())
        self.assertEqual(value['nvml_total_mib'], 12288)
        self.assertEqual(value['interface_gap_mib'], 100)
        for invalid in [dict(probe, available=False), dict(probe, device_count=2),
                        dict(probe, name='other'), dict(probe, total_memory_bytes=11263*1024**2),
                        dict(probe, total_memory_bytes=12289*1024**2)]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                api()._check_cuda_device(invalid, gpu())

    def test_actual_runtime_gate_requires_exact_cuda_builds_and_FP32_settings(self):
        runtime = dict(python='3.11.9', torch='2.2.2+cu121', torchvision='0.17.2+cu121',
                       numpy='1.26.4', pillow='10.3.0', cuda='12.1', cudnn=8902,
                       physical_gpu_uuid='GPU-current', device='cuda:0', cpu_threads=8,
                       amp=False, dtype='float32', seed=42, matmul_tf32=False, cudnn_tf32=False,
                       cudnn_benchmark=False, cudnn_deterministic=True, gpu=gpu(),
                       cuda_visible_devices='GPU-current')
        self.assertEqual(api().check_runtime(runtime, 'GPU-current'), runtime)
        for change in [dict(torch='2.2.2'), dict(torchvision='0.17.2'), dict(python='3.12.1'),
                       dict(device='cpu'), dict(matmul_tf32=True), dict(cpu_threads=1),
                       dict(physical_gpu_uuid='GPU-wrong'), dict(cuda_visible_devices='0')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                api().check_runtime(dict(runtime, **change), 'GPU-current')

    def test_status_needs_exit_progress_summary_checkpoint_and_current_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()/'run'
            self.assertEqual(api().status(output)['state'], 'not_started')
            summary=complete_fixture(output)
            (output/'progress.json').unlink()
            self.assertFalse(api().status(output)['completed'])
            (output/'progress.json').write_text(json.dumps(dict(stage='complete',completed=True,training_started=True,optimizer_steps=120)))
            self.assertTrue(api().status(output)['completed'])
            (output/'run.exit').write_text('1\n')
            self.assertEqual(api().status(output)['state'], 'failed')
            (output/'run.exit').write_text('0\n')
            summary['source_identity']['external_baselines/hgtnet/aid_entry.py'] = 'b'*64
            (output/'summary.json').write_text(json.dumps(summary))
            report = api().status(output)
            self.assertFalse(report['completed'])
            self.assertEqual(report['state'], 'source_changed')

    def test_skeletal_or_tampered_completion_receipts_stay_incomplete(self):
        cases=[('skeletal',None),('summary.json',dict(input_scope='cpu_fixture')),
               ('summary.json',dict(manifest_sha256='a'*64)),('summary.json',dict(weights_sha256='b'*64)),
               ('summary.json',dict(completion_checks={})),('summary.json',dict(runtime=[])),
               ('summary.json',dict(counts=dict(train=240,val=60,test=False))),
               ('binding.json',dict(weights_sha256='c'*64)),('runtime.json',dict(device='cpu')),
               ('first_step.json',dict(diagnostic_passed=False)),('optimizer_progress.json',dict(optimizer_steps=119)),
               ('first_step.json',dict(loss='invalid')),
               ('history.json',[]),('val_predictions.json',dict(labels=[0]*60)),
               ('val_predictions.json',dict(predictions=[30]*60))]
        for filename,change in cases:
            with self.subTest(filename=filename,change=change),tempfile.TemporaryDirectory() as temporary:
                output=Path(temporary).resolve()/'run'
                summary=complete_fixture(output)
                if filename=='skeletal':
                    keep=['schema','completed','training_started','formal_training_started','counts','optimizer_steps','source_identity','checkpoint']
                    (output/'summary.json').write_text(json.dumps({name:summary[name] for name in keep}))
                else:
                    path=output/filename;value=json.loads(path.read_text())
                    value=dict(value,**change) if isinstance(change,dict) else change
                    path.write_text(json.dumps(value))
                report=api().status(output)
                self.assertFalse(report['completed'])
                self.assertTrue(report['training_started'])
                self.assertEqual(report['optimizer_steps'],120)

    def test_missing_or_malformed_required_completion_artifacts_stay_incomplete(self):
        for name in ['binding.json','runtime.json','source_identity.json','history.json','val_predictions.json','first_step.json','optimizer_progress.json']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temporary:
                output=Path(temporary).resolve()/'run';complete_fixture(output)
                path=output/name;original=path.read_text();path.unlink()
                self.assertFalse(api().status(output)['completed'])
                for raw in ['null','[]','42','{"broken":', '{"optimizer_steps":true}']:
                    path.write_text(raw)
                    self.assertFalse(api().status(output)['completed'])
                path.write_text(original)
                self.assertTrue(api().status(output)['completed'])

    def test_end_check_receipt_requires_actual_end_identity_selected_images_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary).resolve()/'run';summary=complete_fixture(output)
            args=api().parser().parse_args(['run','--output',str(output)])
            sources=api()._source_identity();inputs=dict(selected=dict(pool=[{}]*300))
            def identity(path):
                return api().DATA.MANIFEST_SHA if Path(path)==args.manifest else api().WEIGHTS_SHA
            with patch.object(api(),'_sha',side_effect=identity), \
                 patch.object(api(),'_source_identity',return_value=sources), \
                 patch.object(api().DATA,'validate_selected',return_value=dict(checked_count=300,verified_count=300,test_image_bytes_read=0)):
                checks=api()._end_checks(args,sources,inputs,summary)
                self.assertEqual(checks,summary['completion_checks'])
                with patch.object(api(),'_source_identity',return_value={}):
                    with self.assertRaises(ValueError):api()._end_checks(args,sources,inputs,summary)
                with patch.object(api().DATA,'validate_selected',return_value=dict(checked_count=299,verified_count=299,test_image_bytes_read=0)):
                    with self.assertRaises(ValueError):api()._end_checks(args,sources,inputs,summary)
                (output/'smoke_epoch1.pt').write_bytes(b'changed')
                with self.assertRaises(ValueError):api()._end_checks(args,sources,inputs,summary)

    def test_status_handles_malformed_evidence_and_preserves_durable_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary).resolve()
            (output/'summary.json').write_text('[]')
            (output/'progress.json').write_text('42')
            (output/'first_step.json').write_text('null')
            (output/'optimizer_progress.json').write_text(json.dumps(dict(optimizer_steps=3,training_started=True)))
            report=api().status(output)
            self.assertFalse(report['completed'])
            self.assertTrue(report['training_started'])
            self.assertEqual(report['optimizer_steps'],3)
            (output/'summary.json').write_text(json.dumps(dict(completed=False,training_started=False,optimizer_steps=0)))
            report=api().status(output)
            self.assertTrue(report['training_started'])
            self.assertEqual(report['optimizer_steps'],3)

    def test_existing_lock_blocks_run_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary).resolve()/'run'
            api().lock_path(output).write_text('{"pid":123}')
            args=api().parser().parse_args(['run','--output',str(output)])
            with patch.object(api(),'inspect_resources',return_value=dict(ready=True,selected_gpu=gpu())), \
                 self.assertRaises(ValueError):api().run_smoke(args)
            self.assertFalse(output.exists())
            self.assertEqual(api().lock_path(output).read_text(),'{"pid":123}')

    def test_failure_evidence_keeps_update_before_poststep_diagnostic_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            (output/'optimizer_progress.json').write_text(json.dumps(
                dict(optimizer_steps=1,training_started=True,optimizer_step_succeeded=True)))
            self.assertEqual(api()._observed_training(output), dict(training_started=True,optimizer_steps=1))

    def test_failed_run_writes_exit_summary_and_releases_lock_without_ML_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()/'run'
            args = api().parser().parse_args(['run','--output',str(output)])
            preflight = dict(ready=True,selected_gpu=gpu(),blockers=[])
            with patch.object(api(), 'inspect_resources', return_value=preflight), \
                 patch.object(api(), '_runtime_and_torch', side_effect=ValueError('fixture runtime rejection')), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(api().run_smoke(args), 1)
            self.assertEqual((output/'run.exit').read_text(), '1\n')
            summary = json.loads((output/'summary.json').read_text())
            self.assertFalse(summary['training_started'])
            self.assertFalse(summary['formal_training_started'])
            self.assertIn('fixture runtime rejection', summary['error'])
            self.assertFalse(api().lock_path(output).exists())
            with self.assertRaises((ValueError,FileExistsError)):
                api().run_smoke(args)

    def test_failed_run_after_actual_update_keeps_training_started_and_step_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary).resolve()/'run'
            args=api().parser().parse_args(['run','--output',str(output)])
            def fail_after_update(*unused):
                (output/'optimizer_progress.json').write_text(json.dumps(dict(optimizer_steps=1,training_started=True)))
                raise ValueError('postupdate diagnostic fixture')
            with patch.object(api(),'inspect_resources',return_value=dict(ready=True,selected_gpu=gpu())), \
                 patch.object(api(),'_runtime_and_torch',side_effect=fail_after_update), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(api().run_smoke(args),1)
            summary=json.loads((output/'summary.json').read_text())
            self.assertTrue(summary['training_started'])
            self.assertEqual(summary['optimizer_steps'],1)
            self.assertFalse(summary['completed'])
            self.assertEqual(api().status(output)['state'],'failed')

    def test_inspect_status_are_stdlib_only_in_a_fresh_process(self):
        guard = ('import builtins\n_original=builtins.__import__\n'
                 'def guarded(name,*args,**kwargs):\n'
                 ' if name.split(".")[0] in {"torch","torchvision","numpy","PIL"}: '
                 'raise RuntimeError("ML imported: "+name)\n'
                 ' return _original(name,*args,**kwargs)\n'
                 'builtins.__import__=guarded\n')
        with tempfile.TemporaryDirectory() as temporary:
            for action in ('inspect','status'):
                script = guard + ('from external_baselines.hgtnet.aid_entry import main\n'
                                  'raise SystemExit(main('+repr([action,'--output',str(Path(temporary)/'out')])+'))\n')
                result = subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                report = json.loads(result.stdout)
                self.assertFalse(report.get('ready',report.get('completed',False)))
                self.assertFalse((Path(temporary)/'out').exists())


if __name__ == '__main__': unittest.main()
