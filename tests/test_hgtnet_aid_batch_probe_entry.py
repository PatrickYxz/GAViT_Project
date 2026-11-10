"""Stdlib artifact/entry contract, with no CUDA or original data fixture claims."""
import copy
import contextlib
import io
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

def api():
    try:return importlib.import_module('external_baselines.hgtnet.aid_batch_probe')
    except ModuleNotFoundError as exc:raise AssertionError('AID batch16 entry missing') from exc

def records():
    return [dict(path=f'class{i:02}/train{j}.jpg',label=i,split='train',sha256='a'*64,pixel_sha256=f'{i*8+j:064x}') for i in range(30) for j in range(8)]

def gpu():return dict(uuid='GPU-current',name='NVIDIA GeForce RTX 3060',memory_total_mib=12288,memory_free_mib=12000,driver_version='610.57.04')

def fixture(output):
    entry=api();output.mkdir()
    runtime=dict(python='3.11.8',torch='2.2.2+cu121',torchvision='0.17.2+cu121',numpy='1.26.4',pillow='10.3.0',cuda='12.1',cudnn=8902,physical_gpu_uuid='GPU-current',cuda_visible_devices='GPU-current',device='cuda:0',cpu_threads=8,amp=False,dtype='float32',seed=42,matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,gpu=gpu(),cuda_device=dict(available=True,device_count=1,name=gpu()['name'],total_memory_bytes=12487622656))
    gradients=dict(passed=True,missing=[],nonfinite=[],groups={name:dict(tensors=count,l2=1.0) for name,count in [('patch_embed',4),('transformer',300),('dual_attention',100),('graph',45),('head',2)]})
    updated={name:True for name in gradients['groups']}
    steps=[dict(stage='first_optimizer_step' if i==1 else 'measurement',optimizer_step_completed=True,optimizer_step_succeeded=True,optimizer_steps=i,step_index=i,phase='warmup' if i==1 else 'measured',training_started=True,formal_training_started=False,completed=False,batch_size=16,samples_seen=i*16,elapsed_seconds=float(i),loss=1.0,post_update_checks_passed=True) for i in range(1,5)]
    steps[0].update(gradients=gradients,updated_groups=updated)
    sources=entry._source_identity();selection=dict(records=[records()[i*8] for i in range(16)],unique_images=16,counts=dict(train=16,val=0,test=0),manifest_sha256=entry.DATA.MANIFEST_SHA,selection_rule=entry.SELECTION_RULE)
    cfg=json.loads((ROOT/'configs/hgtnet_aid_independent_v0.json').read_text())
    provenance=dict(schema=entry.SCHEMA,input_scope='real_aid',counts=dict(train=16,val=0,test=0),source_identity=sources,config_sha256=entry.CONFIG_SHA,manifest_sha256=entry.DATA.MANIFEST_SHA,weights_sha256=entry.WEIGHTS_SHA,weights_bytes=351654207,data_source='verified_images',data_root_used='/fixture/original/train',selection=selection,runtime=runtime)
    summary=dict(schema=entry.SCHEMA,completed=True,training_started=True,formal_training_started=False,input_scope='real_aid',target_gpu='3060',counts=dict(train=16,val=0,test=0),seed=42,batch_size=16,unique_images=16,processed_images=64,optimizer_steps=4,warmup_steps=1,measured_steps=3,device='cuda:0',step_seconds=[2.,3.,4.],warmup_seconds=1.,median_step_seconds=3.,mean_step_seconds=3.,compute_images_per_second=16/3,first_step_gradients=gradients,first_step_updated_groups=updated,parameters_finite_after=True,gradient_clipping=False,accuracy_metrics=None,checkpoint_saved=False,optimizer=dict(name='Adam',lr=1e-5,weight_decay=5e-5,betas=[.9,.999],eps=1e-8,amsgrad=False,foreach=None,fused=None),parameters=dict(total=74612163,trainable=74612163),physical_gpu_uuid='GPU-current',wall_seconds=15.,cuda_peak_allocated_bytes=100,cuda_peak_reserved_bytes=200,completion_checks=dict(source_identity=True,config=True,manifest=True,weights=True,selected_images=True,selected_image_count=16,test_image_bytes_read=0,val_image_bytes_read=0),**{key:value for key,value in provenance.items() if key not in ['schema','input_scope','counts']})
    artifacts={'summary.json':summary,'runtime.json':runtime,'source_identity.json':sources,'config.json':cfg,'selection.json':selection,'provenance.json':provenance,'first_step.json':steps[0],'steps.json':steps,'optimizer_progress.json':steps[-1],'progress.json':dict(stage='complete',completed=True,training_started=True,formal_training_started=False,optimizer_steps=4,samples_seen=64)}
    artifacts.update({f'step_{i:02}.json':step for i,step in enumerate(steps,1)})
    for name,value in artifacts.items():(output/name).write_text(json.dumps(value))
    (output/'run.exit').write_text('0\n');return summary

class EntryTests(unittest.TestCase):
    def test_selects_only_first_original_train_path_per_labels0_to15(self):
        selected=api().select_probe_records({'selected':{'train':list(reversed(records()))}})
        self.assertEqual([r['label'] for r in selected],list(range(16)))
        self.assertEqual([r['path'] for r in selected],[f'class{i:02}/train0.jpg' for i in range(16)])
        bad=records();bad[0]['split']='val'
        with self.assertRaises(ValueError):api().select_probe_records({'selected':{'train':bad}})
    def test_defaults_and_unique_idle3060_and_lock_prevent_unsafe_start(self):
        args=api().parser().parse_args(['inspect'])
        self.assertEqual(args.output,ROOT/'runs/batch16_seed42_01')
        self.assertEqual(api().select_gpu(None,dict(error=None,gpus=[gpu()],processes=[])),gpu())
        with self.assertRaises(ValueError):api().select_gpu(None,dict(error=None,gpus=[gpu(),dict(gpu(),uuid='GPU-other')],processes=[]))
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp).resolve()/'run'
            with api()._run_lock(output):
                with self.assertRaises(ValueError):api()._new_output(output)
            output.mkdir()
            with self.assertRaises(FileExistsError):api()._new_output(output)
    def test_readonly_import_inspect_status_never_import_ML(self):
        script='import builtins\nf=builtins.__import__\ndef guard(name,*a,**kw):\n if name.split(".")[0] in {"torch","torchvision","numpy","PIL"}: raise RuntimeError(name)\n return f(name,*a,**kw)\nbuiltins.__import__=guard\nfrom external_baselines.hgtnet import aid_batch_probe as e\na=e.parser().parse_args(["inspect","--output","/tmp/no_aid_probe_output_fixture"])\nassert not e.inspect_resources(a)["ready"]\nassert not e.status(a.output)["completed"]\n'
        result=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_complete_fixture_passes_but_skeletal_and_individual_tampering_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp).resolve()/'run';fixture(output)
            self.assertTrue(api().status(output)['completed'])
            mutations=[('summary.json','optimizer_steps',3),('summary.json','accuracy_metrics',{}),('summary.json','checkpoint_saved',True),('summary.json','step_seconds',[2.,float('nan'),4.]),('summary.json','median_step_seconds',4.),('runtime.json','device','cpu'),('selection.json','unique_images',15),('first_step.json','updated_groups',{}),('optimizer_progress.json','optimizer_steps',3),('progress.json','samples_seen',63),('summary.json','completion_checks',{}),('source_identity.json','invented','a'*64),('step_03.json','samples_seen',16)]
            for filename,key,value in mutations:
                path=output/filename;original=path.read_text();item=json.loads(original);item[key]=value;path.write_text(json.dumps(item))
                with self.subTest(filename=filename,key=key):self.assertFalse(api().status(output)['completed'])
                path.write_text(original)
            (output/'first_step.json').unlink();self.assertFalse(api().status(output)['completed'])
            (output/'summary.json').write_text(json.dumps(dict(schema=api().SCHEMA,completed=True,optimizer_steps=4)))
            self.assertFalse(api().status(output)['completed'])
    def test_consistently_forged_incomplete_gradient_receipts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp).resolve()/'run';fixture(output)
            for filename in ('summary.json','first_step.json','steps.json','step_01.json'):
                path=output/filename;value=json.loads(path.read_text())
                if filename=='summary.json':value['first_step_gradients']['groups']['transformer']['tensors']=299
                elif filename=='steps.json':value[0]['gradients']['groups']['transformer']['tensors']=299
                else:value['gradients']['groups']['transformer']['tensors']=299
                path.write_text(json.dumps(value))
            self.assertFalse(api().status(output)['completed'])
    def test_inspect_hashes_only_train16_and_reports_detailed_blockers(self):
        with tempfile.TemporaryDirectory() as temp:
            args=api().parser().parse_args(['inspect','--output',str(Path(temp).resolve()/'run')])
            read_paths=[]
            def validated(root,rows):
                read_paths.extend(row['path'] for row in rows)
                return dict(checked_count=16,verified_count=16,test_image_bytes_read=0)
            with patch.object(api().DATA,'prepare_inputs',return_value=dict(selected=dict(train=records()),manifest_sha256=api().DATA.MANIFEST_SHA)),patch.object(api().DATA,'validate_selected',side_effect=validated),patch.object(api(),'_check_weights',return_value=dict(weights_sha256=api().WEIGHTS_SHA,weights_bytes=351654207)),patch.object(api().RES,'inventory_gpu',return_value=dict(error=None,gpus=[gpu()],processes=[])),patch.object(api().SAFE,'_dependency_blockers',return_value=[]),patch.object(api().RES,'_capacity_path',return_value=dict(free_bytes=100000000000)):
                result=api().inspect_resources(args)
            self.assertTrue(result['ready'],result)
            self.assertEqual(read_paths,[f'class{i:02}/train0.jpg' for i in range(16)])
            self.assertEqual(result['counts'],dict(train=16,val=0,test=0))
            self.assertIn('GPU-current',result['run_command'])
    def test_wrapper_success_requires_end_checks_and_records_four_durable_events(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp).resolve();saved=base/'fixture';summary=fixture(saved)
            engine=dict(summary);events=json.loads((saved/'steps.json').read_text());output=base/'run'
            args=api().parser().parse_args(['run','--output',str(output)])
            preflight=dict(ready=True,selected_gpu=gpu(),source_identity=api()._source_identity())
            def numerical(*values):
                for event in events:values[-1](event)
                return engine
            cfg=json.loads((ROOT/'configs/hgtnet_aid_independent_v0.json').read_text())
            original_sha=api()._sha
            def sha(path):return api().DATA.MANIFEST_SHA if Path(path)==args.manifest else original_sha(path)
            with patch.object(api(),'inspect_resources',return_value=preflight),patch.object(api(),'_runtime_and_torch',return_value=(object(),summary['runtime'])),patch.object(api(),'_check_config',return_value=cfg),patch.object(api().DATA,'prepare_inputs',return_value=dict(selected=dict(train=records()))),patch.object(api().DATA,'validate_selected',return_value=dict(checked_count=16,verified_count=16,test_image_bytes_read=0)),patch.object(api(),'_check_weights',return_value=dict(weights_sha256=api().WEIGHTS_SHA,weights_bytes=351654207)),patch.object(api(),'_sha',side_effect=sha),patch.object(api(),'_run_model',side_effect=numerical),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(api().run_probe(args),0)
            self.assertTrue(api().status(output)['completed'])
            self.assertEqual(len(json.loads((output/'steps.json').read_text())),4)
            self.assertFalse((output/'smoke_epoch1.pt').exists())
            self.assertFalse(api().lock_path(output).exists())
    def test_failure_after_actual_update_keeps_durable_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp).resolve()/'run';args=api().parser().parse_args(['run','--output',str(output)])
            preflight=dict(ready=True,selected_gpu=gpu(),source_identity=api()._source_identity())
            def failed(*args):
                progress=args[-1];progress(dict(stage='first_optimizer_step',optimizer_step_succeeded=True,optimizer_step_completed=True,optimizer_steps=1,step_index=1,samples_seen=16,training_started=True,post_update_checks_passed=False))
                raise ValueError('post update diagnostic failed')
            with patch.object(api(),'inspect_resources',return_value=preflight),patch.object(api(),'_runtime_and_torch',return_value=(object(),{})),patch.object(api(),'_check_config',return_value={}),patch.object(api().DATA,'prepare_inputs',return_value={'selected':{'train':records()}}),patch.object(api().DATA,'validate_selected',return_value={}),patch.object(api(),'_run_model',side_effect=failed),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(api().run_probe(args),1)
            result=api().status(output);self.assertEqual(result['optimizer_steps'],1);self.assertTrue(result['training_started']);self.assertFalse(result['completed'])
            self.assertTrue((output/'step_01.json').is_file());self.assertEqual(json.loads((output/'steps.json').read_text())[0]['optimizer_steps'],1)
if __name__=='__main__':unittest.main()
