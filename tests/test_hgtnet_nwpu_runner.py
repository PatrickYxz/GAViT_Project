"""Stdlib inspect/status and failure bookkeeping at unavailable CUDA boundaries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def api():
    path=Path('tools/run_hgtnet_nwpu_smoke.py')
    if not path.exists(): raise AssertionError('HGTNet NWPU runner API missing')
    spec=importlib.util.spec_from_file_location('_test_hgtnet_runner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class RunnerTests(unittest.TestCase):
    def args(self,m,tmp,action='inspect'):
        return m.parser().parse_args([action,'--output',str(Path(tmp,'output')),
            '--scratch-root',str(Path(tmp,'scratch')),'--manifest',str(Path(tmp,'manifest.json'))])

    def test_status_missing_is_read_only_and_never_complete(self):
        m=api()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp,'missing');record=m.status(output)
            self.assertEqual(record['state'],'not_started');self.assertFalse(output.exists())

    def test_inspect_missing_resources_has_specific_blockers_and_no_training(self):
        m=api()
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(m,tmp);report=m.inspect_resources(args)
            self.assertFalse(report['ready'])
            self.assertIn('manifest',report['blockers'])
            self.assertIn('weights',report['blockers'])
            self.assertFalse(report['formal_training_started'])
            self.assertFalse(args.output.exists())

    def test_generated_command_binds_current_uuid_not_historical_one(self):
        m=api()
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(m,tmp)
            command=m.run_command(args,'GPU-new-current')
            self.assertIn('--gpu-uuid GPU-new-current',command)
            self.assertNotIn('GPU-70be4364',command)
            self.assertNotIn('GPU-0d9901ca',command)
            self.assertNotIn('$',command)

    def test_present_zip_is_pending_full_hash_not_rejected_by_unverified_size(self):
        m=api()
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(m,tmp);args.archive=Path(tmp,'source.zip');args.archive.write_bytes(b'pending full SHA')
            args.weights=Path(tmp,'swin.pth');args.weights.write_bytes(b'candidate')
            inputs={'counts':{'train':360,'val':90,'pool':450,'test':0},'selected':{'pool':[]}}
            with patch.object(m.DATA,'prepare_inputs',return_value=inputs),patch.object(m.DATA,'validate_selected',side_effect=ValueError('images missing')),\
                 patch.object(m,'WEIGHTS_BYTES',len(b'candidate')),patch.object(m,'_sha',return_value=m.WEIGHTS_SHA),\
                 patch.object(m.RES,'dependency_inventory',return_value={}),patch.object(m.RES,'dependency_blockers',return_value=[]),\
                 patch.object(m.RES,'inventory_gpu',return_value={}),patch.object(m.RES,'select_gpu',return_value={'uuid':'GPU-current'}):
                report=m.inspect_resources(args)
            self.assertTrue(report['ready'],report['blockers'])
            self.assertFalse(report['archive_sha_verified'])
            self.assertEqual(report['data_source'],'persistent_zip_pending_full_hash_and_selected_recovery')

    def test_run_without_uuid_is_rejected_before_creating_output(self):
        m=api()
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(m,tmp,'run')
            with self.assertRaises(ValueError):m.run_smoke(args)
            self.assertFalse(args.output.exists())

    def test_runtime_failure_is_not_success_and_has_exit_one(self):
        m=api()
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(m,tmp,'run');args.gpu_uuid='GPU-current'
            preflight={'ready':True,'selected_gpu':{'uuid':'GPU-current'},'data_source':'verified_images'}
            with patch.object(m,'inspect_resources',return_value=preflight),patch.object(m,'_runtime_and_torch',side_effect=ValueError('actual CUDA mismatch')):
                self.assertEqual(m.run_smoke(args),1)
            summary=json.loads((args.output/'summary.json').read_text())
            self.assertFalse(summary['completed']);self.assertFalse(summary['training_started'])
            self.assertFalse(summary['formal_training_started'])
            self.assertEqual((args.output/'run.exit').read_text().strip(),'1')

    def test_status_requires_both_summary_complete_and_exit_zero(self):
        m=api()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);(output/'summary.json').write_text(json.dumps({'completed':True,'training_started':True}))
            (output/'run.exit').write_text('1\n')
            self.assertNotEqual(m.status(output)['state'],'complete')
            (output/'run.exit').write_text('0\n')
            self.assertEqual(m.status(output)['state'],'complete')

    def test_completed_zip_recovery_receipt_replaces_pending_inspect_state(self):
        """Drive the wrapper through real CPU JPEG training, validation and reload."""
        import hashlib
        import shutil
        from types import SimpleNamespace
        import torch
        from torch import nn
        from PIL import Image
        from external_baselines.hgtnet import nwpu_engine

        class Tiny(nn.Module):
            def __init__(self, config=None):
                super().__init__(); self.head=nn.Linear(3,45)
            def forward(self, images):
                return self.head(images.mean((2,3)))

        m=api()
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(m,tmp,'run');args.gpu_uuid='GPU-current-fixture'
            args.weights=Path(tmp,'candidate.pt');args.weights.write_bytes(b'CPU candidate fixture')
            args.archive=Path(tmp,'persistent.zip');args.archive.write_bytes(b'staged ZIP boundary fixture')
            args.manifest.write_text('frozen manifest boundary fixture')
            names=[f'class{i:02}' for i in range(45)]
            staged=Path(tmp,'staged');train=[];val=[]
            for split,count,rows in [('train',4,train),('val',2,val)]:
                for index in range(count):
                    relative=f'{names[index]}/{split}.jpg';path=staged/relative
                    path.parent.mkdir(parents=True,exist_ok=True)
                    Image.new('RGB',(19,17),(50+index,100,150)).save(path)
                    rows.append({'path':relative,'label':index,'split':split,
                                 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            inputs={'class_names':names,'selected':{'train':train,'val':val,'pool':train+val},
                    'counts':{'train':4,'val':2,'pool':6,'test':0}}
            preflight={'ready':True,'selected_gpu':{'uuid':args.gpu_uuid},
                       'data_source':'persistent_zip_pending_full_hash_and_selected_recovery',
                       'archive_sha_verified':False}
            cuda=SimpleNamespace(reset_peak_memory_stats=lambda *_:None,
                                 max_memory_allocated=lambda *_:0,max_memory_reserved=lambda *_:0)
            runtime_torch=SimpleNamespace(Generator=torch.Generator,cuda=cuda)
            actual_engine=nwpu_engine.run_epoch_smoke
            def cpu_engine(model,train_loader,val_loader,output,*,binding,device,**kwargs):
                self.assertEqual(device,'cuda:0')
                fixture=dict(binding,input_scope='cpu_fixture',runtime={'device':'cpu'},
                             counts={'train':4,'val':2,'test':0})
                return actual_engine(model,train_loader,val_loader,output,binding=fixture,
                                     device='cpu',**kwargs)
            def staged_recovery(archive,records,destination):
                self.assertEqual(archive,args.archive);self.assertEqual(records,train+val)
                shutil.copytree(staged,destination)
                return destination
            with patch.object(m,'inspect_resources',return_value=preflight),\
                 patch.object(m,'_runtime_and_torch',return_value=(runtime_torch,{'device':'cpu'})),\
                 patch.object(m.DATA,'prepare_inputs',return_value=inputs),\
                 patch.object(m.DATA,'recover_selected',side_effect=staged_recovery) as recovery,\
                 patch.object(m.DATA,'MANIFEST_SHA',hashlib.sha256(args.manifest.read_bytes()).hexdigest()),\
                 patch.object(m,'WEIGHTS_SHA',hashlib.sha256(args.weights.read_bytes()).hexdigest()),\
                 patch('external_baselines.hgtnet.model.HGTNet',Tiny),\
                 patch('external_baselines.hgtnet.pretrained.load_pretrained_file',return_value={'scope':'CPU candidate fixture'}),\
                 patch.object(nwpu_engine,'run_epoch_smoke',side_effect=cpu_engine):
                self.assertEqual(m.run_smoke(args),0)
            recovery.assert_called_once()
            summary=json.loads((args.output/'summary.json').read_text())
            self.assertTrue(summary['completed']);self.assertTrue(summary['training_started'])
            self.assertEqual(summary['checkpoint']['reload_max_abs_error'],0.0)
            self.assertEqual(summary['data_source'],'persistent_zip_verified_selected_recovery')
            self.assertIs(summary['archive_sha_verified'],True)
            self.assertEqual((args.output/'run.exit').read_text(),'0\n')


if __name__=='__main__': unittest.main()
