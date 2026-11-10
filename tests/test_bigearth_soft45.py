"""External launch and artifact gates. CPU fixtures are not CUDA evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Soft45Tests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'tools/run_bigearth_soft45.py'
        self.assertTrue(path.is_file(), 'Soft45 runner has not been implemented')
        self.m = load(path, 'soft45')
        self.b = load(ROOT / 'tools/run_bigearth_seed45.py', 'old45')
        self.b.REPO = self.m.REPO
        self.b.COMMIT = self.m.COMMIT
        self.ready = json.loads((ROOT / 'results/evidence/bigearth_soft45_prepare_20260930/server_preflight_user_return.json').read_text())

    def report(self, stage):
        r = dict(stage=stage,status='complete',model='gavit',seed=45,commit=self.m.COMMIT,
                 hardware=self.ready['hardware'],precision=self.ready['precision'],
                 pretrained_sha256=self.b.PRETRAINED_SHA,splits={k:list(v) for k,v in self.b.FROZEN.items()},
                 architecture=self.m.ARCH,checkpoint_sha256='a'*64)
        if stage == 'gavit_train':
            r.update(final_epoch=30,parameters=31466766)
        else:
            r['metrics'] = dict(samples=119825,mAP_percent=70.,macro_f1_percent=65.,micro_f1_percent=77.,
                                f1_threshold=.5,checkpoint_sha256='a'*64,per_class_ap_percent={str(n):70. for n in range(19)})
        return r

    def test_only_soft_training_and_best_checkpoint_route(self):
        argv=self.m.entry_args(Path('/out'), 'gavit_train', self.b)
        args=dict(zip(argv[1::2],argv[2::2]))
        for k,v in {'--model':'gavit','--grouping':'soft_prototype','--seed':'45','--epochs':'30',
                    '--batch_size':'32','--run_stage':'formal','--edge_type':'sparse_hybrid'}.items():
            self.assertEqual(args[k],v)
        test=self.m.entry_args(Path('/out'),'gavit_test',self.b)
        self.assertEqual(test[test.index('--ckpt')+1],args['--checkpoint_path'])
        self.assertNotIn('--resume',argv)
        with self.assertRaises(RuntimeError): self.m.entry_args(Path('/out'),'swin_train',self.b)

    def test_training_requires_complete_history_correct_best_and_soft_identity(self):
        history=[dict(epoch=i,val_mAP_percent=80.-abs(i-10),train_loss=.1,val_macro_f1_percent=60.,next_epoch_lr=.001,epoch_wall_seconds=2.) for i in range(1,31)]
        meta=dict(model='gavit',dataset='BigEarthNet-19',architecture=self.m.ARCH,training=self.m.TRAINING,
                  execution={'run_stage':'formal'},git={'commit':self.m.COMMIT[:7],'dirty':False},
                  best={'metric':'val_mAP','epoch':10,'value':80.})
        self.m.check_training(meta,history,30)
        for which in ('epochs','best','temperature','seed','final_state'):
            h=copy.deepcopy(history); m=copy.deepcopy(meta); last=30
            if which=='epochs': h.pop()
            if which=='best': m['best']['epoch']=11
            if which=='temperature': m['architecture']['soft_assignment']['assignment_temperature']=.5
            if which=='seed': m['training']['seed']=44
            if which=='final_state': last=29
            with self.subTest(which=which),self.assertRaises(RuntimeError): self.m.check_training(m,h,last)

    def test_success_failure_and_incomplete_exit_zero_stop_queue(self):
        for failure in ('none','process','report'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as td:
                root=Path(td); jobs=[]
                for stage in ('gavit_train','gavit_test'):
                    r=self.report(stage)
                    if failure=='report' and stage=='gavit_train': r['final_epoch']=29
                    code='import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); (p/"run.json").write_text(sys.argv[2]); sys.exit(int(sys.argv[3]))'
                    jobs.append((stage,[sys.executable,'-c',code,str(root/stage),json.dumps(r),'7' if failure=='process' and stage=='gavit_train' else '0']))
                self.b.idle_gpu=lambda: None
                self.b.identity=lambda: None
                if failure=='none':
                    reports=self.m.run_stages(root,jobs,self.b,self.ready)
                    self.assertEqual(list(reports),['gavit_train','gavit_test'])
                    self.assertEqual((root/'gavit_test/run.exit').read_text().strip(),'exit_code=0')
                    with self.assertRaises(FileExistsError): self.m.run_stages(root,jobs,self.b,self.ready)
                else:
                    with self.assertRaises((RuntimeError,subprocess.CalledProcessError)):
                        self.m.run_stages(root,jobs,self.b,self.ready)
                    self.assertFalse((root/'gavit_test').exists())
                    self.assertTrue((root/'gavit_train/run.log').exists())

    def test_environment_and_test_checkpoint_drift_rejected(self):
        r=self.report('gavit_test')
        self.m.check_report(r,'gavit_test',self.ready,self.b)
        for key in ('hardware','architecture','metrics'):
            broken=copy.deepcopy(r)
            if key=='hardware': broken[key]['gpu_identity']='changed after preflight'
            if key=='architecture': broken[key]['grouping']='attentive_spatial'
            if key=='metrics': broken[key]['samples']=1
            with self.subTest(key=key),self.assertRaises(RuntimeError): self.m.check_report(broken,'gavit_test',self.ready,self.b)

    def test_test_arrays_reject_misaligned_labels_nan_and_duplicate_paths(self):
        import numpy as np
        ns=dict(probs_np=np.full((2,19),.5),labels_np=np.ones((2,19)),test_labels=np.ones((2,19)),
                test_patches=['a','b'],ap_per_class=[.5]*19,CLASSES_19=[str(i) for i in range(19)],
                mAP=50.,macro_f1=50.,micro_f1=50.)
        self.m.check_test_arrays(ns,np,count=2)
        for mode in ('nan','duplicate','labels','class_missing'):
            s=copy.deepcopy(ns)
            if mode=='nan': s['probs_np'][0,0]=np.nan
            if mode=='duplicate': s['test_patches']=['a','a']
            if mode=='labels': s['labels_np'][0,0]=0
            if mode=='class_missing': s['labels_np'][:,0]=0; s['test_labels'][:,0]=0
            with self.subTest(mode=mode),self.assertRaises(RuntimeError): self.m.check_test_arrays(s,np,count=2)

    def test_recording_preserves_save_return_and_training_rng(self):
        import torch
        with tempfile.TemporaryDirectory() as td:
            calls=[]
            def original(*a,**kw):
                calls.append(kw['epoch'])
                return kw['metric_value'],True
            utils=types.SimpleNamespace(save_epoch_artifacts=original)
            history,saved=self.m.install_history_recorder(utils,Path(td),self.b)
            state=torch.get_rng_state().clone()
            scope={'utils':utils,'optimizer':types.SimpleNamespace(param_groups=[{'lr':.0001}])}
            exec('avg_loss=.123456789\nval_f1=60.123456789\nresult=utils.save_epoch_artifacts(None,optimizer,epoch=1,metric_value=70.987654321)',scope)
            self.assertEqual(scope['result'],(70.987654321,True))
            self.assertIs(saved,original)
            self.assertEqual(calls,[1])
            self.assertEqual(history[0]['val_mAP_percent'],70.987654321)
            self.assertEqual(history[0]['train_loss'],.123456789)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            self.assertEqual(json.loads((Path(td)/'history.json').read_text()),history)

    def test_actual_assignment_and_graph_snapshots_and_corruptions(self):
        import torch
        from models.soft_region_grouping import SoftPrototypeGrouping
        from models.graph_construction import build_sparse_hybrid_graph
        torch.set_num_threads(2)
        g=SoftPrototypeGrouping()
        tensors={}
        g.assignment_observer=tensors.update
        with torch.no_grad():
            features,_=g(torch.randn(8,49,768))
            tensors['edge_index']=build_sparse_hybrid_graph(features)[0]
        tensors={k:v.detach().clone() for k,v in tensors.items()}
        ids=[str(i) for i in range(8)]
        ident=dict(split='val',sample_ids=ids,soft_assignment=self.m.ARCH['soft_assignment'])
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            (p/'probe_manifest.json').write_text(json.dumps(ident))
            (p/'assignment_diagnostics.jsonl').write_text('\n'.join(json.dumps(dict(epoch=i,split='val',sample_ids=ids)) for i in range(3)))
            for name,epoch in [('epoch_000.pt',0),('best.pt',1),('final.pt',2)]:
                torch.save(dict(epoch=epoch,identity=ident,tensors=tensors),p/name)
            result=self.m.check_probe(p,torch,g.candidate_mask,ids,1,final_epoch=2)
            self.assertTrue(result['actual_forward_checked'])
            for mode in ('normalize','edge','identity','epoch'):
                values=copy.deepcopy(tensors); identity=copy.deepcopy(ident); epoch=2
                if mode=='normalize': values['S']*=.5
                if mode=='edge': values['edge_index'][0,0]=16
                if mode=='identity': identity['split']='test'
                if mode=='epoch': epoch=1
                torch.save(dict(epoch=epoch,identity=identity,tensors=values),p/'final.pt')
                with self.subTest(mode=mode),self.assertRaises(RuntimeError): self.m.check_probe(p,torch,g.candidate_mask,ids,1,final_epoch=2)

    def test_duplicate_claim_preserves_live_exit_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            self.assertTrue(hasattr(self.m,'claim_run'),'Exclusive claim must precede failure-exit handling')
            self.m.claim_run(root)
            before=(root/'started.claim').read_bytes()
            with self.assertRaises(FileExistsError): self.m.claim_run(root)
            self.assertEqual((root/'started.claim').read_bytes(),before)
            self.assertFalse((root/'run.exit').exists())


if __name__=='__main__': unittest.main()
