"""Seed43 scheduling checks; subprocess fixtures are not GPU evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import copy
import types
from unittest.mock import patch
import hashlib
import io
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / 'tools' / (name + '.py')
    assert path.is_file(), 'Seed43 launcher not implemented'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Seed43Tests(unittest.TestCase):
    def test_nwpu_preflight_accepts_authorized_4090_and_rejects_other_cards(self):
        m = load('run_nwpu_seed43')
        row = 'GPU-b1911368-563d-916d-cb16-e7c2c9ead04a, NVIDIA GeForce RTX 4090, 610.57.04, 24564 MiB'
        manifest = dict(dataset='NWPU-RESISC45', counts={'train':5040,'val':1260,'test':25200},
                        records=[{'path':'fixture.jpg'}]*31500)
        with patch.object(m,'identity'),patch.object(m,'read',return_value=manifest),patch.object(Path,'is_file',return_value=True),\
             patch.object(m,'existing_runs',return_value=[]),patch.object(m,'idle_gpu'),patch.object(m,'output',return_value=row) as query:
            ready=m.preflight()
            self.assertEqual(ready['status'],'NWPU43_PREFLIGHT_PASS')
            self.assertEqual(ready['gpu'],row)
            self.assertFalse(ready['training_started'])
            for bad in (row.replace('RTX 4090','RTX 2080 Ti'),row+'\n'+row,
                        row.replace('GPU-b1911368-563d-916d-cb16-e7c2c9ead04a',''),
                        row.replace('24564 MiB','unknown')):
                with self.subTest(gpu=bad):
                    query.return_value=bad
                    with self.assertRaises(RuntimeError): m.preflight()

    def test_bigearth_preflight_payload_hash_is_current(self):
        m=load('run_bigearth_soft43')
        p=ROOT/'results/evidence/seed43_gpu4090_20261011/preflight_bigearth43.py'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),m.PREFLIGHT_SHA)

    def test_nwpu_failed_report_still_discovers_existing_checkpoint(self):
        m=load('run_nwpu_seed43')
        self.assertTrue(hasattr(m,'existing_runs'),'Partial checkpoint discovery not implemented')
        with tempfile.TemporaryDirectory() as td:
            work=Path(td)
            dest=work/'nwpu_old'/'soft'; dest.mkdir(parents=True)
            (dest/'run.json').write_text(json.dumps({'status':'failed','phase':'calibrate'}))
            meta=dict(training={'seed':43},data={'manifest_sha256':m.MANIFEST_SHA},execution={'phase':'calibrate'})
            (dest/'best.meta.json').write_text(json.dumps(meta))
            prior=m.existing_runs(work)
            self.assertEqual(len(prior),1)
            self.assertEqual(prior[0]['path'],str(dest/'best.meta.json'))
            meta['training']['seed']=42
            (dest/'best.meta.json').write_text(json.dumps(meta))
            self.assertEqual(m.existing_runs(work),[])

    def test_nwpu_actual_receipts_use_current_best_not_seed42_epochs(self):
        m = load('run_nwpu_seed43')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shared = dict(training={'seed': 43}, code={'frozen': True}, hardware={'gpu': 'fixture'})
            for name, best in zip(m.NAMES, (7, 9, 11)):
                for phase in ('calibrate', 'refit'):
                    dest = root / (name + '_' + phase)
                    dest.mkdir()
                    ckpt = dest / ('best.pth' if phase == 'calibrate' else 'last.pth')
                    ckpt.write_bytes((name + phase).encode())
                    r = dict(**shared, status='complete', phase=phase, seed=43, model='swin' if name == 'swin' else 'gavit',
                             manifest_sha256=m.MANIFEST_SHA, pretrained_sha256=m.PRE_SHA, reload_max_logit_diff=0.,
                             checkpoint_sha256=m.digest(ckpt), architecture={} if name == 'swin' else {'grouping': 'attentive_spatial' if name == 'fixed' else 'soft_prototype'},
                             graph_capture={'passed': True}, counts={'train':5040,'val':1260} if phase == 'calibrate' else {'train':6300,'val':0},
                             final_epoch=30 if phase == 'calibrate' else best, best_epoch=best if phase == 'calibrate' else None)
                    if phase == 'refit':
                        cal = root / (name + '_calibrate')
                        r['refit_source'] = dict(run_json_sha256=m.digest(cal/'run.json'), best_checkpoint_sha256=m.read(cal/'run.json')['checkpoint_sha256'])
                    (dest/'run.json').write_text(json.dumps(r))
                    (dest/'run.exit').write_text('exit_code=0\n')
                    history=[dict(epoch=i,train_loss=.1,validation={'oa_percent':90.-abs(i-best)} if phase=='calibrate' else None) for i in range(1,r['final_epoch']+1)]
                    (dest/'history.json').write_text(json.dumps(history))
                    (dest/'last.pth').touch(exist_ok=True)
                    (dest/'last.train_state.pth').write_bytes(b'fixture')
                    m.check_stage(root, name + '_' + phase)
                r['final_epoch'] = 29
                (dest/'run.json').write_text(json.dumps(r))
                with self.assertRaisesRegex(RuntimeError, 'Wrong refit'):
                    m.check_stage(root, name + '_refit')
                r['final_epoch']=best
                (dest/'run.json').write_text(json.dumps(r))
                (dest/'history.json').unlink()
                with self.assertRaises((FileNotFoundError,RuntimeError)):
                    m.check_stage(root, name + '_refit')

    def test_nwpu_own_calibration_routes_and_nine_stages(self):
        m = load('run_nwpu_seed43')
        jobs = m.jobs(Path('/output'))
        self.assertEqual([s for s, _ in jobs], [n + '_' + p for p in ('calibrate', 'refit', 'test') for n in ('swin', 'fixed', 'soft')])
        for n, argv in jobs:
            if not n.endswith('_test'):
                self.assertEqual(argv[argv.index('--seed') + 1], '43')
            if n.endswith('_refit'):
                self.assertEqual(argv[argv.index('--calibration') + 1], '/output/' + n.split('_')[0] + '_calibrate')
                self.assertNotIn('--epochs', argv)
            if n.endswith('_test'):
                self.assertEqual(argv[argv.index('--checkpoint') + 1], '/output/' + n.split('_')[0] + '_refit/last.pth')

    def test_nwpu_real_child_nonzero_stops_and_preserves_logs(self):
        m = load('run_nwpu_seed43')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            jobs = [('first', [sys.executable, '-c', 'print("first batch fixture"); raise SystemExit(7)']),
                    ('second', [sys.executable, '-c', 'print("should not run")'])]
            with self.assertRaises(subprocess.CalledProcessError):
                m.run_jobs(root, jobs, lambda: None, lambda name: None)
            self.assertEqual((root / 'first.process.exit').read_text().strip(), 'exit_code=7')
            self.assertIn('first batch fixture', (root / 'first.process.log').read_text())
            self.assertFalse((root / 'second.process.log').exists())
            before = (root / 'first.process.log').read_bytes()
            with self.assertRaises(FileExistsError):
                m.run_jobs(root, jobs, lambda: None, lambda name: None)
            self.assertEqual((root / 'first.process.log').read_bytes(), before)

    def test_nwpu_exit_zero_with_incomplete_artifact_stops(self):
        m = load('run_nwpu_seed43')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def reject(name):
                raise RuntimeError('Missing completed run.json')
            with self.assertRaisesRegex(RuntimeError, 'Missing completed'):
                m.run_jobs(root, [('first', [sys.executable, '-c', 'pass']), ('second', [sys.executable, '-c', 'pass'])], lambda: None, reject)
            self.assertFalse((root / 'second.process.log').exists())

    def test_bigearth_only_soft43_and_correct_best_weight_route(self):
        m = load('run_bigearth_soft43')
        b = load('run_bigearth_seed45')
        b.REPO = m.REPO
        argv = m.entry_args(Path('/output'), 'gavit_train', b)
        for k, v in {'--seed': '43', '--grouping': 'soft_prototype', '--epochs': '30', '--run_stage': 'formal'}.items():
            self.assertEqual(argv[argv.index(k) + 1], v)
        test = m.entry_args(Path('/output'), 'gavit_test', b)
        self.assertEqual(test[test.index('--seed') + 1], '43')
        self.assertEqual(test[test.index('--ckpt') + 1], argv[argv.index('--checkpoint_path') + 1])
        self.assertEqual(m.TRAINING['seed'], 43)
        with self.assertRaises(RuntimeError):
            m.entry_args(Path('/output'), 'swin_train', b)

    def test_bigearth_complete_report_and_matching_test_required(self):
        m = load('run_bigearth_soft43')
        b = load('run_bigearth_seed45')
        ready = dict(hardware={'gpu':'fixture'}, precision={'amp':False})
        def report(stage):
            r = dict(status='complete', stage=stage, model='gavit', seed=43, commit=m.COMMIT,
                     architecture=m.ARCH, pretrained_sha256=b.PRETRAINED_SHA,
                     splits={k:list(v) for k,v in b.FROZEN.items()}, **ready, checkpoint_sha256='a'*64)
            if stage == 'gavit_train':
                r.update(final_epoch=30, parameters=31466766)
            else:
                r['metrics'] = dict(samples=119825, f1_threshold=.5, checkpoint_sha256='a'*64,
                    mAP_percent=70., macro_f1_percent=65., micro_f1_percent=77., per_class_ap_percent={str(i):70. for i in range(19)})
            return r
        for failure in ('none', 'nonzero', 'incomplete', 'weight'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as td:
                root=Path(td); jobs=[]
                for stage in ('gavit_train','gavit_test'):
                    r=report(stage)
                    if failure=='incomplete' and stage=='gavit_train': r['final_epoch']=21
                    if failure=='weight' and stage=='gavit_test':
                        r['checkpoint_sha256']=r['metrics']['checkpoint_sha256']='b'*64
                    code='import pathlib,sys; (pathlib.Path(sys.argv[1])/"run.json").write_text(sys.argv[2]); sys.exit(int(sys.argv[3]))'
                    jobs.append((stage,[sys.executable,'-c',code,str(root/stage),json.dumps(r),'7' if failure=='nonzero' and stage=='gavit_train' else '0']))
                b.idle_gpu=lambda:None
                b.identity=lambda:None
                if failure=='none':
                    self.assertEqual(list(m.run_stages(root,jobs,b,ready)), ['gavit_train','gavit_test'])
                else:
                    with self.assertRaises((RuntimeError,subprocess.CalledProcessError)):
                        m.run_stages(root,jobs,b,ready)
                    if failure!='weight': self.assertFalse((root/'gavit_test').exists())

    def test_bigearth_seed45_metadata_cannot_be_reused_as_seed43(self):
        m=load('run_bigearth_soft43')
        history=[dict(epoch=i,val_mAP_percent=80.-abs(i-10),train_loss=.1,val_macro_f1_percent=60.,next_epoch_lr=.001,epoch_wall_seconds=2.) for i in range(1,31)]
        meta=dict(model='gavit',dataset='BigEarthNet-19',architecture=m.ARCH,training=copy.deepcopy(m.TRAINING),
                  execution={'run_stage':'formal'},git={'commit':m.COMMIT[:7],'dirty':False},best={'metric':'val_mAP','epoch':10,'value':80.})
        m.check_training(meta,history,30)
        meta['training']['seed']=45
        with self.assertRaises(RuntimeError): m.check_training(meta,history,30)

    def test_bigearth_preflight_preserves_historical_epoch21_boundary(self):
        source = (ROOT/'results/evidence/seed43_gpu4090_20261011/preflight_bigearth43.py').read_text()
        with tempfile.TemporaryDirectory() as td:
            work=Path(td)/'work'; work.mkdir()
            data=Path(td)/'data/BigEarthNet-S2'; data.mkdir(parents=True)
            old=work/'bigearth_seed45_20260922_01'; old.mkdir()
            helper='''import hashlib,json\nfrom pathlib import Path\nFROZEN={"train":(237871,"a"),"val":(122342,"b"),"test":(119825,"c")}\nSPLITS=Path("/fixture")\ndef identity(): pass\ndef idle_gpu(): pass\ndef output(argv): return "GPU-fixture, NVIDIA GeForce RTX 4090, 610.57.04"\ndef digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()\ndef scan_split(path,count,sha,seen): return dict(count=count,sha256=sha,examples=[])\n'''
            (old/'run_seed45.py').write_text(helper)
            script=source.replace('/home/featurize/work',str(work)).replace('/home/featurize/data',str(Path(td)/'data')).replace('8ac0083f41ccfb38555ed428d99971d075cb1d7ec3e413b4fa56548dcc1d0369',hashlib.sha256(helper.encode()).hexdigest())
            torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True,device_count=lambda:1,get_device_name=lambda i:'NVIDIA GeForce RTX 4090'),
                version=types.SimpleNamespace(cuda='fixtureCUDA'),backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(version=lambda:1,allow_tf32=True),cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False))),
                set_num_threads=lambda i:None,get_num_threads=lambda:8,get_default_dtype=lambda:'float32',load=lambda p,**kw:json.loads(Path(p).read_text()))
            reference=dict(hardware=dict(gpu='NVIDIA GeForce RTX 2080 Ti',cuda='fixtureCUDA',cudnn=1,threads=8,python=sys.version,
                packages={n:'fixture' for n in ('torch','torchvision','timm','torch-geometric','numpy','scikit-learn','pillow')},gpu_identity='uuid, NVIDIA GeForce RTX 2080 Ti, driver'),
                precision=dict(dtype='float32',matmul_tf32=False,cudnn_tf32=True,amp=False))
            ref=work/'bigearth_soft45_20260930_01'; ref.mkdir(); (ref/'preflight.json').write_text(json.dumps(reference))
            base=work/'GAViT_Project'; (base/'checkpoints').mkdir(parents=True); (base/'logs').mkdir()
            for model,stem,log,epoch,best,metrics in (
                ('swin','best_bigearth_swin_final_multiseed_bb9f1cf_seed43_retry1','final_multiseed_swin_bb9f1cf_seed43_retry1_test.log',30,6,('71.1%','64.1%','76.0%')),
                ('gavit','best_bigearth_gavit_sparse_hybrid_4n_top2_final_multiseed_bb9f1cf_seed43_retry1','final_multiseed_sparse_hybrid_bb9f1cf_seed43_retry1_test_retry2.log',21,8,('71.7%','65.7%','76.8%'))):
                ckpt=base/'checkpoints'/(stem+'.pth'); ckpt.write_bytes(model.encode())
                m=dict(dataset='BigEarthNet-19',model=model,git={'commit':'bb9f1cf','dirty':False},training=dict(seed=43,epochs=30,batch_size=32,lr=.0003,weight_decay=.0001,optimizer='AdamW',scheduler='CosineAnnealingLR',loss='BCEWithLogitsLoss',f1_threshold=.5),execution={'run_stage':'formal'},best={'epoch':best},checkpoint_sha256=hashlib.sha256(model.encode()).hexdigest(),
                    architecture=dict(num_regions=16,knn_k=5,gat_hidden=256,gat_heads=4,gat_layers=2,dropout=.1,grouping='attentive_spatial',edge_type='sparse_hybrid',integration='token_feedback'))
                ckpt.with_suffix('.meta.json').write_text(json.dumps(m))
                ckpt.with_suffix('.last.train_state.pth').write_text(json.dumps({'epoch':epoch}))
                (base/'logs'/log).write_text('\n'.join(k+v for k,v in zip(('Test mAP   (macro): ','Test F1    (macro): ','Test F1    (micro): '),metrics)))
            def run():
                ns={'__name__':'fixture'}
                with patch.dict(sys.modules,{'torch':torch}),patch('importlib.metadata.version',return_value='fixture'),redirect_stdout(io.StringIO()):
                    exec(compile(script,'<preflight-fixture>','exec'),ns)
                return ns['report']
            r=run()
            self.assertEqual(r['controls']['gavit']['final_epoch'],21)
            self.assertIn('different code',r['comparison_limitations'])
            self.assertEqual(r['hardware']['gpu'],'NVIDIA GeForce RTX 4090')
            self.assertTrue(r['physical_gpu_identity_changed'])
            self.assertTrue(r['gpu_model_changed'])
            torch.cuda.get_device_name=lambda i:'NVIDIA GeForce RTX 2080 Ti'
            with self.assertRaisesRegex(AssertionError,'RTX 4090'): run()
            torch.cuda.get_device_name=lambda i:'NVIDIA GeForce RTX 4090'
            torch.version.cuda='changedCUDA'
            with self.assertRaisesRegex(AssertionError,'Software'): run()
            torch.version.cuda='fixtureCUDA'
            torch.backends.cuda.matmul.allow_tf32=True
            with self.assertRaisesRegex(AssertionError,'Precision'): run()
            torch.backends.cuda.matmul.allow_tf32=False
            ckpt.with_suffix('.last.train_state.pth').write_text(json.dumps({'epoch':30}))
            with self.assertRaisesRegex(AssertionError,'saved epoch'):
                run()


if __name__ == '__main__':
    unittest.main()
