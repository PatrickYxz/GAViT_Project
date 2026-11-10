"""Missing-seed handoff safety; these checks do not run CUDA training."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Soft42Tests(unittest.TestCase):
    def test_formal_routes_train42_and_tests_its_best_checkpoint(self):
        m = load("run_bigearth_soft42")
        b = load("run_bigearth_seed45")
        b.REPO = m.REPO
        train = m.entry_args(Path("/output"), "gavit_train", b)
        test = m.entry_args(Path("/output"), "gavit_test", b)
        for flag, value in (("--seed", "42"), ("--epochs", "30"),
                            ("--grouping", "soft_prototype"), ("--run_stage", "formal")):
            self.assertEqual(train[train.index(flag) + 1], value)
        self.assertEqual(test[test.index("--seed") + 1], "42")
        self.assertEqual(test[test.index("--ckpt") + 1], "/output/gavit_train/gavit.pth")
        with self.assertRaises(RuntimeError):
            m.entry_args(Path("/output"), "swin_train", b)

    def test_old_smoke_does_not_hide_nested_formal_or_partial42(self):
        gate = load("preflight_bigearth_soft42")
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            records = (
                ("bigearth_smoke42/a.meta.json", "smoke", 42),
                ("bigearth_old/a/b/gavit.meta.json", "formal", 42),
                ("GAViT_SOFT_20260927/checkpoints/gavit42.meta.json", "formal", 42),
                ("bigearth_soft43/gavit.meta.json", "formal", 43),
            )
            for name, stage, seed in records:
                p = work / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(dict(model="gavit", dataset="BigEarthNet-19",
                    architecture=dict(grouping="soft_prototype"), training=dict(seed=seed),
                    execution=dict(run_stage=stage))))
            # A failed process can still have a reusable/partial formal checkpoint.
            (work / "bigearth_old/a/b/run.json").write_text(json.dumps(dict(status="failed")))
            inventory = gate.existing_soft42(work)
            self.assertEqual(len(inventory["formal_or_partial"]), 2)
            self.assertEqual(len(inventory["engineering_smoke"]), 1)

    def test_missing_stage_and_run_report_are_not_ignored(self):
        gate = load("preflight_bigearth_soft42")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bigearth_custom/gavit_train/run.json"
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(model="gavit", seed=42, architecture=dict(grouping="soft_prototype"),
                stage="gavit_train", status="running", splits=dict(train=[237871,"sha"]))))
            found = gate.existing_soft42(Path(td))
            self.assertEqual(len(found["formal_or_partial"]), 1)

    def test_soft42_weights_without_sidecar_require_review(self):
        gate = load("preflight_bigearth_soft42")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'bigearth_soft42_unknown/checkpoints/gavit.pth'
            p.parent.mkdir(parents=True)
            p.write_bytes(b'partial weights with lost metadata')
            inventory = gate.existing_soft42(Path(td))
            self.assertEqual(len(inventory['formal_or_partial']),1)
            self.assertEqual(p.read_bytes(),b'partial weights with lost metadata')

    def test_report_rejects_another_seed_incomplete_epochs_or_mismatched_map(self):
        m = load("run_bigearth_soft42")
        b = load("run_bigearth_seed45")
        ready = dict(hardware={"gpu":"fixture"}, precision={"amp":False})
        train = dict(status="complete", stage="gavit_train", model="gavit", seed=42,
            commit=m.COMMIT, architecture=copy.deepcopy(m.ARCH), pretrained_sha256=b.PRETRAINED_SHA,
            splits={k:list(v) for k,v in b.FROZEN.items()}, **ready, final_epoch=30, parameters=31466766)
        m.check_report(train, "gavit_train", ready, b)
        for key, value in (("seed",43),("final_epoch",21)):
            bad = {**train, key:value}
            with self.assertRaises(RuntimeError):
                m.check_report(bad, "gavit_train", ready, b)
        test = {**train, "stage":"gavit_test", "checkpoint_sha256":"a"*64,
            "metrics":dict(samples=119825,f1_threshold=.5,checkpoint_sha256="a"*64,
                mAP_percent=70.,macro_f1_percent=65.,micro_f1_percent=77.,
                per_class_ap_percent={str(i):70. for i in range(19)})}
        m.check_report(test, "gavit_test", ready, b)
        test["metrics"]["mAP_percent"] = 71.
        with self.assertRaises(RuntimeError):
            m.check_report(test, "gavit_test", ready, b)

    def test_duplicate_claim_preserves_original_marker(self):
        m = load("run_bigearth_soft42")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            m.claim_run(root)
            original = (root / "started.claim").read_bytes()
            with self.assertRaises(FileExistsError):
                m.claim_run(root)
            self.assertEqual((root / "started.claim").read_bytes(), original)

    def test_smoke_gate_checks_identity_exit_and_checkpoint_before_launch(self):
        m = load("handoff_bigearth_soft42")
        helper = load("run_bigearth_soft42")
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt = root / 'gavit.pth'
            ckpt.write_bytes(b'checkpoint fixture')
            sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
            ready = dict(hardware={'gpu':'fixture'}, precision={'amp':False})
            meta = dict(architecture=copy.deepcopy(helper.ARCH), training={**helper.TRAINING,'epochs':1},
                        execution=dict(run_stage='smoke'), git=dict(commit='ed480e6',dirty=False),
                        checkpoint_sha256=sha)
            ckpt.with_suffix('.meta.json').write_text(json.dumps(meta))
            summary = dict(status='complete',seed=42,commit=helper.COMMIT,counts=dict(train=96,val=64),epochs=1,
                reload_max_logit_diff=1e-7,checkpoint_sha256=sha,parameters=31466766,metadata=meta,**ready,
                graph=dict(edges_per_image=80,spatial_edges=48,feature_edges=32,actual_forward_checked=True),
                graph_probe=dict(actual_forward_checked=True,normalization_errors=dict(S_row=1e-7,W_column=1e-7)),
                train_loss=.5,val_mAP_percent=30.,val_macro_f1_percent=4.)
            (root/'summary.json').write_text(json.dumps(summary))
            (root/'run.exit').write_text('exit_code=0\n')
            m.check_smoke(root,ready,helper)
            for field, bad in (('seed',43),('reload_max_logit_diff',.1),('counts',dict(train=96,val=32))):
                (root/'summary.json').write_text(json.dumps({**summary,field:bad}))
                with self.assertRaises(RuntimeError):
                    m.check_smoke(root,ready,helper)
            (root/'summary.json').write_text(json.dumps(summary))
            ckpt.write_bytes(b'changed checkpoint')
            with self.assertRaises(RuntimeError):
                m.check_smoke(root,ready,helper)
            ckpt.write_bytes(b'checkpoint fixture')
            (root/'run.exit').write_text('exit_code=7\n')
            with self.assertRaises(RuntimeError):
                m.check_smoke(root,ready,helper)

    def test_completed_engineering_smoke_can_be_added_between_preflights(self):
        m = load('handoff_bigearth_soft42')
        ready = dict(status='BIGEARTH_SOFT42_PREFLIGHT_PASS',seed=42,
                     hardware=dict(gpu_identity='GPU-original'),controls=dict(swin=dict(checkpoint_sha256='a')),
                     inventory=dict(formal_or_partial=[],engineering_smoke=[]))
        fresh = {**ready,'inventory':dict(formal_or_partial=[],engineering_smoke=[dict(path='new smoke')])}
        m.check_preflight_continuity(ready,fresh)
        for changed in ({**fresh,'hardware':dict(gpu_identity='GPU-different')},
                        {**fresh,'status':'BIGEARTH_SOFT42_EXISTING_RESULT_REVIEW_REQUIRED'}):
            with self.assertRaises(RuntimeError):
                m.check_preflight_continuity(ready,changed)

    def test_smoke_process_returns_before_launch_checks_atexit_marker(self):
        # Reproduce the real wrapper boundary; replace only CUDA/data work with a
        # tiny child fixture. Reading run.exit inside smoke() must fail this test.
        import hashlib
        helper = load('run_bigearth_soft42')
        m = load('handoff_bigearth_soft42')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root/'smoke'
            ready = dict(status='BIGEARTH_SOFT42_PREFLIGHT_PASS',hardware={'gpu':'fixture'},precision={'amp':False},controls={})
            (root/'ready.json').write_text(json.dumps(ready))
            (root/'run_bigearth_soft42.py').write_text('# archived helper fixture\n')
            (root/'prepare_smoke_source.py').write_text('from pathlib import Path\nout=Path('+repr(str(output))+')\nout.mkdir()\n')
            sha = hashlib.sha256(b'fixture checkpoint').hexdigest()
            meta = dict(architecture=helper.ARCH,training={**helper.TRAINING,'epochs':1},
                execution={'run_stage':'smoke'},git={'commit':'ed480e6','dirty':False},checkpoint_sha256=sha)
            summary = dict(status='complete',seed=42,commit=helper.COMMIT,counts={'train':96,'val':64},epochs=1,
                parameters=31466766,reload_max_logit_diff=1e-7,train_loss=.5,val_mAP_percent=30.,val_macro_f1_percent=4.,
                hardware=ready['hardware'],precision=ready['precision'],metadata=meta,checkpoint_sha256=sha,
                graph={'edges_per_image':80,'spatial_edges':48,'feature_edges':32,'actual_forward_checked':True},
                graph_probe={'actual_forward_checked':True,'normalization_errors':{'S_row':1e-7,'W_column':1e-7}})
            train = ('import atexit,json\n'
                '(out/"gavit.pth").write_bytes(b"fixture checkpoint")\n'
                '(out/"gavit.meta.json").write_text('+repr(json.dumps(meta))+')\n'
                '(out/"summary.json").write_text('+repr(json.dumps(summary))+')\n'
                'atexit.register(lambda:(out/"run.exit").write_text("exit_code=0\\n"))\n')
            (root/'train_reload_smoke_source.py').write_text(train)
            child = '''import importlib.util,sys,types
from pathlib import Path
spec=importlib.util.spec_from_file_location('handoff',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.SMOKE=Path(sys.argv[3])
b=types.SimpleNamespace(identity=lambda:None,idle_gpu=lambda:None)
helper=types.SimpleNamespace(load_base=lambda:b,environment=lambda torch,base:({'gpu':'fixture'},{'amp':False}))
m.load=lambda root,name:helper
sys.modules['torch']=types.ModuleType('torch')
m.smoke(Path(sys.argv[2]))
'''
            completed = subprocess.run([sys.executable,'-B','-c',child,str(ROOT/'tools/handoff_bigearth_soft42.py'),str(root),str(output)],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertEqual((output/'run.exit').read_text(),'exit_code=0\n')
            self.assertEqual(m.check_smoke(output,ready,helper)['status'],'complete')

    def test_queue_preflight_saves_report_without_gate_global_b(self):
        # The42 gate defines b locally inside preflight(), unlike the43 gate.
        # Execute the real CLI branch against a stdlib-only preflight fixture.
        import ast
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            report=dict(status='BIGEARTH_SOFT42_PREFLIGHT_PASS',seed=42)
            script=('report='+repr(report)+'\n').encode()
            (root/'preflight.py').write_bytes(script)
            worker=(ROOT/'tools/run_bigearth_soft42.py').read_text()
            base=ROOT/'tools/run_bigearth_seed45.py'
            worker=worker.replace("OLD_TOOL = Path('/home/featurize/work/bigearth_seed45_20260922_01/run_seed45.py')",'OLD_TOOL = Path('+repr(str(base))+')')
            tree=ast.parse(worker)
            values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('OLD_SHA','PREFLIGHT_SHA')}
            worker=worker.replace(values['OLD_SHA'],hashlib.sha256(base.read_bytes()).hexdigest())
            worker=worker.replace(values['PREFLIGHT_SHA'],hashlib.sha256(script).hexdigest())
            path=root/'queue.py';path.write_text(worker)
            completed=subprocess.run([sys.executable,'-B',str(path),str(root),'preflight'],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertEqual(json.loads((root/'preflight.json').read_text()),report)


if __name__ == "__main__":
    unittest.main()
