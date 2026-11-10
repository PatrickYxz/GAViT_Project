"""Frozen BEN K9 supplement safety; tensor checks run separately on Featurize."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Grid42Tests(unittest.TestCase):
    def test_cuda_forward_gate_accepts_roundoff_but_rejects_large_or_nonfinite_error(self):
        checks=load('ben_grid42_tensor_checks')
        checks.check_logit_difference(0.)
        checks.check_logit_difference(2.384185791015625e-7)
        for value in (2e-5,float('nan'),float('inf'),-1.):
            with self.assertRaises(RuntimeError):
                checks.check_logit_difference(value)

    def test_topology_records_node_dependent_budget_and_rejects_unapproved_grids(self):
        p = load('ben_grid42_patch')
        for k, edges in ((9, (24, 18, 42)), (16, (48, 32, 80))):
            t = p.topology(k)
            self.assertEqual((t['spatial_directed_edges'], t['feature_directed_edges'],
                              t['total_directed_edges']), edges)
            self.assertEqual(t['message_direction'], 'neighbor_to_query')
        for k in (4, 25, 90, True, 9.0):
            with self.assertRaises(ValueError):
                p.topology(k)

    def test_exported_snapshot_has_honest_identity_and_detects_tampering(self):
        p = load('ben_grid42_patch')
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'code'
            manifest = p.export_snapshot(ROOT, target)
            p.verify_snapshot(target, manifest)
            ns = {'__name__': 'frozen_identity', '__file__': str(target / 'experiment_identity.py')}
            exec(compile((target / 'experiment_identity.py').read_bytes(), ns['__file__'], 'exec'), ns)
            self.assertEqual(ns['graph_topology_identity']('sparse_hybrid', knn_k=5, num_regions=9), p.topology(9))
            self.assertEqual(ns['graph_topology_identity']('sparse_hybrid', knn_k=5), p.topology(16))
            state = ns['get_git_state']()
            self.assertTrue(state['dirty'])
            self.assertEqual(state['commit'], 'cd39be9')
            ns['assert_clean_git_state'](state)
            (target / 'models/graph_construction.py').write_text('corrupted')
            with self.assertRaises(RuntimeError):
                p.verify_snapshot(target, manifest)
            with self.assertRaises(RuntimeError):
                ns['assert_clean_git_state'](state)
            with self.assertRaises(FileExistsError):
                p.export_snapshot(ROOT, target)

    def test_formal_command_trains_only_k9_and_tests_its_best(self):
        m = load('run_ben_grid42')
        root = Path('/example/run')
        train = m.entry_args(root, 'train')
        test = m.entry_args(root, 'test')
        smoke = m.entry_args(root, 'smoke')
        for flag, value in (('--seed', '42'), ('--epochs', '30'), ('--num_regions', '9'),
                            ('--grouping', 'attentive_spatial'), ('--edge_type', 'sparse_hybrid'),
                            ('--integration', 'token_feedback'), ('--run_stage', 'formal')):
            self.assertEqual(train[train.index(flag)+1], value)
        self.assertEqual(smoke[smoke.index('--epochs')+1], '1')
        self.assertEqual(test[test.index('--ckpt')+1], '/example/run/train/gavit.pth')
        self.assertNotEqual(smoke[smoke.index('--checkpoint_path')+1], test[test.index('--ckpt')+1])
        with self.assertRaises(ValueError):
            m.entry_args(root, 'swin_train')

    def test_existing_artifacts_are_preserved_on_duplicate_claim(self):
        m = load('run_ben_grid42')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            m.claim_run(root)
            (root / 'run.exit').write_text('exit_code=0\n')
            with self.assertRaises(FileExistsError):
                m.claim_run(root)
            self.assertEqual((root / 'run.exit').read_text(), 'exit_code=0\n')

    def test_metadata_and_best_selection_reject_wrong_seed_topology_or_epoch(self):
        m = load('run_ben_grid42')
        meta = dict(model='gavit', dataset='BigEarthNet-19', architecture=m.architecture(),
                    training=m.recipe(30), graph_topology=m.topology(9),
                    execution=dict(run_stage='formal'), best=dict(metric='val_mAP', value=79.0, epoch=2),
                    code=dict(snapshot_sha256='fixture'))
        rows = [dict(epoch=i, train_loss=.4, val_mAP_percent=79.0 if i in (2,3) else 78.0,
                     val_macro_f1_percent=60., next_epoch_lr=.0001, epoch_wall_seconds=1.) for i in range(1,31)]
        m.check_training(meta, rows, 30, 30, 'fixture')
        for field, value in (('training', {**m.recipe(30), 'seed':43}),
                             ('graph_topology',m.topology(16)),
                             ('best',dict(metric='val_mAP',value=79.,epoch=3))):
            bad=copy.deepcopy(meta);bad[field]=value
            with self.assertRaises(RuntimeError):
                m.check_training(bad,rows,30,30,'fixture')
        with self.assertRaises(RuntimeError):
            m.check_training(meta,rows,21,30,'fixture')

    def test_metric_validator_rejects_partial_or_inconsistent_test(self):
        m = load('run_ben_grid42')
        metrics = dict(samples=119825, mAP_percent=70., macro_f1_percent=64., micro_f1_percent=76.,
                       f1_threshold=.5, checkpoint_sha256='weights', per_class_ap_percent={str(i):70. for i in range(19)})
        m.check_metrics(metrics,'weights')
        for key,value in (('samples',96),('mAP_percent',71.),('checkpoint_sha256','other'),('micro_f1_percent',float('nan'))):
            with self.assertRaises(RuntimeError):
                m.check_metrics({**metrics,key:value},'weights')

    def test_stage_failure_stops_before_formal_and_saves_child_exit(self):
        m = load('run_ben_grid42')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with self.assertRaises(subprocess.CalledProcessError):
                m.run_stage(root,'smoke',['/usr/bin/false'])
            self.assertEqual((root/'smoke/run.exit').read_text(),'exit_code=1\n')
            self.assertFalse((root/'train').exists())

    def test_existing_other_k9_seed42_blocks_retraining_but_current_export_does_not(self):
        m=load('run_ben_grid42')
        with tempfile.TemporaryDirectory() as td:
            work=Path(td);own=work/'ben_grid3_seed42';own.mkdir()
            other=work/'bigearth_prior/checkpoints/model.meta.json';other.parent.mkdir(parents=True)
            meta=dict(model='gavit',dataset='BigEarthNet-19',architecture=dict(num_regions=9),
                      training=dict(seed=42),execution=dict(run_stage='formal'))
            other.write_text(json.dumps(meta));(own/'model.meta.json').write_text(json.dumps(meta))
            self.assertEqual(m.existing_k9(work,own),[dict(path=str(other),status='formal')])

    def test_installer_exports_exact_release_and_refuses_overwrite(self):
        p=load('ben_grid42_patch');installer=load('install_ben_grid42')
        with tempfile.TemporaryDirectory() as td:
            expected=p.export_snapshot(ROOT,Path(td)/'expected')
            files={name:(ROOT/'tools'/name).read_text() for name in
                   ('ben_grid42_patch.py','run_ben_grid42.py','ben_grid42_tensor_checks.py')}
            payload=dict(files=files,snapshot=expected)
            target=Path(td)/'installed'
            installer.install(payload,target,ROOT)
            self.assertEqual(json.loads((target/'release.json').read_text())['snapshot'],expected)
            self.assertFalse((target/'run.pid').exists())
            with self.assertRaises(FileExistsError):
                installer.install(payload,target,ROOT)
            self.assertEqual((target/'run_ben_grid42.py').read_text(),files['run_ben_grid42.py'])

    def test_installer_mismatch_never_creates_a_launch_marker(self):
        p=load('ben_grid42_patch');installer=load('install_ben_grid42')
        with tempfile.TemporaryDirectory() as td:
            expected=p.export_snapshot(ROOT,Path(td)/'expected');expected['snapshot_sha256']='incorrect'
            payload=dict(files={name:(ROOT/'tools'/name).read_text() for name in
                         ('ben_grid42_patch.py','run_ben_grid42.py','ben_grid42_tensor_checks.py')},snapshot=expected)
            target=Path(td)/'installed'
            with self.assertRaises(RuntimeError):
                installer.install(payload,target,ROOT)
            self.assertFalse((target/'run.pid').exists())


if __name__ == '__main__':
    unittest.main()
