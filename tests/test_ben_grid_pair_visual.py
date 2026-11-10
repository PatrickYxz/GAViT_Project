"""Identity, topology, paired-outcome and export checks without torch."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / 'tools/ben_grid_pair_visual.py'
    if not path.exists():
        raise AssertionError('paired trained-model capture/render tool is missing')
    spec = importlib.util.spec_from_file_location('ben_grid_pair_visual', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def edges(k):
    side = {9: 3, 16: 4}[k]
    spatial, feature = [], []
    for q in range(k):
        near = lambda s: abs(s//side-q//side) + abs(s%side-q%side)
        spatial.extend([s, q] for s in range(k) if near(s) == 1)
        feature.extend([s, q] for s in range(k) if s != q and near(s) > 1)
    feature = [pair for q in range(k) for pair in [e for e in feature if e[1] == q][:2]]
    return spatial + feature


def model_record(m, k):
    a = m.partition(k)
    kinds = m.validate_topology(a, edges(k), k)
    return dict(num_regions=k, assignments=a, edges=edges(k), edge_kinds=kinds,
                probabilities=[.25]*19, pooling_weights=[
                    [1/a.count(r) if region == r else 0. for r in range(k)] for region in a],
                gate_max_abs_errors=dict(capture=0., restoration=0., pooling=0., feedback=0.))


def fixture_summary(m):
    source = dict(snapshot_sha256='frozen')
    metrics = dict(samples=119825, checkpoint_sha256='K9')
    common = dict(status='complete', seed=42, checkpoint_sha256='K9', source=source)
    return dict(status='complete', dataset='BigEarthNet-19', seed=42, stages=dict(
        smoke=dict(common, final_epoch=1, reload_max_logit_diff=0.),
        train=dict(common, final_epoch=30, counts=dict(train=237871, val=122342)),
        test=dict(common, metrics=metrics)), control=dict(checkpoint_sha256='K16'))


class GridPairVisualTests(unittest.TestCase):
    def test_fixed_membership_and_actual_edge_budgets_for_both_grids(self):
        m = load()
        for k, counts in ((9, [9,6,6,6,4,4,6,4,4]),
                          (16, [4,4,4,2,4,4,4,2,4,4,4,2,2,2,2,1])):
            a = m.partition(k)
            self.assertEqual([a.count(i) for i in range(k)], counts)
            kinds = m.validate_topology(a, edges(k), k)
            self.assertEqual(kinds.count('spatial'), 24 if k == 9 else 48)
            self.assertEqual(kinds.count('feature'), 2*k)
        for k in (True, 9.0, 4, 25):
            with self.assertRaises(ValueError):
                m.partition(k)

    def test_topology_rejects_fake_assignments_duplicates_and_reversed_feature_flow(self):
        m = load()
        a, e = m.partition(9), edges(9)
        bad_assignment = a.copy(); bad_assignment[0] = 1
        invalid = [e[:-1], e[:-1]+[e[0]], e[:24]+[[t,s] for s,t in e[24:]],
                   e[:-1]+[[True,0]], e[:-1]+[[0,9]]]
        for case in invalid:
            with self.assertRaises(ValueError):
                m.validate_topology(a, case, 9)
        with self.assertRaises(ValueError):
            m.validate_topology(bad_assignment, e, 9)

    def test_outcomes_include_both_directions_false_positives_and_unchanged_cases(self):
        m = load()
        labels = [1,0,1,0] + [0]*15
        p9 = [.4,.2,.7,.8] + [.1]*15
        p16 = [.6,.7,.8,.9] + [.2]*15
        rows = m.compare_predictions(labels, p9, p16, [str(i) for i in range(19)])
        self.assertEqual([r['outcome'] for r in rows[:4]],
                         ['k16_only_correct','k9_only_correct','both_correct','both_wrong'])
        self.assertEqual(len(rows), 19)
        self.assertAlmostEqual(rows[0]['delta_probability'], .2)
        for bad in ([float('nan')]*19, [-.1]*19, [1.1]*19, [.1]*18):
            with self.assertRaises(ValueError):
                m.compare_predictions(labels, bad, p16, [str(i) for i in range(19)])

    def test_summary_must_bind_seed_epochs_source_and_both_checkpoints(self):
        m = load(); s = fixture_summary(m)
        m.validate_summary(s, 'K9', 'K16', 'frozen')
        mutations = [('seed',43), ('status','running')]
        for key, value in mutations:
            bad = copy.deepcopy(s); bad[key] = value
            with self.assertRaises(ValueError):
                m.validate_summary(bad, 'K9', 'K16', 'frozen')
        for stage, key, value in [('train','final_epoch',29), ('test','checkpoint_sha256','wrong'),
                                  ('smoke','reload_max_logit_diff',float('nan'))]:
            bad = copy.deepcopy(s); bad['stages'][stage][key] = value
            with self.assertRaises(ValueError):
                m.validate_summary(bad, 'K9', 'K16', 'frozen')
        with self.assertRaises(ValueError):
            m.validate_summary(s, 'K9', 'K16', 'other_source')

    def test_pooling_weights_keep_membership_and_unit_column_mass(self):
        m = load(); row = model_record(m, 9)
        m.validate_pooling(row['assignments'], row['pooling_weights'], 9)
        for i,j,value in ((0,0,-.1), (0,1,.1), (0,0,float('nan')), (0,0,1.)):
            w = copy.deepcopy(row['pooling_weights']); w[i][j] = value
            with self.assertRaises(ValueError):
                m.validate_pooling(row['assignments'], w, 9)

    def test_svg_preserves_input_bytes_and_every_captured_edge_with_direction(self):
        m = load(); png = b'lossless archived input fixture'
        row = dict(row_index=4439, labels=[0]*19,
                   models={'9': model_record(m,9), '16': model_record(m,16)})
        svg = m.render_svg(row, png, fixture=True)
        tree = ET.fromstring(svg)
        imgs = [n for n in tree.iter() if n.tag.endswith('image')]
        self.assertEqual(len(imgs), 3)
        for n in imgs:
            value = n.get('href')
            self.assertEqual(base64.b64decode(value.split(',',1)[1]), png)
        drawn = [(int(n.get('data-k')),int(n.get('data-source')),int(n.get('data-target')))
                 for n in tree.iter() if n.get('data-source') is not None]
        self.assertEqual(sorted(drawn), sorted((k,s,t) for k in (9,16) for s,t in edges(k)))
        self.assertIn('FIXTURE', svg)

    def test_installer_refuses_overwrite_path_escape_and_modified_content(self):
        m = load()
        data = dict(files={'ben_grid_pair_visual.py':'print(1)\n','config.json':'{}\n'})
        data['sha256'] = {name:m.sha_bytes(text.encode()) for name,text in data['files'].items()}
        with tempfile.TemporaryDirectory() as td:
            target = Path(td)/'new'; m.install(data,target)
            with self.assertRaises(FileExistsError):
                m.install(data,target)
            self.assertEqual((target/'config.json').read_text(), '{}\n')
            for name in ('../escape','/tmp/escape'):
                bad = copy.deepcopy(data); bad['files'][name] = 'bad'
                with self.assertRaises(ValueError):
                    m.install(bad,Path(td)/'escape_target')
            bad = copy.deepcopy(data); bad['files']['config.json'] = 'changed'
            with self.assertRaises(ValueError):
                m.install(bad,Path(td)/'tampered_target')

    def test_packager_pins_original_pngs_and_keeps_command_below_shell_argument_limit(self):
        path = ROOT/'tools/package_ben_grid_pair_visual.py'
        self.assertTrue(path.exists(), 'one-line visual handoff packager is missing')
        spec = importlib.util.spec_from_file_location('package_ben_grid_pair_visual',path)
        p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
        payload = p.make_payload(ROOT.parents[1])
        cfg = json.loads(payload['files']['config.json'])
        self.assertEqual([r['row_index'] for r in cfg['cases']], [97389,14680,28450,4439])
        self.assertEqual(cfg['device'],'cpu')
        self.assertEqual(len(cfg['classes']),19)
        for row in cfg['cases']:
            self.assertTrue(row['png_path'].startswith('/home/featurize/work/pair44_feedback_k_7ehbne/'))
            self.assertEqual(len(row['png_sha256']),64)
        command = p.make_command(payload)
        self.assertNotIn('\n',command)
        self.assertLess(len(command.encode()), 64000)
        import shlex
        argv = shlex.split(command)
        self.assertEqual(argv[:3],['python','-B','-c'])
        compile(argv[3],'<one-line installer>','exec')

    def test_artifact_gate_reads_persistent_files_and_rejects_changed_weights_before_inference(self):
        m = load()
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); training=base/'training'; code=training/'code'
            code.mkdir(parents=True)
            for name in ('smoke','train','test'):
                (training/name).mkdir(); (training/name/'run.exit').write_text('exit_code=0\n')
            (training/'run.exit').write_text('exit_code=0\n')
            (code/'dummy.py').write_text('# fixture source\n')
            (code/'legacy_graph.py').write_text('# fixture graph\n')
            snapshot=dict(snapshot_sha256='frozen',files={'dummy.py':m.digest(code/'dummy.py')},
                          original_files={'models/graph_construction.py':m.digest(code/'legacy_graph.py')})
            (code/'snapshot.json').write_text(json.dumps(snapshot))
            paths={}
            for k in ('9','16'):
                path=base/f'K{k}.pth'; path.write_bytes(b'fixture weights without tensors '+k.encode())
                paths[k]=dict(path=str(path),sha256=m.digest(path))
            s=fixture_summary(m)
            for name in ('train','test'):
                s['stages'][name]['checkpoint_sha256']=paths['9']['sha256']
            s['stages']['test']['metrics']['checkpoint_sha256']=paths['9']['sha256']
            s['control']['checkpoint_sha256']=paths['16']['sha256']
            for k in ('9','16'):
                meta=dict(checkpoint_sha256=paths[k]['sha256'],architecture=dict(num_regions=int(k)),
                          training=dict(seed=42,epochs=30),best=dict(epoch=13),code=snapshot)
                Path(paths[k]['path']).with_suffix('.meta.json').write_text(json.dumps(meta))
                if k == '9': s['stages']['train']['metadata']=meta
                else: s['control']['metadata']=meta
            for name,obj in (('summary.json',s),('test/metrics.json',s['stages']['test']['metrics']),
                             ('preflight.json',{'fixture':True}),('train/run.json',{}),('test/run.json',{})):
                (training/name).write_text(json.dumps(obj))
            val=base/'val.csv'
            val.write_text('patch_path,'+','.join(f'label_{i}' for i in range(19))+'\n'+
                           ('/fixture/patch,'+','.join(['0']*19)+'\n')*122342)
            png=ROOT/'results/evidence/pair44_feedback_k_7ehbne/case_sheet_return_20260919/row_4439.png'
            cfg=dict(training_root=str(training),snapshot=snapshot,checkpoints=paths,architecture={},
                     val_csv=str(val),val_csv_sha256=m.digest(val),selection='fixture only',
                     reported_metrics=s['stages']['test']['metrics'],cases=[dict(row_index=i,labels=[0]*19,
                        patch_path='/fixture/patch',png_path=str(png),png_sha256=m.digest(png))
                        for i in (97389,14680,28450,4439)])
            output=base/'output'; output.mkdir()
            identities=m.preflight(output,cfg)
            self.assertEqual(set(identities),{'9','16'})
            self.assertTrue((output/'source_reports/summary.json').exists())
            self.assertEqual((output/'inputs/row_4439.png').read_bytes(),png.read_bytes())
            Path(paths['9']['path']).write_bytes(b'changed')
            other=base/'other'; other.mkdir()
            with self.assertRaisesRegex(ValueError,'Weights changed'):
                m.preflight(other,cfg)
            self.assertFalse((other/'inputs').exists())


if __name__ == '__main__':
    unittest.main()
