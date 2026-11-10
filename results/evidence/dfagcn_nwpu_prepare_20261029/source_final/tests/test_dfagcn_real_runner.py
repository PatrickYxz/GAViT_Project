"""Offline resource/selection/ZIP/GPU gates; no paid GPU work in tests."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import zipfile


ROOT=Path(__file__).resolve().parents[1]


def module():
    path=ROOT/'tools/run_dfagcn_nwpu_smoke.py'
    if not path.exists():
        raise AssertionError('real NWPU runner is not implemented')
    spec=importlib.util.spec_from_file_location('dfagcn_real_runner_test',path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_manifest():
    names=[f'class_{i:02}' for i in range(45)]
    rows=[]
    for label,name in enumerate(names):
        for split,count in [('train',112),('val',28),('test',560)]:
            rows.extend({'path':f'{name}/{split}_{i:03}.jpg','label':label,
                         'split':split,'sha256':'1'*64} for i in range(count))
    return {'schema_version':1,'dataset':'NWPU-RESISC45','class_names':names,
            'counts':{'train':5040,'val':1260,'test':25200},'records':rows}


class RealRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_frozen_manifest_selects_exact_360_90_without_test(self):
        value=module()
        p=self.root/'manifest.json';p.write_text(json.dumps(fixture_manifest()))
        manifest=value.load_manifest(p,expected_sha256=sha(p))
        selected=value.select_smoke(manifest)
        self.assertEqual({k:len(v) for k,v in selected.items()},{'train':360,'val':90,'pool':450})
        self.assertFalse(any(r['split']=='test' for r in selected['pool']))
        self.assertEqual(selected['pool'],sorted(selected['train']+selected['val'],key=lambda r:r['path']))
        with self.assertRaisesRegex(ValueError,'SHA'):
            value.load_manifest(p,expected_sha256='2'*64)

    def test_manifest_bad_counts_or_duplicate_or_classmap_is_rejected(self):
        value=module()
        for kind in ('counts','duplicate','classmap'):
            manifest=fixture_manifest()
            if kind=='counts':manifest['counts']['train']=5039
            elif kind=='duplicate':manifest['records'][1]=copy.deepcopy(manifest['records'][0])
            else:manifest['records'][0]['label']=1
            path=self.root/f'{kind}.json';path.write_text(json.dumps(manifest))
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                value.load_manifest(path,expected_sha256=sha(path))

    def zip_fixture(self,members):
        archive=self.root/'images.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for name,raw in members:z.writestr(name,raw)
        rows=[{'path':'class_a/a.jpg','sha256':hashlib.sha256(b'a').hexdigest()},
              {'path':'class_b/b.jpg','sha256':hashlib.sha256(b'b').hexdigest()}]
        return archive,rows

    def test_archive_restores_only_selected_members_and_checks_image_bytes(self):
        value=module()
        archive,rows=self.zip_fixture([('NWPU-RESISC45/class_a/a.jpg',b'a'),
                                      ('NWPU-RESISC45/class_b/b.jpg',b'b'),
                                      ('NWPU-RESISC45/class_c/test.jpg',b'not-selected')])
        result=value.recover_selected(archive,rows,self.root/'selected',expected_sha256=sha(archive))
        self.assertEqual(sorted(p.relative_to(result).as_posix() for p in result.rglob('*.jpg')),
                         ['class_a/a.jpg','class_b/b.jpg'])
        self.assertTrue(value.validate_selected(result,rows))
        with self.assertRaises(FileExistsError):
            value.recover_selected(archive,rows,result,expected_sha256=sha(archive))
        (result/'class_a/a.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA'):
            value.validate_selected(result,rows)

    def test_archive_wrong_global_hash_or_unsafe_or_ambiguous_member_is_rejected(self):
        value=module()
        for kind in ('hash','traversal','ambiguous'):
            folder=self.root/kind;folder.mkdir()
            archive=folder/'input.zip'
            members=[('class_a/a.jpg',b'a'),('class_b/b.jpg',b'b')]
            if kind=='traversal':members.append(('../outside.jpg',b'x'))
            if kind=='ambiguous':members.append(('NWPU-RESISC45/class_a/a.jpg',b'a'))
            with zipfile.ZipFile(archive,'w') as z:
                for name,raw in members:z.writestr(name,raw)
            rows=[{'path':'class_a/a.jpg','sha256':hashlib.sha256(b'a').hexdigest()},
                  {'path':'class_b/b.jpg','sha256':hashlib.sha256(b'b').hexdigest()}]
            expected='0'*64 if kind=='hash' else sha(archive)
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                value.recover_selected(archive,rows,folder/'output',expected_sha256=expected)
            self.assertFalse((self.root/'outside.jpg').exists())

    def test_archive_bad_selected_content_remains_incomplete(self):
        value=module()
        archive,rows=self.zip_fixture([('class_a/a.jpg',b'wrong'),('class_b/b.jpg',b'b')])
        with self.assertRaisesRegex(ValueError,'SHA'):
            value.recover_selected(archive,rows,self.root/'incomplete',expected_sha256=sha(archive))
        self.assertFalse((self.root/'incomplete'/'recovery.json').exists())

    def test_gpu_gate_requires_exact_uuid_4090_and_idle_card(self):
        value=module()
        inventory={'gpus':[{'name':'NVIDIA GeForce RTX 4090','uuid':'GPU-expected','memory_total_mib':24564}],
                   'processes':[],'error':None}
        self.assertEqual(value.require_gpu('GPU-expected',inventory)['uuid'],'GPU-expected')
        for kind in ('uuid','model','busy'):
            changed=copy.deepcopy(inventory)
            if kind=='uuid':changed['gpus'][0]['uuid']='GPU-other'
            elif kind=='model':changed['gpus'][0]['name']='NVIDIA RTX 3080'
            else:changed['processes']=[{'gpu_uuid':'GPU-expected','pid':123}]
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                value.require_gpu('GPU-expected',changed)

    def test_full_weight_sha_required_and_metadata_conflicts_rejected(self):
        value=module()
        metadata=self.root/'weights.json'
        expected='397923af'+'1'*56
        metadata.write_text(json.dumps({'sha256':expected}))
        self.assertEqual(value.read_weights_sha(metadata,None),expected)
        with self.assertRaises(ValueError):value.read_weights_sha(metadata,'397923af'+'2'*56)
        metadata.write_text(json.dumps({'sha256':'397923af'}))
        with self.assertRaises(ValueError):value.read_weights_sha(metadata,None)

    def test_gpu_selection_requires_fresh_cuda_context_before_environment_change(self):
        value=module()
        fake=types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda:True))
        args=types.SimpleNamespace(cpu_threads=1)
        with mock.patch.dict('sys.modules',{'torch':fake}),mock.patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'GPU-old'}):
            try:
                value._verified_imports_and_device(args,{'uuid':'GPU-new'})
            except ValueError as exc:
                self.assertIn('fresh',str(exc))
            except Exception as exc:
                self.fail('initialized CUDA context was not rejected before imports: '+type(exc).__name__)
            else:
                self.fail('initialized CUDA context was accepted')
            import os
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'GPU-old')

    def test_inspect_missing_resources_is_stdlib_read_only_and_never_trains(self):
        value=module()
        args=value.parser().parse_args(['inspect','--manifest',str(self.root/'missing.json'),
            '--data-root',str(self.root/'missingdata'),'--archive',str(self.root/'missing.zip'),
            '--weights',str(self.root/'missing.pth'),'--weights-metadata',str(self.root/'missingweights.json'),
            '--output',str(self.root/'run')])
        with mock.patch.object(value,'gpu_inventory',return_value={'gpus':[],'processes':[],'error':'not running'}):
            report=value.inspect_resources(args)
        self.assertFalse(report['ready'])
        self.assertFalse((self.root/'run').exists())
        self.assertIn('manifest',report['blockers'])
        self.assertFalse(report['training_started'])

    def test_cli_run_requires_uuid_before_any_resource_mutation(self):
        value=module()
        with mock.patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit) as result:
            value.parser().parse_args(['run'])
        self.assertEqual(result.exception.code,2)

    def test_config_file_changes_after_binding_are_rejected(self):
        value=module()
        self.assertTrue(hasattr(value,'require_unchanged_recipe'),
                        'recipe file must remain bound to its pre-run checksum')
        path=self.root/'recipe.json'
        path.write_text('{"epochs":1}')
        expected=sha(path)
        self.assertEqual(value.require_unchanged_recipe(path,expected),expected)
        path.write_text('{"epochs":2}')
        with self.assertRaisesRegex(ValueError,'recipe'):
            value.require_unchanged_recipe(path,expected)


if __name__=='__main__':unittest.main()
