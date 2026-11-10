"""Portable delivery exercises real zipapp extraction/build without importing Torch."""
import contextlib
import importlib
import inspect
import io
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile


def api():
    try:
        return importlib.import_module('tools.package_dfagcn_nwpu_smoke')
    except ModuleNotFoundError as exc:
        raise AssertionError('portable DFAGCN delivery builder is not implemented') from exc


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.runner=self.root/'runner.py'
        self.runner.write_text('import json,os,sys\nprint(json.dumps({"args":sys.argv[1:],"cwd":os.getcwd(),"delivery_entry":os.environ.get("DFAGCN_DELIVERY_ENTRY")}))\n')
        self.bundle=self.root/'delivery.pyz'

    def build(self,**kwargs):
        return api().build_bundle(self.bundle,runner_path=self.runner,**kwargs)

    def invoke(self,*args,bundle=None):
        return subprocess.run([sys.executable,str(bundle or self.bundle),*args],capture_output=True,text=True)

    def test_bundle_excludes_weights_and_binaries_and_pins_source_bytes(self):
        record=self.build()
        self.assertEqual(len(record['sha256']),64)
        with zipfile.ZipFile(self.bundle) as z:
            manifest=json.loads(z.read('delivery_manifest.json'))
            names=set(z.namelist())
        self.assertIn('external_baselines/dfagcn/pipeline.py',names)
        self.assertIn('third_party/liblinear/liblinear-v250-source.tar.gz',names)
        self.assertFalse(any(n.endswith(('.pth','.pt')) or '/bin/' in n for n in names))
        self.assertEqual(record['bundle_id'],manifest['bundle_id'])
        self.assertTrue(self.bundle.with_suffix('.pyz.manifest.json').exists())
        with self.assertRaises(FileExistsError):
            self.build()

    def test_real_zipapp_inspect_delegates_and_reuse_checks_immutable_files(self):
        self.build()
        first=self.invoke('inspect')
        self.assertEqual(first.returncode,0,first.stderr)
        result=json.loads(first.stdout)
        self.assertEqual(result['args'],['inspect'])
        self.assertEqual(result['cwd'],str(self.bundle.with_suffix('').resolve()))
        self.assertEqual(result['delivery_entry'],str(self.bundle.resolve()))
        second=self.invoke('inspect')
        self.assertEqual(second.returncode,0,second.stderr)
        target=self.bundle.with_suffix('')/'external_baselines/dfagcn/pipeline.py'
        target.write_text('changed')
        altered=self.invoke('inspect')
        self.assertNotEqual(altered.returncode,0)
        self.assertIn('changed',altered.stderr)
        self.assertEqual(target.read_text(),'changed')

    def test_custom_runner_member_is_dispatched_from_manifest(self):
        self.assertIn('runner_member',inspect.signature(api().build_bundle).parameters,
                      'custom entrypoint support is missing')
        custom='tools/probe_dfagcn_gpu.py'
        self.build(runner_member=custom)
        with zipfile.ZipFile(self.bundle) as z:
            manifest=json.loads(z.read('delivery_manifest.json'))
            self.assertEqual(manifest['runner'],custom)
            self.assertIn(custom,z.namelist())
            self.assertNotIn('tools/run_dfagcn_nwpu_smoke.py',z.namelist())
        run=self.invoke('inspect','--probe-option')
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(json.loads(run.stdout)['args'],['inspect','--probe-option'])
        self.assertTrue((self.bundle.with_suffix('')/custom).is_file())

    def test_custom_runner_status_is_forwarded_without_starting_a_run(self):
        self.runner.write_text(
            'import json,sys\n'
            'from pathlib import Path\n'
            'if sys.argv[1:]==["status"]:\n'
            ' print(json.dumps({"state":"not_started"}))\n'
            'elif sys.argv[1:]==["run"]:\n'
            ' Path("run_started").write_text("unexpected")\n'
            'else:\n'
            ' raise SystemExit(2)\n')
        self.build(runner_member='tools/run_dfagcn_nwpu_formal.py')
        status=self.invoke('status')
        self.assertEqual(status.returncode,0,status.stderr)
        self.assertEqual(json.loads(status.stdout),{'state':'not_started'})
        self.assertFalse((self.bundle.with_suffix('')/'run_started').exists())

    def test_invalid_runner_member_is_rejected_before_output_creation(self):
        self.assertIn('runner_member',inspect.signature(api().build_bundle).parameters,
                      'custom entrypoint validation is missing')
        for member in ('../escape.py','/absolute.py','tools//probe.py','tools/../probe.py',
                       'tools\\probe.py','.', '__main__.py','delivery_manifest.json',
                       'external_baselines/dfagcn/pipeline.py'):
            with self.subTest(member=member),self.assertRaises(ValueError):
                self.build(runner_member=member)
            self.assertFalse(self.bundle.exists())
            self.assertFalse(self.bundle.with_suffix('.pyz.manifest.json').exists())

    def test_extra_helpers_are_importable_from_real_custom_entrypoint(self):
        self.assertIn('extra_members',inspect.signature(api().build_bundle).parameters,
                      'explicit helper delivery is missing')
        self.runner.write_text(
            'import json,sys\n'
            'from tools import probe_dfagcn_gpu,run_dfagcn_nwpu_smoke\n'
            'print(json.dumps({"probe_root":str(probe_dfagcn_gpu.ROOT),'
            '"smoke_root":str(run_dfagcn_nwpu_smoke.ROOT),'
            '"torch_imported":"torch" in sys.modules}))\n')
        helpers=('tools/probe_dfagcn_gpu.py','tools/run_dfagcn_nwpu_smoke.py')
        record=self.build(runner_member='tools/run_dfagcn_nwpu_formal.py',extra_members=helpers)
        repository=Path(__file__).resolve().parents[1]
        with zipfile.ZipFile(self.bundle) as archive:
            for name in helpers:
                self.assertEqual(archive.read(name),(repository/name).read_bytes())
                self.assertIn(name,record['manifest']['files'])
        run=self.invoke('inspect')
        self.assertEqual(run.returncode,0,run.stderr)
        expected_root=str(self.bundle.with_suffix('').resolve())
        self.assertEqual(json.loads(run.stdout),{'probe_root':expected_root,
                         'smoke_root':expected_root,'torch_imported':False})
        helper=self.bundle.with_suffix('')/helpers[0]
        helper.write_text('changed helper')
        changed=self.invoke('inspect')
        self.assertNotEqual(changed.returncode,0)
        self.assertIn('changed',changed.stderr)
        self.assertIn(helpers[0],changed.stderr)
        self.assertEqual(helper.read_text(),'changed helper')

    def test_extra_member_paths_duplicates_and_payload_overrides_are_rejected(self):
        self.assertIn('extra_members',inspect.signature(api().build_bundle).parameters,
                      'explicit helper validation is missing')
        invalid=[('../escape.py',),('/absolute.py',),('tools//helper.py',),
                 ('tools/../helper.py',),('tools\\helper.py',),('',),('.',),
                 ('C:/escape.py',),('tools/helper\x00.py',),
                 ('tools/probe_dfagcn_gpu.py','tools/probe_dfagcn_gpu.py'),
                 ('__main__.py',),('delivery_manifest.json',),('.delivery_receipt.json',),
                 ('external_baselines/dfagcn/pipeline.py',),('experiment_identity.py',),
                 ('third_party/liblinear/COPYRIGHT',),('bundled_weights_provenance.json',)]
        for members in invalid:
            with self.subTest(members=members),self.assertRaises(ValueError):
                self.build(extra_members=members)
            self.assertFalse(self.bundle.exists())
            self.assertFalse(self.bundle.with_suffix('.pyz.manifest.json').exists())
        for entry in ('tools/run_dfagcn_nwpu_smoke.py','tools/run_dfagcn_nwpu_formal.py'):
            with self.subTest(entry=entry),self.assertRaises(ValueError):
                self.build(runner_member=entry,extra_members=(entry,))
            self.assertFalse(self.bundle.exists())
        with self.assertRaises(ValueError):
            self.build(extra_members='tools/probe_dfagcn_gpu.py')
        self.assertFalse(self.bundle.exists())

    def test_extra_member_directories_missing_files_and_symlinks_are_rejected(self):
        self.assertIn('extra_members',inspect.signature(api().build_bundle).parameters,
                      'explicit helper source validation is missing')
        original=Path(api().__file__).resolve().parents[1]
        repository=self.root/'repository'
        for folder in ('external_baselines','scene_classification','configs',
                       'results/evidence/dfagcn_pipeline_20261029/liblinear'):
            shutil.copytree(original/folder,repository/folder)
        for name in ('experiment_identity.py','grouping_identity.py',
                     'tools/package_dfagcn_nwpu_smoke.py'):
            target=repository/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original/name,target)
        (repository/'linked_file.py').symlink_to(self.runner)
        (repository/'linked_directory').symlink_to(self.root,target_is_directory=True)
        with patch.object(api(),'__file__',str(repository/'tools/package_dfagcn_nwpu_smoke.py')):
            for member in ('tools','missing.py','linked_file.py','linked_directory/runner.py'):
                with self.subTest(member=member),self.assertRaises(ValueError):
                    self.build(extra_members=(member,))
                self.assertFalse(self.bundle.exists())
                self.assertFalse(self.bundle.with_suffix('.pyz.manifest.json').exists())

    def test_cli_extra_members_are_appended_to_a_custom_bundle(self):
        builder=Path(api().__file__).resolve()
        run=subprocess.run([sys.executable,str(builder),'--output',str(self.bundle),
                            '--runner',str(self.runner),'--runner-member','tools/formal.py',
                            '--extra-member','tools/probe_dfagcn_gpu.py',
                            '--extra-member','tools/run_dfagcn_nwpu_smoke.py'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        record=json.loads(run.stdout)
        self.assertIn('tools/probe_dfagcn_gpu.py',record['manifest']['files'])
        self.assertIn('tools/run_dfagcn_nwpu_smoke.py',record['manifest']['files'])
        launched=self.invoke('inspect')
        self.assertEqual(launched.returncode,0,launched.stderr)

    def test_current_reader_accepts_immutable_legacy_bundle_without_runner_field(self):
        original=Path(__file__).resolve().parents[1]/'output/dfagcn_nwpu_smoke_20261029.pyz'
        with zipfile.ZipFile(original) as z:
            before=z.read('delivery_manifest.json')
        legacy=self.root/'legacy.pyz'
        legacy.write_bytes(original.read_bytes())
        workspace,manifest=api().install_bundle(legacy)
        self.assertNotIn('runner',manifest)
        self.assertTrue((workspace/'tools/run_dfagcn_nwpu_smoke.py').is_file())
        with zipfile.ZipFile(original) as z:
            self.assertEqual(z.read('delivery_manifest.json'),before)

    def test_preexisting_or_partial_workspace_is_never_overwritten(self):
        self.build()
        workspace=self.bundle.with_suffix('')
        workspace.mkdir()
        original=workspace/'unrelated.txt'
        original.write_text('preserve')
        run=self.invoke('inspect')
        self.assertNotEqual(run.returncode,0)
        self.assertEqual(original.read_text(),'preserve')
        self.assertFalse((workspace/'tools').exists())

    def test_path_traversal_and_duplicate_archive_members_are_rejected(self):
        self.build()
        for kind in ('traversal','duplicate'):
            with self.subTest(kind=kind):
                bad=self.root/(kind+'.pyz')
                with zipfile.ZipFile(self.bundle) as source, zipfile.ZipFile(bad,'w') as z:
                    for item in source.infolist():
                        z.writestr(item,source.read(item.filename))
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',UserWarning)
                        z.writestr('../escape.txt' if kind=='traversal' else 'tools/run_dfagcn_nwpu_smoke.py',b'bad')
                run=self.invoke('inspect',bundle=bad)
                self.assertNotEqual(run.returncode,0)
                self.assertFalse(bad.with_suffix('').exists())
                self.assertFalse((self.root/'escape.txt').exists())

    def test_optional_weights_metadata_is_validated_and_shipped(self):
        meta=self.root/'weights.json'
        meta.write_text(json.dumps({'filename':'vgg16-397923af.pth','sha256':'397923af'+'a'*56}))
        self.build(weights_metadata_path=meta)
        with zipfile.ZipFile(self.bundle) as z:
            self.assertEqual(json.loads(z.read('bundled_weights_provenance.json'))['sha256'],'397923af'+'a'*56)

    def test_invalid_weights_metadata_fails_before_artifact_creation(self):
        meta=self.root/'bad_weights.json'
        meta.write_text(json.dumps({'sha256':'397923af'}))
        with self.assertRaises(ValueError):
            self.build(weights_metadata_path=meta)
        self.assertFalse(self.bundle.exists())

    def test_real_official_source_build_produces_verified_local_binaries(self):
        self.build()
        built=self.invoke('build-linear')
        self.assertEqual(built.returncode,0,built.stderr)
        record=json.loads(built.stdout)
        self.assertEqual(record['operation'],'build-linear')
        self.assertIn('platform',record,'compiled binary platform identity is missing')
        self.assertEqual(record['platform'],{'system':platform.system(),'machine':platform.machine()})
        self.assertTrue(record.get('compiler_version'))
        workspace=self.bundle.with_suffix('')
        for name in ('train','predict'):
            self.assertTrue(os.access(workspace/'liblinear/bin'/name,os.X_OK))
            self.assertEqual(len(record['binaries'][name]['sha256']),64)
        repeated=self.invoke('build-linear')
        self.assertEqual(repeated.returncode,0,repeated.stderr)
        self.assertTrue(json.loads(repeated.stdout)['reused'])
        build_record=workspace/'liblinear/build.json'
        saved=build_record.read_bytes()
        foreign=json.loads(saved)
        foreign['platform']={'system':'another-os','machine':'another-arch'}
        build_record.write_text(json.dumps(foreign))
        wrong_platform=self.invoke('build-linear')
        self.assertNotEqual(wrong_platform.returncode,0,'foreign-platform binaries must not be reused')
        self.assertIn('platform',wrong_platform.stderr)
        build_record.write_bytes(saved)
        target=workspace/'liblinear/bin/train'
        with target.open('ab') as stream:
            stream.write(b'changed')
        changed=self.invoke('build-linear')
        self.assertNotEqual(changed.returncode,0)
        self.assertIn('changed',changed.stderr)


if __name__=='__main__':
    unittest.main()
