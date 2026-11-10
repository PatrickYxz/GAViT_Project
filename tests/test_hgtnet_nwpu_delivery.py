"""Exercise the source-only HGTNet delivery with real zipapps and files."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile


def api():
    try:
        return importlib.import_module('tools.package_hgtnet_nwpu_smoke')
    except ModuleNotFoundError as exc:
        raise AssertionError('HGTNet standalone delivery is not implemented') from exc


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.repository = self.root / 'repository'
        sources = {
            'external_baselines/__init__.py': '',
            'external_baselines/hgtnet/__init__.py': '',
            'external_baselines/hgtnet/model.py': 'raise RuntimeError("ML module imported")\n',
            'external_baselines/hgtnet/nwpu_data.py': 'SENTINEL = 42\n',
            'configs/hgtnet_independent_v0.json': '{"independent": true}\n',
            'tools/run_dfagcn_nwpu_smoke.py': 'HELPER = 42\n',
            'tools/run_hgtnet_nwpu_smoke.py': (
                'import json,os,sys\nfrom pathlib import Path\n'
                'ROOT=Path(__file__).resolve().parents[1]\n'
                'from external_baselines.hgtnet import nwpu_data\n'
                'from tools import run_dfagcn_nwpu_smoke\n'
                'if sys.argv[1:2]==["run"]: (ROOT/"explicit_run").write_text("started")\n'
                'print(json.dumps({"args":sys.argv[1:],"root":str(ROOT),'
                '"cwd":os.getcwd(),"entry":os.environ.get("HGTNET_DELIVERY_ENTRY"),'
                '"state":"not_started","helper":run_dfagcn_nwpu_smoke.HELPER,'
                '"torch_imported":"torch" in sys.modules}))\n'),
            'external_baselines/dfagcn/unwanted.py': 'raise RuntimeError\n',
            'external_baselines/hgtnet/README.md': 'not bundled\n',
            'external_baselines/hgtnet/test_model.txt': 'not bundled\n',
        }
        for name, text in sources.items():
            path = self.repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.bundle = self.root / 'smoke.pyz'

    def build(self):
        with patch.object(api(), 'ROOT', self.repository):
            return api().build_bundle(self.bundle)

    def invoke(self, *args, bundle=None):
        return subprocess.run([sys.executable, str(bundle or self.bundle), *args],
                              capture_output=True, text=True)

    def rewrite(self, filename, transform):
        destination = self.root / filename
        with zipfile.ZipFile(self.bundle) as source:
            entries = [(item, source.read(item.filename)) for item in source.infolist()]
        with zipfile.ZipFile(destination, 'w') as output:
            for item, data in transform(entries):
                output.writestr(item, data)
        return destination

    def test_build_pins_exact_scoped_sources_and_full_candidate_metadata(self):
        record = self.build()
        self.assertEqual(record['sha256'], hashlib.sha256(self.bundle.read_bytes()).hexdigest())
        self.assertEqual(record['bytes'], self.bundle.stat().st_size)
        expected = {
            'external_baselines/__init__.py', 'external_baselines/hgtnet/__init__.py',
            'external_baselines/hgtnet/model.py', 'external_baselines/hgtnet/nwpu_data.py',
            'configs/hgtnet_independent_v0.json', 'tools/run_dfagcn_nwpu_smoke.py',
            'tools/run_hgtnet_nwpu_smoke.py', 'bundled_weights_provenance.json',
        }
        with zipfile.ZipFile(self.bundle) as archive:
            self.assertEqual(set(archive.namelist()), expected | {'__main__.py', 'delivery_manifest.json'})
            manifest = json.loads(archive.read('delivery_manifest.json'))
            self.assertEqual(set(manifest['files']), expected)
            for name in expected - {'bundled_weights_provenance.json'}:
                raw = (self.repository / name).read_bytes()
                self.assertEqual(archive.read(name), raw)
                self.assertEqual(manifest['files'][name], {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
            provenance = json.loads(archive.read('bundled_weights_provenance.json'))
            self.assertEqual(provenance['sha256'], '68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893')
            self.assertEqual(provenance['bytes'], 351654207)
            self.assertEqual(provenance['filename'], 'swin_b-68c6b09e.pth')
            self.assertEqual(provenance['source_url'], 'https://download.pytorch.org/models/swin_b-68c6b09e.pth')
        self.assertEqual(record['manifest'], manifest)
        self.assertTrue(self.bundle.with_suffix('.pyz.manifest.json').is_file())
        with self.assertRaises(FileExistsError):
            self.build()

    def test_real_zipapp_forwards_inspect_status_and_run_with_stdlib_only(self):
        self.build()
        for action in ('inspect', 'status'):
            with self.subTest(action=action):
                result = self.invoke(action, '--output', 'path with spaces')
                self.assertEqual(result.returncode, 0, result.stderr)
                value = json.loads(result.stdout)
                workspace = str(self.bundle.with_suffix(''))
                self.assertEqual(value['args'], [action, '--output', 'path with spaces'])
                self.assertEqual(value['root'], workspace)
                self.assertEqual(value['cwd'], workspace)
                self.assertEqual(value['entry'], str(self.bundle))
                self.assertFalse(value['torch_imported'])
                self.assertFalse((self.bundle.with_suffix('')/'explicit_run').exists())
        run = self.invoke('run', '--gpu-uuid', 'GPU-test')
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)['args'], ['run', '--gpu-uuid', 'GPU-test'])
        self.assertTrue((self.bundle.with_suffix('')/'explicit_run').is_file())

    def test_unrecognized_actions_do_not_install_or_start_runner(self):
        self.build()
        for arguments in ((), ('build-linear',), ('install-deps',), ('train',)):
            with self.subTest(arguments=arguments):
                result = self.invoke(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.bundle.with_suffix('').exists())

    def test_foreign_or_partial_workspace_is_preserved(self):
        self.build()
        workspace = self.bundle.with_suffix('')
        workspace.mkdir()
        sentinel = workspace / 'existing.txt'
        sentinel.write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'foreign|incomplete'):
            api().install_bundle(self.bundle)
        self.assertEqual(sentinel.read_text(), 'preserve')
        self.assertFalse((workspace/'tools').exists())

    def test_reuse_refuses_source_receipt_and_manifest_changes(self):
        self.build()
        workspace, _ = api().install_bundle(self.bundle)
        targets = ('external_baselines/hgtnet/model.py', '.delivery_receipt.json', 'delivery_manifest.json')
        for target in targets:
            with self.subTest(target=target):
                path = workspace / target
                original = path.read_bytes()
                path.write_bytes(b'{}')
                with self.assertRaises((ValueError, KeyError)):
                    api().install_bundle(self.bundle)
                self.assertEqual(path.read_bytes(), b'{}')
                path.write_bytes(original)

    def test_workspace_and_tracked_file_symlinks_are_rejected(self):
        self.build()
        workspace = self.bundle.with_suffix('')
        workspace.symlink_to(self.repository, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symbolic'):
            api().install_bundle(self.bundle)
        workspace.unlink()
        api().install_bundle(self.bundle)
        for target in ('external_baselines/hgtnet/model.py', '.delivery_receipt.json', 'delivery_manifest.json'):
            with self.subTest(target=target):
                path = workspace / target
                original = path.read_bytes()
                outside = self.root / ('copy-' + path.name)
                outside.write_bytes(original)
                path.unlink()
                path.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, 'symbolic'):
                    api().install_bundle(self.bundle)
                path.unlink()
                path.write_bytes(original)

    def test_archive_tampering_duplicate_traversal_collision_and_symlink_are_rejected(self):
        self.build()
        def append(name, data=b'bad'):
            return lambda entries: entries + [(name, data)]
        symlink = zipfile.ZipInfo('external_baselines/hgtnet/link.py')
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        changes = {
            'traversal':append('../escape.py'),
            'duplicate':append('tools/run_hgtnet_nwpu_smoke.py'),
            'collision':append('tools'),
            'symlink':append(symlink),
            'source':lambda entries:[(i, b'changed' if i.filename.endswith('/model.py') else b) for i,b in entries],
            'bootstrap':lambda entries:[(i, b'# changed' if i.filename=='__main__.py' else b) for i,b in entries],
            'manifest':lambda entries:[(i, b'{}' if i.filename=='delivery_manifest.json' else b) for i,b in entries],
        }
        for kind, change in changes.items():
            with self.subTest(kind=kind), warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                invalid = self.rewrite(kind+'.pyz', change)
                with self.assertRaises((ValueError, KeyError)):
                    api().install_bundle(invalid)
                self.assertFalse(invalid.with_suffix('').exists())
        self.assertFalse((self.root/'escape.py').exists())

    def test_build_rejects_linked_source_and_sidecar_collision_without_overwrite(self):
        path = self.repository/'external_baselines/hgtnet/model.py'
        path.unlink()
        path.symlink_to(self.repository/'external_baselines/hgtnet/nwpu_data.py')
        with self.assertRaisesRegex(ValueError, 'symbolic'):
            self.build()
        self.assertFalse(self.bundle.exists())
        path.unlink()
        path.write_text('pass\n')
        sidecar = self.bundle.with_suffix('.pyz.manifest.json')
        sidecar.write_text('preserve')
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertFalse(self.bundle.exists())
        self.assertEqual(sidecar.read_text(), 'preserve')

    def test_reuse_refuses_added_importable_sources_in_workspace(self):
        self.build()
        workspace, _ = api().install_bundle(self.bundle)
        for name in ('tools/__init__.py', 'external_baselines/hgtnet/injected.py',
                     'configs/extra.json', 'torch.py', 'injected/__init__.py'):
            with self.subTest(name=name):
                target = workspace / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('raise RuntimeError("injected")\n')
                with self.assertRaisesRegex(ValueError, 'unexpected|immutable'):
                    api().install_bundle(self.bundle)
                self.assertTrue(target.is_file())
                target.unlink()

    def test_cli_builds_a_new_offline_bundle_then_status_runs_without_ml_packages(self):
        builder = self.repository / 'tools/package_hgtnet_nwpu_smoke.py'
        shutil.copyfile(api().__file__, builder)
        built = subprocess.run([sys.executable, str(builder), 'build', '--output', str(self.bundle)],
                               capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stderr)
        self.assertEqual(json.loads(built.stdout)['path'], str(self.bundle))
        guard = self.root / 'guard'
        guard.mkdir()
        (guard/'sitecustomize.py').write_text(
            'import sys\nclass Guard:\n'
            ' def find_spec(self,fullname,*args):\n'
            '  if fullname.split(".")[0] in {"torch","torchvision","numpy","PIL"}:\n'
            '   raise RuntimeError("forbidden ML import: "+fullname)\n'
            'sys.meta_path.insert(0,Guard())\n')
        environment = dict(os.environ, PYTHONPATH=str(guard))
        result = subprocess.run([sys.executable, str(self.bundle), 'status'],
                                capture_output=True, text=True, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['state'], 'not_started')
        self.assertFalse((self.bundle.with_suffix('')/'explicit_run').exists())


if __name__ == '__main__':
    unittest.main()
