"""Actual archive install and stdlib subprocess checks for AID3060 delivery."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile


def api():
    try: return importlib.import_module('tools.package_hgtnet_aid_batch16_probe')
    except ModuleNotFoundError as exc:
        raise AssertionError('independent AID delivery has not been implemented') from exc


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()
        self.bundle=self.root/'aid smoke.pyz'

    def rewrite(self,name,transform):
        path=self.root/name
        with zipfile.ZipFile(self.bundle) as archive:
            members=[(item,archive.read(item.filename)) for item in archive.infolist()]
        with warnings.catch_warnings(),zipfile.ZipFile(path,'w') as archive:
            warnings.simplefilter('ignore',UserWarning)
            for item,raw in transform(members):archive.writestr(item,raw)
        return path

    def test_bundle_is_exact_source_whitelist_with_separate3060_identity(self):
        record=api().build_bundle(self.bundle)
        self.assertEqual(record['sha256'],hashlib.sha256(self.bundle.read_bytes()).hexdigest())
        self.assertEqual(record['bytes'],self.bundle.stat().st_size)
        self.assertEqual(record['manifest']['schema'],'hgtnet-aid-batch16-delivery-v1')
        self.assertEqual(record['manifest']['target_gpu'],'3060')
        self.assertEqual(len(api().SOURCE_FILES),18)
        with zipfile.ZipFile(self.bundle) as archive:
            self.assertEqual(set(archive.namelist()),set(api().SOURCE_FILES)|
                             {'__main__.py',api().MANIFEST,'bundled_weights_provenance.json'})
            self.assertNotIn('configs/hgtnet_independent_v0.json',archive.namelist())
            self.assertEqual(len(archive.namelist()),21)
            self.assertNotIn('external_baselines/hgtnet/aid_engine.py',archive.namelist())
            for name in api().SOURCE_FILES:
                self.assertEqual(archive.read(name),(api().ROOT/name).read_bytes())
        with self.assertRaises(FileExistsError):api().build_bundle(self.bundle)

    def test_real_pyz_inspect_status_use_real_install_directory_without_ML(self):
        api().build_bundle(self.bundle)
        guard=self.root/'guard';guard.mkdir()
        (guard/'sitecustomize.py').write_text(
            'import builtins\n_original=builtins.__import__\n'
            'def guarded(name,*args,**kwargs):\n'
            ' if name.split(".")[0] in {"torch","torchvision","numpy","PIL"}: '
            'raise RuntimeError("ML imported: "+name)\n'
            ' return _original(name,*args,**kwargs)\n'
            'builtins.__import__=guarded\n')
        env=dict(os.environ,PYTHONPATH=str(guard),PYTHONDONTWRITEBYTECODE='1')
        for action in ('inspect','status'):
            result=subprocess.run([sys.executable,str(self.bundle),action],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(result.stdout)
            self.assertFalse(report.get('ready',report.get('completed',False)))
            self.assertEqual(report['output'],str(self.bundle.with_suffix('')/'runs/batch16_seed42_01'))
        installed,manifest=api().install_bundle(self.bundle)
        self.assertEqual(installed,self.bundle.with_suffix(''))
        self.assertEqual(manifest['runner'],'external_baselines/hgtnet/aid_batch_probe.py')
        self.assertFalse((installed/'runs').exists())

    def test_unknown_actions_never_install(self):
        api().build_bundle(self.bundle)
        for action in ([],['resume'],['train'],['install-deps'],['build']):
            result=subprocess.run([sys.executable,str(self.bundle),*action],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(self.bundle.with_suffix('').exists())

    def test_install_rejects_foreign_destination_and_altered_or_extra_source(self):
        api().build_bundle(self.bundle)
        workspace=self.bundle.with_suffix('');workspace.mkdir()
        (workspace/'keep').write_text('preserve')
        with self.assertRaisesRegex(ValueError,'foreign|incomplete'):api().install_bundle(self.bundle)
        (workspace/'keep').unlink();workspace.rmdir()
        api().install_bundle(self.bundle)
        for name in ['external_baselines/hgtnet/model.py',api().RECEIPT,api().MANIFEST]:
            target=workspace/name;original=target.read_bytes();target.write_bytes(b'{}')
            with self.assertRaises(ValueError):api().install_bundle(self.bundle)
            self.assertEqual(target.read_bytes(),b'{}');target.write_bytes(original)
        target=workspace/'external_baselines/hgtnet/injected.py';target.write_text('pass')
        with self.assertRaisesRegex(ValueError,'unexpected'):api().install_bundle(self.bundle)

    def test_install_rejects_member_tamper_duplicate_traversal_collision_and_symlink(self):
        api().build_bundle(self.bundle)
        symlink=zipfile.ZipInfo('external_baselines/hgtnet/link.py')
        symlink.external_attr=(stat.S_IFLNK|0o777)<<16
        variants=[lambda rows:rows+[(rows[0][0],rows[0][1])],
                  lambda rows:rows+[('../escape.py',b'bad')],
                  lambda rows:rows+[('external_baselines',b'bad')],
                  lambda rows:rows+[(symlink,b'target')],
                  lambda rows:[(item,b'changed' if item.filename.endswith('/model.py') else raw)
                               for item,raw in rows]]
        for i,transform in enumerate(variants):
            changed=self.rewrite('bad'+str(i)+'.pyz',transform)
            with self.subTest(i=i),self.assertRaises((ValueError,KeyError)):
                api().install_bundle(changed)
            self.assertFalse(changed.with_suffix('').exists())

    def test_symlink_install_or_member_are_refused(self):
        api().build_bundle(self.bundle)
        workspace=self.bundle.with_suffix('');other=self.root/'other';other.mkdir()
        workspace.symlink_to(other,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'symbolic'):api().install_bundle(self.bundle)
        workspace.unlink();api().install_bundle(self.bundle)
        target=workspace/'external_baselines/hgtnet/model.py';original=target.read_bytes()
        copy=self.root/'copy';copy.write_bytes(original);target.unlink();target.symlink_to(copy)
        with self.assertRaisesRegex(ValueError,'symbolic'):api().install_bundle(self.bundle)


if __name__ == '__main__':unittest.main()
