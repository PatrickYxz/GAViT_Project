"""Stdlib delivery, data staging, and status gates without a GPU or ML import."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    path = ROOT / 'tools' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class DeliveryTests(unittest.TestCase):
    def test_delivery_implementation_exists(self):
        self.assertTrue((ROOT / 'tools/run_aid_campaign.py').is_file(), 'campaign CLI missing')
        self.assertTrue((ROOT / 'tools/package_aid_campaign.py').is_file(), 'campaign packager missing')

    def test_frozen_jobs_refuse_unbudgeted_reruns(self):
        cli = load('run_aid_campaign')
        for variant, seed in [('soft', 46), ('swin', 42), ('fixed', 44)]:
            with self.assertRaisesRegex(ValueError, 'frozen'):
                cli.validate_job(variant, seed)
        for variant, seed in [('soft', 42), ('soft', 45), ('swin', 45), ('fixed', 45)]:
            cli.validate_job(variant, seed)

    def test_manifest_change_refused_and_inspect_imports_no_ml(self):
        cli = load('run_aid_campaign')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'manifest.json'; p.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'manifest SHA256'):
                cli.load_manifest(p)
        proc = subprocess.run([sys.executable, '-I', str(ROOT / 'tools/run_aid_campaign.py'), 'inspect'],
                              capture_output=True, text=True)
        report = json.loads(proc.stdout)
        self.assertFalse(report['import_tested'])
        self.assertFalse(report['training_started'])

    def test_stage_selected_bytes_and_reject_zip_traversal(self):
        cli = load('run_aid_campaign')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root / 'aid.zip'; raw = b'an image'
            records = [{'path': 'Airport/a.jpg', 'sha256': hashlib.sha256(raw).hexdigest()}]
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('AID/AID/Airport/a.jpg', raw)
            staged = cli.stage_images(archive, root / 'scratch', records,
                                      expected_archive_sha256=cli.sha256_file(archive))
            self.assertEqual((staged / 'Airport/a.jpg').read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                cli.stage_images(archive, root / 'scratch', records,
                                 expected_archive_sha256=cli.sha256_file(archive))
            bad = root / 'bad.zip'
            with zipfile.ZipFile(bad, 'w') as z:
                z.writestr('../escape', b'oops')
                z.writestr('AID/AID/Airport/a.jpg', raw)
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                cli.stage_images(bad, root / 'other', records,
                                 expected_archive_sha256=cli.sha256_file(bad))
            self.assertFalse((root / 'escape').exists())

    def test_image_hash_mismatch_refused_without_overwriting_existing_data(self):
        cli = load('run_aid_campaign')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'Airport').mkdir(); (root / 'Airport/a.jpg').write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'image SHA256'):
                cli.verify_images(root, [{'path':'Airport/a.jpg', 'sha256':'0'*64}])
            self.assertEqual((root / 'Airport/a.jpg').read_bytes(), b'wrong')

    def test_nvml_cuda_capacity_gap_is_tolerated_but_wrong_device_refused(self):
        cli = load('run_aid_campaign')
        nvml = {'uuid':'GPU-test', 'name':'NVIDIA RTX 4090', 'memory_total_mib':24576}
        runtime = {'uuid':'GPU-test', 'name':'NVIDIA RTX 4090', 'memory_total_bytes':int(23.55*2**30)}
        cli.verify_cuda_identity(nvml, runtime)
        runtime['uuid'] = 'GPU-other'
        with self.assertRaisesRegex(ValueError, 'UUID'):
            cli.verify_cuda_identity(nvml, runtime)

    def test_status_needs_exit_markers_summary_and_hashes_not_just_pid(self):
        cli = load('run_aid_campaign')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); (out / 'run.pid').write_text(str(__import__('os').getpid()))
            self.assertNotEqual(cli.status_report(out)['status'], 'complete')
            artifacts=['job_identity.json','selection.json','metrics.json','calibration/first_update.json',
                       'refit/first_update.json','calibration/completion.json','refit/completion.json']
            for name in artifacts:
                path=out/name; path.parent.mkdir(exist_ok=True);path.write_text('{}\n')
            (out / 'run.exit').write_text('exit_code=0\n')
            artifacts.append('run.exit')
            (out / 'completion.json').write_text(json.dumps({'status':'complete', 'identity':{},
                'summary':{'status':'complete'}, 'artifact_sha256':{name:cli.sha256_file(out/name) for name in artifacts}}))
            self.assertEqual(cli.status_report(out)['status'], 'incomplete')
            (out / 'COMPLETED').write_text(cli.sha256_file(out / 'completion.json')+'\n')
            self.assertEqual(cli.status_report(out)['status'], 'complete')
            (out / 'metrics.json').write_text('corrupted')
            self.assertEqual(cli.status_report(out)['status'], 'incomplete')

    def test_inspect_and_status_really_import_no_ml(self):
        code = """import contextlib,io,runpy,sys
sys.argv=[sys.argv[1],sys.argv[2]]
stream=io.StringIO()
with contextlib.redirect_stdout(stream):
 try: runpy.run_path(sys.argv[0],run_name="__main__")
 except SystemExit: pass
print(','.join(sorted(set(sys.modules)&{"torch","torchvision","timm","torch_geometric","numpy"})))
"""
        for command in ('inspect','status'):
            proc = subprocess.run([sys.executable,'-I','-c',code,str(ROOT / 'tools/run_aid_campaign.py'),command], capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertEqual(proc.stdout.strip(),'')

    def test_stage_refuses_persistent_work_and_archive_hash_mismatch(self):
        cli=load('run_aid_campaign')
        with self.assertRaisesRegex(ValueError,'instance-local'):
            cli.stage_images('/does/not/exist','/home/featurize/work/new_raw',[])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); archive=root / 'zip'; archive.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'archive SHA256'):
                cli.stage_images(archive,root / 'stage',[])
            self.assertFalse((root / 'stage').exists())

    def test_selected_gpu_busy_or_ambiguous_is_refused(self):
        cli=load('run_aid_campaign')
        gpu={'uuid':'GPU-abcd','memory_total_mib':24576,'memory_free_mib':24500}
        self.assertEqual(cli.select_gpu({'gpus':[gpu],'processes':[]},'GPU-abcd'),gpu)
        with self.assertRaisesRegex(ValueError,'compute process'):
            cli.select_gpu({'gpus':[gpu],'processes':[{'uuid':'GPU-abcd','pid':999999}]},'GPU-abcd')
        with self.assertRaisesRegex(ValueError,'uniquely'):
            cli.select_gpu({'gpus':[gpu,gpu],'processes':[]},'GPU-abcd')

    def test_pyz_inspect_safe_without_training_and_extra_source_refused(self):
        package=load('package_aid_campaign')
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp) / 'campaign.pyz'; package.build_bundle(output)
            proc=subprocess.run([sys.executable,'-I',str(output),'inspect'],capture_output=True,text=True)
            report=json.loads(proc.stdout)
            self.assertFalse(report['training_started']); self.assertFalse(report['import_tested'])
            self.assertEqual(proc.returncode,2)  # Local machine lacks target data/GPU.
            workspace=output.with_suffix('');(workspace / 'models/foreign.py').write_text('bad=1')
            with self.assertRaisesRegex(ValueError,'Unexpected immutable'):
                package.install_bundle(output)

    def test_package_exact_source_identity_and_workspace_tamper_refused(self):
        package = load('package_aid_campaign')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'delivery.pyz'
            record = package.build_bundle(output)
            self.assertEqual(record['manifest']['weights']['architecture'], 'swin_tiny_patch4_window7_224')
            self.assertEqual(record['manifest']['build_provenance']['source_state'], 'uncommitted_source_snapshot')
            self.assertIn('scene_classification/aid_campaign_engine.py', record['manifest']['files'])
            workspace, manifest = package.install_bundle(output)
            target = workspace / 'tools/run_aid_campaign.py'
            target.write_bytes(target.read_bytes()+b'\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                package.install_bundle(output)

if __name__ == '__main__': unittest.main()
