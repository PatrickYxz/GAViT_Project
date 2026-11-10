"""AID formal stdlib entry and safety contracts; CPU artifacts are receipts only."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def api():
    module=importlib.import_module('external_baselines.hgtnet.aid_formal_entry')
    for name in ('parser','status','_metrics','_completion_evidence_ok'):
        if not callable(getattr(module,name,None)):
            raise AssertionError('Missing formal entry behavior: '+name)
    return module

class EntryTests(unittest.TestCase):
    def test_explicit_actions_new_sources_and_no_scratch(self):
        e=api();args=e.parser().parse_args(['inspect'])
        self.assertEqual(len(e.SOURCE_FILES),22)
        self.assertEqual(args.output.name,'aid_seed42_01')
        self.assertFalse(hasattr(args,'scratch_root'))
        self.assertNotIn('external_baselines/hgtnet/aid_engine.py',e.SOURCE_FILES)
        self.assertFalse(e.status(Path('/tmp/nonexistent_aid_formal_fixture'))['completed'])
    def test_exclusive_lock_and_owned_output(self):
        e=api()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temp:
            output=Path(temp)/'run'
            with e._run_lock(output):
                with self.assertRaises((ValueError,FileExistsError)):
                    with e._run_lock(output):pass
            self.assertFalse(e.lock_path(output).exists());output.mkdir()
            with self.assertRaises(FileExistsError):
                with e._run_lock(output):pass
            with e._run_lock(output,resume=True):
                self.assertTrue(e.lock_path(output).exists())
    def test_capacity_is_sha_pinned_status_only_same_current_uuid(self):
        e=api();args=e.parser().parse_args(['inspect'])
        packet=dict(schema='hgtnet-aid-batch16-capacity-probe-v1',state='complete',completed=True,exit_code=0,
          source_identity_matches=True,completion_evidence_valid=True,physical_gpu_uuid='GPU-current',
          config_sha256=e.CONFIG_SHA,manifest_sha256=e.DATA.MANIFEST_SHA,weights_sha256=e.WEIGHTS_SHA,
          counts=dict(train=16,val=0,test=0),batch_size=16,unique_images=16,processed_images=64,
          optimizer_steps=4,warmup_steps=1,measured_steps=3)
        proc=subprocess.CompletedProcess([],0,json.dumps(packet),'')
        with patch.object(e,'_sha',return_value=e.CAPACITY_SHA),patch.object(e.subprocess,'run',return_value=proc) as run:
            self.assertEqual(e._capacity_proof(args,'GPU-current')['status'],packet)
            cmd=run.call_args.args[0];self.assertEqual(cmd,[sys.executable,str(args.capacity_bundle),'status','--output',str(args.capacity_output)])
            with self.assertRaises(ValueError):e._capacity_proof(args,'GPU-other')
        with patch.object(e,'_sha',return_value='a'*64),patch.object(e.subprocess,'run') as run:
            with self.assertRaises(ValueError):e._capacity_proof(args,'GPU-current')
            run.assert_not_called()
    def test_metrics_are_pure_python_and_correct(self):
        metric=api()._metrics([0,0,1],[0,1,1],['a','b'])
        self.assertEqual(metric['confusion_matrix'],[[1,1],[0,1]])
        self.assertAlmostEqual(metric['accuracy_percent'],200/3)
        self.assertAlmostEqual(metric['macro_f1_percent'],200/3)
    def test_skeletal_complete_is_rejected(self):
        e=api()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temp:
            p=Path(temp);(p/'summary.json').write_text(json.dumps(dict(completed=True,schema=e.SCHEMA)))
            (p/'run.exit').write_text('0\n');(p/'completion.marker').write_text(e.MARKER)
            self.assertFalse(e.status(p)['completed'])
    def test_importguard_status_and_inspect(self):
        script='import builtins\nf=builtins.__import__\ndef guard(n,*a,**k):\n if n.split(".")[0] in {"torch","torchvision","numpy","PIL"}: raise RuntimeError(n)\n return f(n,*a,**k)\nbuiltins.__import__=guard\nfrom external_baselines.hgtnet import aid_formal_entry as e\na=e.parser().parse_args(["inspect","--output","/tmp/no_aid_formal_fixture"])\nassert not e.inspect_resources(a)["ready"]\nassert not e.status(a.output)["completed"]\n'
        p=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
if __name__=='__main__':unittest.main()
