"""Rollback publication and durable formal-started reporting; no ML or GPU."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from external_baselines.hgtnet import aid_formal_entry as e
from tests.test_hgtnet_aid_formal_status import fixture, write


class ResumeEntryTests(unittest.TestCase):
    def test_archive_moves_only_receipts_beyond_committed_progress_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();attempt=root/'attempt';attempt.mkdir()
            committed=b'{"optimizer_steps": 1, "phase": "calibration"}\n'
            discarded=b'{"optimizer_steps":25001,"post_update_checks_passed":false}\n'
            (root/'first_calibration_step.json').write_bytes(committed)
            (root/'first_step.json').write_bytes(committed)
            (root/'first_refit_step.json').write_bytes(discarded)
            e._archive_discarded_first_steps(root,attempt,25000)
            self.assertEqual((root/'first_calibration_step.json').read_bytes(),committed)
            self.assertEqual((root/'first_step.json').read_bytes(),committed)
            self.assertFalse((root/'first_refit_step.json').exists())
            self.assertEqual((attempt/'discarded_first_refit_step.json').read_bytes(),discarded)
            # Repeated helper calls cannot overwrite a previously discarded proof.
            (root/'first_refit_step.json').write_bytes(b'{"optimizer_steps":25002}\n')
            with self.assertRaises(FileExistsError):e._archive_discarded_first_steps(root,attempt,25000)
            self.assertEqual((attempt/'discarded_first_refit_step.json').read_bytes(),discarded)
            self.assertTrue((root/'first_refit_step.json').is_file())

    def test_resume_callback_replaces_discarded_refit_proof_and_preserves_update_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()/'run';root.mkdir();summary=fixture(root)
            accepted=json.loads((root/'first_refit_step.json').read_text())
            failed=dict(accepted,post_update_checks_passed=False)
            old_raw=json.dumps(failed,separators=(',',':')).encode()+b'\n'
            (root/'first_refit_step.json').write_bytes(old_raw)
            write(root/'optimizer_progress.json',failed)
            summary.update(completed=False,optimizer_steps=25001)
            write(root/'summary.json',summary)
            prior=root/'attempts'/'0001';prior.mkdir(parents=True)
            (prior/'first_refit_step.json').write_bytes(old_raw)
            args=e.parser().parse_args(['resume','--output',str(root)])
            binding=json.loads((root/'binding.json').read_text())
            runtime=summary['runtime'];gpu=runtime['gpu']
            inputs=dict(counts=dict(e.PROTO.REAL_COUNTS))
            provenance=dict(git_base_commit='a'*40,source_state='uncommitted_source_snapshot')
            preflight=dict(ready=True,selected_gpu=gpu,build_provenance=provenance)
            proof={}
            def model(*args):
                callback=args[-2]
                callback(dict(stage='resumed',optimizer_steps=25000,training_started=True,formal_training_started=True))
                self.assertFalse((root/'first_refit_step.json').exists())
                callback(accepted)
                callback(dict(stage='training_epoch_complete',phase='refit',epoch=2,optimizer_steps=25626,training_started=True,formal_training_started=True,optimizer_step_completed=True,optimizer_step_succeeded=True,post_update_checks_passed=True))
                callback(dict(stage='before_test',optimizer_steps=25626))
                return copy.deepcopy(summary)|dict(completed=True,optimizer_steps=25626)
            with patch.object(e,'inspect_resources',return_value=preflight),patch.object(e.DATA,'prepare_inputs',return_value=inputs),\
                 patch.object(e,'_binding',return_value=binding),patch.object(e,'_require_owned',return_value={}),\
                 patch.object(e.RES,'inventory_gpu',return_value={}),patch.object(e,'select_gpu',return_value=gpu),\
                 patch.object(e,'_capacity_proof',return_value=proof),patch.object(e.RES,'_capacity_path',return_value=dict(free_bytes=10*2**30)),\
                 patch.object(e,'_host_memory',return_value={}),patch.object(e,'_assert_unchanged',return_value=summary['completion_checks']),\
                 patch.object(e,'_runtime_and_torch',return_value=(object(),runtime)),patch.object(e,'_run_model',side_effect=model),\
                 contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(e.run_formal_entry(args,resume=True),0)
            attempt=root/'attempts'/'0002'
            self.assertEqual((attempt/'discarded_first_refit_step.json').read_bytes(),old_raw)
            self.assertEqual((prior/'first_refit_step.json').read_bytes(),old_raw)
            self.assertFalse(json.loads((attempt/'previous_optimizer_progress.json').read_text())['post_update_checks_passed'])
            self.assertTrue(json.loads((root/'first_refit_step.json').read_text())['post_update_checks_passed'])
            self.assertEqual(json.loads((root/'summary.json').read_text())['discarded_partial_optimizer_steps'],1)
            self.assertEqual(json.loads((root/'summary.json').read_text())['build_provenance'],provenance)
            self.assertTrue(e.status(root)['completed'])

    def test_partial_real_aid_status_uses_durable_update_despite_initial_false_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();summary=fixture(root)
            write(root/'summary.json',dict(schema=e.SCHEMA,input_scope='real_aid',optimizer_steps=0,
                training_started=False,formal_training_started=False,source_identity=e._source_identity()))
            write(root/'optimizer_progress.json',dict(stage='first_optimizer_step',optimizer_steps=1,
                optimizer_step_succeeded=True,training_started=True,formal_training_started=True))
            (root/'progress.json').unlink();(root/'first_step.json').unlink(missing_ok=True)
            (root/'run.exit').write_text('running\n')
            status=e.status(root)
            self.assertEqual(status['optimizer_steps'],1)
            self.assertTrue(status['training_started'])
            self.assertTrue(status['formal_training_started'])
            self.assertFalse(status['completed'])
            binding=json.loads((root/'binding.json').read_text())
            binding['input_scope']='cpu_fixture';binding['recipe']['max_epochs']=1
            binding['recipe_sha256']=e.PROTO.recipe_sha256(binding['recipe'])
            write(root/'binding.json',binding)
            self.assertEqual(e.PROTO.validate_binding(binding)['input_scope'],'cpu_fixture')
            self.assertFalse(e.status(root)['formal_training_started'])

    def test_build_provenance_reads_only_delivery_metadata_and_bare_source_is_none(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            with patch.object(e,'ROOT',root):
                self.assertIsNone(e._build_provenance())
                provenance=dict(git_base_commit='a'*40,source_state='uncommitted_source_snapshot')
                write(root/'delivery_manifest.json',dict(build_provenance=provenance))
                self.assertEqual(e._build_provenance(),provenance)

if __name__=='__main__':unittest.main()
