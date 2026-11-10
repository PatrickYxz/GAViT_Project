"""Recovery must trust committed, hashed transactions rather than loose files."""
import json
from pathlib import Path
import tempfile
import unittest

from scene_classification import aid_campaign_state as state


class CampaignStateTests(unittest.TestCase):
    def test_existing_output_requires_explicit_resume_and_exact_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'job'
            state.establish_identity(out, {'seed': 42}, resume=False)
            with self.assertRaises(FileExistsError):
                state.establish_identity(out, {'seed': 42}, resume=False)
            with self.assertRaisesRegex(ValueError, 'identity'):
                state.establish_identity(out, {'seed': 43}, resume=True)
            state.establish_identity(out, {'seed': 42}, resume=True)

    def test_corrupted_committed_artifact_is_never_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            def write(where):
                (where / 'state.bin').write_bytes(b'epoch one')
            state.commit_transaction(out, 1, write)
            folder, receipt = state.verified_latest(out)
            self.assertEqual(receipt['epoch'], 1)
            (folder / 'state.bin').write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'hash'):
                state.verified_latest(out)

    def test_interrupted_write_retains_prior_full_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            state.commit_transaction(out, 0, lambda p: (p / 'state.bin').write_bytes(b'initial'))
            def broken(where):
                (where / 'state.bin').write_bytes(b'partial epoch')
                raise RuntimeError('process failed before commit')
            with self.assertRaises(RuntimeError):
                state.commit_transaction(out, 1, broken)
            folder, receipt = state.verified_latest(out)
            self.assertEqual(receipt['epoch'], 0)
            self.assertEqual((folder / 'state.bin').read_bytes(), b'initial')
            self.assertTrue(list((out / 'transactions').glob('.pending-*')))

    def test_receipt_path_cannot_escape_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            state.atomic_json(out / 'latest.json', {'transaction': '../other', 'epoch': 1,
                                                    'files_sha256': {}})
            with self.assertRaisesRegex(ValueError, 'transaction'):
                state.verified_latest(out)

    def test_only_latest_two_accepted_full_states_remain(self):
        from scene_classification.aid_campaign_engine import _retire_old_training_states
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            accepted = []
            for epoch in range(4):
                def write(folder):
                    (folder / 'train_state.pth').write_bytes(str(epoch).encode())
                    (folder / 'history.json').write_text(str(list(range(epoch + 1))))
                folder, _ = state.commit_transaction(out, epoch, write)
                accepted.append(folder)
                _retire_old_training_states(out, folder)
            self.assertEqual([p.exists() for p in (f / 'train_state.pth' for f in accepted)],
                             [False, False, True, True])
            self.assertTrue(all((p / 'history.json').is_file() for p in accepted))


if __name__ == '__main__':
    unittest.main()
