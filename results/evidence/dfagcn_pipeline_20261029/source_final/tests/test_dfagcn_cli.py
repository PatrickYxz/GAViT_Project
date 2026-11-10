"""The executable entrypoint exposes the actual staged workflow."""
import importlib
import io
import unittest
from contextlib import redirect_stdout


class CLITests(unittest.TestCase):
    def test_help_exposes_all_workflow_commands(self):
        try:
            cli=importlib.import_module('external_baselines.dfagcn.cli')
        except ModuleNotFoundError as exc:
            raise AssertionError('DFAGCN CLI is not implemented') from exc
        output=io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            cli.main(['--help'])
        self.assertEqual(caught.exception.code,0)
        for name in ('cache','calibrate','refit','test','smoke'):
            self.assertIn(name,output.getvalue())


if __name__ == '__main__':
    unittest.main()
