"""Torch-free wiring checks for the sparse-hybrid Featurize gate.

These checks do not replace the tensor-level graph and model tests. They catch
the cheaper failure mode where one train/test entry point cannot reconstruct
the architecture selected by another entry point.
"""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = (
    "train_bigearth.py",
    "test_bigearth.py",
    "train_gavit.py",
    "test_gavit.py",
)


def string_constants(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


class SparseHybridStaticContractTests(unittest.TestCase):
    def test_every_entry_point_accepts_sparse_hybrid_identity(self):
        missing = [
            filename
            for filename in ENTRY_POINTS
            if "sparse_hybrid" not in string_constants(ROOT / filename)
        ]
        self.assertEqual(missing, [])

    def test_model_dispatches_to_dedicated_builder(self):
        source = (ROOT / "models" / "gavit.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("build_sparse_hybrid_graph", calls)


if __name__ == "__main__":
    unittest.main()
