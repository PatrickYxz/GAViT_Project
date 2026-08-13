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
TRAIN_ENTRY_POINTS = ("train_bigearth.py", "train_gavit.py")


def edge_type_choices(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not any(
            isinstance(argument, ast.Constant) and argument.value == "--edge_type"
            for argument in node.args
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices" and isinstance(
                keyword.value, (ast.List, ast.Tuple)
            ):
                return {
                    element.value
                    for element in keyword.value.elts
                    if isinstance(element, ast.Constant)
                    and isinstance(element.value, str)
                }
    return set()


class SparseHybridStaticContractTests(unittest.TestCase):
    def test_every_entry_point_accepts_sparse_hybrid_identity(self):
        missing = [
            filename
            for filename in ENTRY_POINTS
            if "sparse_hybrid" not in edge_type_choices(ROOT / filename)
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

    def test_train_entry_points_record_explicit_graph_topology(self):
        missing = []
        for filename in TRAIN_ENTRY_POINTS:
            tree = ast.parse(
                (ROOT / filename).read_text(encoding="utf-8"),
                filename=filename,
            )
            calls = {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            if "graph_topology_identity" not in calls:
                missing.append(filename)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
