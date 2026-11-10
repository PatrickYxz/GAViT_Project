"""Independent full-branch recipe tests; no original Table I parity claim."""

import copy
import importlib
import io
import json
from pathlib import Path
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def api(name):
    try:
        return importlib.import_module("external_baselines.dfagcn." + name)
    except ModuleNotFoundError as exc:
        raise AssertionError("DFAGCN independent recipe/model interface is missing") from exc


def recipe():
    return dict(schema="dfagcn-independent-recipe-v1", implementation="dfagcn_independent_v1",
                pooling_hw=[3, 4], knn_k=2, metric="euclidean", gcn_dims=[8],
                fc_dims=[512], dropout=0.0, seed=42, epochs=1, batch_size=4,
                learning_rate=0.001, weight_decay=0.0, lr_step=50, lr_gamma=0.1,
                fusion=dict(alpha=0.7, beta=0.3, gamma=0.4), C_grid=[0.1, 1, 10],
                liblinear_solver=2, liblinear_bias=1.0,
                choices={"scope": "Small engineering fixture; not Table I."})


def features(batch=2):
    generator = torch.Generator().manual_seed(1337)
    return {**{name: torch.randn(batch, 12, width, generator=generator)
               for name, width in (("conv3", 256), ("conv4", 512), ("conv5", 512))},
            "fc": torch.randn(batch, 4096, generator=generator)}


class RecipeTests(unittest.TestCase):
    def test_valid_recipe_is_detached_and_loadable_from_json(self):
        module = api("config")
        source = recipe()
        checked = module.validate_recipe(source)
        self.assertEqual(checked, source)
        source["gcn_dims"].append(99)
        source["fusion"]["alpha"] = 10
        self.assertEqual(checked["gcn_dims"], [8])
        self.assertEqual(checked["fusion"]["alpha"], 0.7)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipe.json"
            path.write_text(json.dumps(checked))
            self.assertEqual(module.load_recipe(path), checked)

    def test_checked_in_recipe_explicitly_records_independent_choices(self):
        checked = api("config").load_recipe(ROOT / "configs/dfagcn_independent_v1.json")
        self.assertEqual(checked["pooling_hw"], [7, 7])
        self.assertEqual(checked["gcn_dims"], [256, 256])
        self.assertEqual(checked["fc_dims"], [1024, 512])
        self.assertEqual(checked["knn_k"], 10)
        self.assertEqual(checked["C_grid"], [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
        self.assertGreaterEqual(len(checked["choices"]), 5)
        self.assertIn("independent", json.dumps(checked["choices"]).lower())

    def test_missing_and_unknown_fields_are_not_silently_filled(self):
        module = api("config")
        for field in ("pooling_hw", "gcn_dims", "fc_dims", "learning_rate", "C_grid"):
            changed = recipe()
            del changed[field]
            with self.subTest(field=field), self.assertRaises((TypeError, ValueError)):
                module.validate_recipe(changed)
        changed = recipe()
        changed["official_table_I_verified"] = True
        with self.assertRaises(ValueError):
            module.validate_recipe(changed)

    def test_rejects_invalid_architecture_or_optimization_values(self):
        module = api("config")
        invalid = [("schema", "official"), ("implementation", "official"),
                   ("pooling_hw", [1, 2]), ("pooling_hw", [3]), ("pooling_hw", [True, 4]),
                   ("knn_k", 0), ("knn_k", 12), ("knn_k", True),
                   ("metric", "position"), ("gcn_dims", []), ("gcn_dims", [0]),
                   ("gcn_dims", [True]), ("fc_dims", [128]), ("fc_dims", []),
                   ("dropout", -0.1), ("dropout", 1), ("dropout", float("nan")),
                   ("seed", -1), ("seed", True), ("epochs", 0), ("batch_size", True),
                   ("learning_rate", 0), ("learning_rate", float("inf")),
                   ("weight_decay", -1), ("lr_step", 0), ("lr_gamma", 0), ("lr_gamma", 2),
                   ("C_grid", []), ("C_grid", [1, 1]), ("C_grid", [1, 0.1]),
                   ("C_grid", [float("inf")]), ("liblinear_solver", 0),
                   ("liblinear_bias", float("nan")), ("choices", {"claim": True})]
        for name, value in invalid:
            changed = recipe()
            changed[name] = value
            with self.subTest(name=name, value=value), self.assertRaises((TypeError, ValueError)):
                module.validate_recipe(changed)
        for bad in ({"alpha": 0.7, "beta": 0.3}, {"alpha": True, "beta": 0.3, "gamma": 0.4},
                    {"alpha": -0.1, "beta": 0.3, "gamma": 0.4}):
            changed = recipe()
            changed["fusion"] = bad
            with self.subTest(fusion=bad), self.assertRaises((TypeError, ValueError)):
                module.validate_recipe(changed)


class BranchModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self, configuration=None):
        torch.manual_seed(42)
        return api("model").DFAGCNBranches(configuration or recipe(), num_classes=3)

    def test_three_raw_logit_heads_have_independent_parameters_and_gradients(self):
        model = self.model()
        inputs = features()
        logits = model(inputs)
        self.assertEqual(set(logits), {"conv3", "conv4", "conv5"})
        target = torch.tensor([0, 2])
        sum(nn.functional.cross_entropy(value, target) for value in logits.values()).backward()
        parameter_ids = []
        for name, branch in model.branches.items():
            self.assertEqual(logits[name].shape, (2, 3))
            ids = {id(p) for p in branch.parameters()}
            parameter_ids.append(ids)
            for parameter in branch.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        for left, right in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(parameter_ids[left] & parameter_ids[right])

    def test_export_is_last_relu_512_and_preserves_fc_then_fuses_published_order(self):
        model = self.model().eval()
        inputs = features()
        exported = model.export_features(inputs)
        for name in ("conv3", "conv4", "conv5"):
            self.assertEqual(exported[name].shape, (2, 512))
            self.assertTrue((exported[name] >= 0).all())
        self.assertIs(exported["fc"], inputs["fc"])
        fused = model.fused_features(inputs)
        self.assertEqual(fused.shape, (2, 5632))
        for actual, expected in zip(torch.split(fused, [512, 512, 512, 4096], dim=1),
                                    [exported["conv5"], 0.7 * exported["conv4"],
                                     0.3 * exported["conv3"], 0.4 * exported["fc"]]):
            torch.testing.assert_close(actual, expected)

    def test_eval_is_deterministic_and_samples_are_independent(self):
        configuration = recipe()
        configuration["dropout"] = 0.5
        model = self.model(configuration).eval()
        inputs = features()
        first, second = model(inputs), model(inputs)
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
            separately = torch.cat([model({k: v[i:i+1] for k, v in inputs.items()})[name]
                                    for i in range(2)], dim=0)
            torch.testing.assert_close(first[name], separately)

    def test_save_reload_state_preserves_logits_and_fused_features(self):
        model = self.model().eval()
        inputs = features()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        loaded = self.model().eval()
        loaded.load_state_dict(torch.load(buffer, weights_only=True))
        for name, value in model(inputs).items():
            torch.testing.assert_close(value, loaded(inputs)[name], rtol=0, atol=0)
        torch.testing.assert_close(model.fused_features(inputs), loaded.fused_features(inputs), rtol=0, atol=0)

    def test_forward_accepts_no_fc_but_export_requires_it(self):
        model = self.model().eval()
        inputs = features()
        del inputs["fc"]
        self.assertEqual(set(model(inputs)), {"conv3", "conv4", "conv5"})
        with self.assertRaises(ValueError):
            model.export_features(inputs)

    def test_rejects_incompatible_pooled_inputs(self):
        model = self.model().eval()
        bad_values = [torch.ones(2, 11, 256), torch.ones(1, 12, 256), torch.ones(2, 12, 255),
                      torch.ones(2, 12, 256, dtype=torch.float64),
                      torch.full((2, 12, 256), float("nan")), torch.empty(2, 12, 256, device="meta")]
        for value in bad_values:
            inputs = features()
            inputs["conv3"] = value
            with self.subTest(shape=value.shape, dtype=value.dtype, device=value.device), self.assertRaises((TypeError, ValueError)):
                model(inputs)
        for key in ("conv3", "conv4", "conv5"):
            inputs = features()
            del inputs[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                model(inputs)
        inputs = features()
        inputs["fc"] = torch.ones(2, 4095)
        with self.assertRaises(ValueError):
            model.export_features(inputs)

    def test_rejects_invalid_class_counts_and_detaches_recipe(self):
        module = api("model")
        for count in (0, 1, True, 1.5):
            with self.subTest(count=count), self.assertRaises((TypeError, ValueError)):
                module.DFAGCNBranches(recipe(), num_classes=count)
        configuration = recipe()
        model = self.model(configuration)
        configuration["fusion"]["alpha"] = 10
        self.assertEqual(model.recipe["fusion"]["alpha"], 0.7)


if __name__ == "__main__":
    unittest.main()
