"""Cache metadata identity tests; no image/tensor cache is implied."""
import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path


def api():
    try:
        return importlib.import_module("external_baselines.dfagcn.identity")
    except ModuleNotFoundError as exc:
        raise AssertionError("DFAGCN cache identity interface is not implemented") from exc


def inputs():
    return dict(manifest_sha256="a" * 64, class_names=["airport", "forest"],
                samples=[{"path": "airport/a.jpg", "sha256": "b" * 64},
                         {"path": "forest/b.jpg", "sha256": "c" * 64}],
                weights_sha256="d" * 64, preprocessing={"resize": [224, 448], "channels": "RGB"},
                feature_spec={"conv": "pre_relu", "fc": "post_relu"},
                source_sha256="e" * 64, stage="frozen_pretrained")


class CacheIdentityTests(unittest.TestCase):
    def test_canonical_key_is_order_independent_for_mapping_fields(self):
        module = api()
        identity = module.build_cache_identity(**inputs())
        reordered = dict(reversed(list(identity.items())))
        self.assertEqual(module.cache_key(identity), module.cache_key(reordered))
        self.assertEqual(len(module.cache_key(identity)), 64)

    def test_every_provenance_field_and_sample_order_changes_key(self):
        module = api()
        base = inputs()
        original = module.cache_key(module.build_cache_identity(**base))
        variants = []
        for name in ("manifest_sha256", "weights_sha256", "source_sha256"):
            changed = copy.deepcopy(base)
            changed[name] = "f" * 64
            variants.append(changed)
        for name, value in (("class_names", ["forest", "airport"]),
                            ("preprocessing", {"resize": [224, 448], "channels": "BGR"}),
                            ("feature_spec", {"conv": "post_relu"})):
            changed = copy.deepcopy(base)
            changed[name] = value
            variants.append(changed)
        changed = copy.deepcopy(base)
        changed["samples"].reverse()
        variants.append(changed)
        changed = copy.deepcopy(base)
        changed["samples"][0]["sha256"] = "f" * 64
        variants.append(changed)
        changed = copy.deepcopy(base)
        changed["samples"][0]["path"] = "airport/other.jpg"
        variants.append(changed)
        for changed in variants:
            self.assertNotEqual(original, module.cache_key(module.build_cache_identity(**changed)))

    def test_learned_stages_require_fit_identity_and_cannot_alias(self):
        module = api()
        keys = []
        for stage in ("calibration", "refit"):
            args = inputs()
            args["stage"] = stage
            with self.assertRaises(ValueError):
                module.build_cache_identity(**args)
            args.update(fitted_on_sha256="1" * 64, training_config={"seed": 42, "epochs": 1})
            keys.append(module.cache_key(module.build_cache_identity(**args)))
            args["training_config"]["seed"] = 43
            self.assertNotEqual(keys[-1], module.cache_key(module.build_cache_identity(**args)))
            args["fitted_on_sha256"] = "2" * 64
            self.assertNotEqual(keys[-1], module.cache_key(module.build_cache_identity(**args)))
        self.assertNotEqual(*keys)

    def test_returned_identity_is_detached_from_caller_containers(self):
        module = api()
        args = inputs()
        identity = module.build_cache_identity(**args)
        key = module.cache_key(identity)
        args["samples"][0]["path"] = "changed.jpg"
        args["preprocessing"]["resize"].append(32)
        self.assertEqual(key, module.cache_key(identity))

    def test_exclusive_json_roundtrip_rejects_wrong_identity_and_tampering(self):
        module = api()
        identity = module.build_cache_identity(**inputs())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            module.write_cache_index(path, identity)
            self.assertEqual(module.read_cache_index(path, expected_identity=identity), identity)
            with self.assertRaises(FileExistsError):
                module.write_cache_index(path, identity)
            changed = inputs()
            changed["weights_sha256"] = "f" * 64
            with self.assertRaises(ValueError):
                module.read_cache_index(path, expected_identity=module.build_cache_identity(**changed))
            record = json.loads(path.read_text())
            record["identity"]["samples"][0]["sha256"] = "0" * 64
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                module.read_cache_index(path, expected_identity=identity)

    def test_rejects_incomplete_or_ambiguous_inputs(self):
        module = api()
        invalid = [("manifest_sha256", "not-a-sha"), ("weights_sha256", None),
                   ("class_names", []), ("class_names", ["same", "same"]),
                   ("samples", []), ("samples", [{"path": "../a", "sha256": "a" * 64}]),
                   ("samples", [{"path": "/a", "sha256": "a" * 64}]),
                   ("samples", [{"path": "a", "sha256": "z" * 64}]),
                   ("samples", inputs()["samples"] * 2),
                   ("stage", "test_tuned"), ("preprocessing", {}),
                   ("preprocessing", {"std": float("nan")}),
                   ("feature_spec", {1: "ambiguous-key"})]
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                args = inputs()
                args[name] = value
                with self.assertRaises((TypeError, ValueError)):
                    module.build_cache_identity(**args)
        with self.assertRaises(ValueError):
            module.build_cache_identity(**inputs(), fitted_on_sha256="f" * 64)


if __name__ == "__main__":
    unittest.main()
