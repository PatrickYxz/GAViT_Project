"""Frozen AID formal identities and explicit bounded CPU fixture behavior."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest


def api():
    try:
        return importlib.import_module('external_baselines.hgtnet.aid_formal_protocol')
    except ModuleNotFoundError as exc:
        raise AssertionError('AID formal protocol interface must be implemented') from exc


def binding():
    p = api()
    recipe = dict(copy.deepcopy(p.RECIPE), max_epochs=2, batch_size=2, eval_batch_size=2)
    return dict(schema='hgtnet-aid-formal-binding-v1', input_scope='cpu_fixture', recipe=recipe,
                recipe_sha256=p.recipe_sha256(recipe), counts=dict(train=4, val=2, pool=6, test=3),
                class_names=[f'class_{i:02d}' for i in range(30)],
                records=dict(val=[dict(path=f'class_00/val_{i}.png', label=0) for i in range(2)],
                             test=[dict(path=f'class_01/test_{i}.png', label=1) for i in range(3)]),
                manifest_sha256='a'*64, config_sha256='b'*64, weights_sha256='c'*64,
                source_identity={'fixture.py':'d'*64})


class AidFormalProtocolTests(unittest.TestCase):
    def test_cpu_fixture_accepts_thirty_classes_and_returns_owned_snapshot(self):
        value = binding(); checked = api().validate_binding(value)
        value['class_names'][0] = 'changed'; value['recipe']['betas'][0] = 0
        self.assertEqual(checked['class_names'][0], 'class_00')
        self.assertEqual(checked['recipe']['betas'], [.9, .999])

    def test_old_scope_wrong_classes_recipe_and_cpu_bounds_fail_closed(self):
        mutations = [('schema', 'hgtnet-formal-binding-v1'), ('input_scope', 'real_nwpu'),
                     ('class_names', [f'c{i}' for i in range(45)]), ('manifest_sha256', 'bad')]
        for key, value in mutations:
            bad = binding(); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): api().validate_binding(bad)
        for key, value in [('max_epochs', 4), ('batch_size', 17), ('eval_batch_size', True), ('lr', 1e-4)]:
            bad = binding(); bad['recipe'][key] = value
            bad['recipe_sha256'] = api().recipe_sha256(bad['recipe'])
            with self.subTest(key=key), self.assertRaises(ValueError): api().validate_binding(bad)

    def test_real_aid_requires_frozen_full_counts_and_input_identities(self):
        p = api(); value = binding(); value.update(input_scope='real_aid', recipe=copy.deepcopy(p.RECIPE),
            recipe_sha256=p.recipe_sha256(p.RECIPE), counts=dict(train=4000, val=1000, pool=5000, test=5000),
            manifest_sha256='30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22',
            config_sha256='f313abb893b29d674ace281d8d2905670ee65db131f4240640079123e783589b',
            weights_sha256='68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893')
        value['records'] = {split:[dict(path=f'class_00/{split}_{i:05d}.png',label=0) for i in range(n)]
                            for split,n in [('val',1000),('test',5000)]}
        self.assertEqual(p.validate_binding(value)['counts']['test'], 5000)
        for key in ['manifest_sha256','config_sha256','weights_sha256']:
            bad = copy.deepcopy(value); bad[key] = '0'*64
            with self.subTest(key=key), self.assertRaises(ValueError): p.validate_binding(bad)
        value['counts']['train'] = 3999
        with self.assertRaises(ValueError): p.validate_binding(value)

    def test_ordered_evaluation_rejects_overlap_unsafe_paths_and_invalid_labels(self):
        for mutation in ['overlap','unsafe','label','duplicate']:
            value = binding()
            if mutation == 'overlap': value['records']['test'][0] = copy.deepcopy(value['records']['val'][0])
            elif mutation == 'unsafe': value['records']['test'][0]['path'] = '../outside.png'
            elif mutation == 'label': value['records']['test'][0]['label'] = 30
            else: value['records']['test'][1] = copy.deepcopy(value['records']['test'][0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): api().validate_binding(value)

    def test_config_load_binds_full_disclosed_recipe_and_rejects_changes(self):
        p = api(); root = Path(__file__).resolve().parents[1]
        loaded = p.load_recipe(root/'configs/hgtnet_aid_formal_v1.json')
        self.assertEqual(loaded['schema'], 'hgtnet-aid-independent-formal-recipe-v1')
        self.assertEqual(loaded['train_transform'], ['RGB','Resize256x256_bilinear','RandomCrop224',
            'RandomHorizontalFlip_p0.5','ToTensor','ImageNet_normalize'])
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            path = Path(directory)/'bad.json'; loaded['scheduler']='cosine'; path.write_text(json.dumps(loaded))
            with self.assertRaises(ValueError): p.load_recipe(path)


if __name__ == '__main__': unittest.main()
