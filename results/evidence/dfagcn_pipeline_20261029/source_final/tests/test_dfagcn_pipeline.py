"""Stage/data identities and actual cached-feature training with LIBLINEAR."""
import copy
import importlib
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scene_classification.data import prepare_manifest, select_records


def api():
    try:
        return importlib.import_module('external_baselines.dfagcn.pipeline')
    except ModuleNotFoundError as exc:
        raise AssertionError('DFAGCN stage pipeline is not implemented') from exc


LINEAR_ROOT = Path(os.environ.get('GAVIT_LIBLINEAR_ROOT',
    Path(__file__).resolve().parents[1]/'results/evidence/dfagcn_pipeline_20261029/liblinear/bin'))
TRAIN = str(LINEAR_ROOT/'train')
PREDICT = str(LINEAR_ROOT/'predict')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        images = self.root / 'images'
        for label, name in enumerate(('a', 'b')):
            (images / name).mkdir(parents=True)
            for i in range(10):
                Image.new('RGB', (12, 12), (label * 120, i * 13, 17)).save(images / name / f'{i}.png')
        self.manifest = self.root / 'manifest.json'
        self.data = prepare_manifest(images, self.manifest, dataset='synthetic')

    def fixtures(self):
        # Cache fixtures are arrays, not claimed pretrained or actual extraction.
        module = api()
        from external_baselines.dfagcn.config import load_recipe
        from external_baselines.dfagcn.cache import write_feature_cache
        from external_baselines.dfagcn.identity import build_cache_identity
        from external_baselines.dfagcn.extraction import PREPROCESSING
        recipe = load_recipe('configs/dfagcn_independent_v1.json')
        recipe.update(pooling_hw=[3, 4],knn_k=2,gcn_dims=[8],fc_dims=[512],dropout=0.0,
                      epochs=1,batch_size=4,C_grid=[0.1,1.0])
        manifest_sha = module.sha256_file(self.manifest)
        preprocessing = dict(PREPROCESSING, weights_kind='random_smoke')
        caches = {}
        for split in ('train','val','pool','test'):
            rows = select_records(self.data, split)
            identity = build_cache_identity(manifest_sha256=manifest_sha,
                class_names=self.data['class_names'],samples=[{'path':r['path'],'sha256':r['sha256']} for r in rows],
                weights_sha256='f'*64,preprocessing=preprocessing,
                feature_spec={'pooling_hw':[3,4],'conv_activation':'pre_relu','fc_activation':'post_relu'},
                source_sha256=module.feature_source_sha256(),stage='frozen_pretrained')
            values = np.asarray([r['label'] * 2 - 1 for r in rows],dtype=np.float32)
            batch = {name:np.broadcast_to(values[:,None,None],(len(rows),12,width)).copy()
                     for name,width in [('conv3',256),('conv4',512),('conv5',512)]}
            batch['fc'] = np.broadcast_to(values[:,None],(len(rows),4096)).copy()
            path = self.root / split
            write_feature_cache(path,identity,[batch])
            caches[split]=path
        return recipe,caches

    def test_calibration_refit_and_test_complete_with_actual_linear_binaries(self):
        recipe,caches=self.fixtures()
        module=api()
        selection=module.calibrate(self.manifest,caches['train'],caches['val'],recipe,self.root/'calibration',
                                  train_binary=TRAIN,predict_binary=PREDICT,mode='engineering')
        self.assertEqual(selection['phase'],'calibration')
        self.assertEqual(selection['selected_epoch'],1)
        report=module.refit(self.manifest,caches['pool'],self.root/'calibration'/'selection.json',self.root/'refit',
                            train_binary=TRAIN,predict_binary=PREDICT)
        self.assertEqual(report['fit_scope'],'train_plus_val')
        self.assertEqual(report['fit_samples'],10)
        metrics=module.evaluate(self.manifest,caches['test'],self.root/'refit',self.root/'evaluation',predict_binary=PREDICT)
        self.assertEqual(metrics['metrics']['samples'],10)
        self.assertEqual(metrics['phase'],'test')
        self.assertTrue((self.root/'refit'/'model.pt').exists())
        self.assertTrue((self.root/'calibration'/'final.pt').exists())
        with self.assertRaises(FileExistsError):
            module.calibrate(self.manifest,caches['train'],caches['val'],recipe,self.root/'calibration',
                             train_binary=TRAIN,predict_binary=PREDICT,mode='engineering')

    def test_stage_rejects_wrong_split_random_formal_and_calibration_as_final(self):
        recipe,caches=self.fixtures()
        module=api()
        with self.assertRaises(ValueError):
            module.calibrate(self.manifest,caches['pool'],caches['val'],recipe,self.root/'bad_pool',
                             train_binary=TRAIN,predict_binary=PREDICT)
        with self.assertRaises(ValueError):
            module.calibrate(self.manifest,caches['train'],caches['val'],recipe,self.root/'bad_formal',
                             train_binary=TRAIN,predict_binary=PREDICT,mode='formal')
        module.calibrate(self.manifest,caches['train'],caches['val'],recipe,self.root/'calibration',
                         train_binary=TRAIN,predict_binary=PREDICT,mode='engineering')
        with self.assertRaises(ValueError):
            module.evaluate(self.manifest,caches['test'],self.root/'calibration',self.root/'bad_test',predict_binary=PREDICT)

    def test_tampered_selection_recipe_and_refit_checkpoint_are_rejected(self):
        recipe,caches=self.fixtures()
        module=api()
        module.calibrate(self.manifest,caches['train'],caches['val'],recipe,self.root/'calibration',
                         train_binary=TRAIN,predict_binary=PREDICT,mode='engineering')
        selection_path=self.root/'calibration'/'selection.json'
        saved=selection_path.read_bytes()
        changed=json.loads(saved)
        changed['recipe']['seed']=43
        selection_path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            module.refit(self.manifest,caches['pool'],selection_path,self.root/'bad_refit',train_binary=TRAIN,predict_binary=PREDICT)
        selection_path.write_bytes(saved)
        module.refit(self.manifest,caches['pool'],selection_path,self.root/'refit',train_binary=TRAIN,predict_binary=PREDICT)
        with (self.root/'refit'/'model.pt').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaises(ValueError):
            module.evaluate(self.manifest,caches['test'],self.root/'refit',self.root/'bad_test',predict_binary=PREDICT)

    def test_subset_smoke_cannot_be_used_as_complete_final_refit(self):
        module=api()
        self.assertIn('train_per_class',inspect.signature(module.calibrate).parameters,
                      'explicit train/validation subset support is missing')
        recipe,caches=self.fixtures()
        from external_baselines.dfagcn.cache import write_feature_cache
        train=select_records(self.data,'train',per_class=1)
        val=select_records(self.data,'val',per_class=1)
        for name,rows in [('small_train',train),('small_val',val),('small_pool',sorted(train+val,key=lambda r:r['path']))]:
            template=json.loads((caches['train']/'index.json').read_text())['identity']
            template['samples']=[{'path':r['path'],'sha256':r['sha256']} for r in rows]
            values=np.asarray([r['label']*2-1 for r in rows],dtype=np.float32)
            batch={key:np.broadcast_to(values[:,None,None],(len(rows),12,width)).copy()
                   for key,width in [('conv3',256),('conv4',512),('conv5',512)]}
            batch['fc']=np.broadcast_to(values[:,None],(len(rows),4096)).copy()
            write_feature_cache(self.root/name,template,[batch])
        module.calibrate(self.manifest,self.root/'small_train',self.root/'small_val',recipe,self.root/'small_calibration',
                         train_binary=TRAIN,predict_binary=PREDICT,train_per_class=1,val_per_class=1)
        report=module.refit(self.manifest,self.root/'small_pool',self.root/'small_calibration'/'selection.json',
                            self.root/'small_refit',train_binary=TRAIN,predict_binary=PREDICT)
        self.assertEqual(report['fit_scope'],'train_plus_val_subset')
        self.assertEqual(report['fit_samples'],4)
        with self.assertRaisesRegex(ValueError,'full|pool'):
            module.evaluate(self.manifest,caches['test'],self.root/'small_refit',self.root/'bad_subset_test',predict_binary=PREDICT)
        with self.assertRaisesRegex(ValueError,'subset'):
            module.calibrate(self.manifest,self.root/'small_train',self.root/'small_val',recipe,self.root/'bad_formal_subset',
                             train_binary=TRAIN,predict_binary=PREDICT,mode='formal',train_per_class=1,val_per_class=1)


if __name__ == '__main__':
    unittest.main()
