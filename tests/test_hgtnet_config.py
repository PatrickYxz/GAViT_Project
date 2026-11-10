"""Architecture identity and explicit independent choices, without tensor imports."""
import copy
import importlib
import json
from pathlib import Path
import unittest


def api():
    try:return importlib.import_module('external_baselines.hgtnet.config')
    except ModuleNotFoundError as exc:raise AssertionError('HGTNet config API is missing') from exc


class ConfigTests(unittest.TestCase):
    def config(self):return api().load_config('configs/hgtnet_independent_v0.json')

    def test_full_candidate_preserves_three_downsamples_and_legal_graphs(self):
        cfg=self.config()
        self.assertEqual(cfg['stage_downsamples'],['patch_merge','patch_merge','conv2_same_channels'])
        self.assertEqual(cfg['graph_transition_paddings'],[1,0])
        self.assertEqual(cfg['graph_k'],20)
        self.assertFalse(cfg['official'])
        self.assertEqual(api().architecture_shapes(cfg)['stage_outputs'],[[256,28,28],[512,14,14],[512,7,7]])
        self.assertEqual(api().architecture_shapes(cfg)['graph_outputs'],[[256,7,7],[512,5,5]])

    def test_class_count_changes_head_identity_without_changing_body_shape(self):
        cfg=self.config();other=copy.deepcopy(cfg);other['num_classes']=30
        other=api().validate_config(other)
        self.assertNotEqual(api().config_sha256(cfg),api().config_sha256(other))
        self.assertEqual(api().architecture_shapes(cfg),api().architecture_shapes(other))

    def test_unsupported_or_false_architecture_identity_is_rejected(self):
        for key,value in [('official',True),('implementation','official_hgtnet'),('shape_policy','independent_14_to_7'),
                          ('stb_counts',[1,1,18]),('stage_channels',[96,192,384]),('graph_k',48),('fusion','concat')]:
            cfg=self.config();cfg[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):api().validate_config(cfg)

    def test_invalid_numeric_and_type_values_fail_before_model_build(self):
        for key,value in [('seed',True),('num_classes',19),('image_size',112),('norm_eps',float('nan')),
                          ('dropout',-0.1),('graph_k',True),('graph_exclude_self',False)]:
            cfg=self.config();cfg[key]=value
            with self.subTest(key=key),self.assertRaises((ValueError,TypeError)):api().validate_config(cfg)

    def test_unknown_missing_fields_and_mutable_aliases_are_not_silent(self):
        cfg=self.config();changed=copy.deepcopy(cfg);changed['new_hidden_choice']=1
        with self.assertRaises(ValueError):api().validate_config(changed)
        changed=copy.deepcopy(cfg);changed.pop('graph_operator')
        with self.assertRaises(ValueError):api().validate_config(changed)
        copied=api().validate_config(cfg);copied['choices']['identity']='changed'
        self.assertNotEqual(copied['choices'],cfg['choices'])

    def test_relative_mean_is_an_explicit_different_identity(self):
        cfg=self.config();other=copy.deepcopy(cfg);other['graph_operator']='relative_mean'
        self.assertEqual(api().validate_config(other)['graph_operator'],'relative_mean')
        self.assertNotEqual(api().config_sha256(cfg),api().config_sha256(other))

    def test_candidate_provenance_cannot_be_swapped_to_swin_t_or_unrelated_url(self):
        for key,value in [('kind','swin_t'),('url','https://example.invalid/weights'),('sha256_prefix','12345678')]:
            cfg=self.config();cfg['pretrained_candidate'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):api().validate_config(cfg)


if __name__=='__main__':unittest.main()
