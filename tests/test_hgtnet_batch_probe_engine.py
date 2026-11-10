"""Real tensor optimizer/measurement behavior for the bounded capacity probe."""
import importlib
import math
import unittest
from unittest.mock import patch
import torch
from torch import nn


def api():
    try:return importlib.import_module('external_baselines.hgtnet.batch_probe_engine')
    except ModuleNotFoundError as exc:raise AssertionError('batch16 numeric probe engine missing') from exc


class Tiny(nn.Module):
    def __init__(self):super().__init__();self.head=nn.Linear(3,45)
    def forward(self,images):return self.head(images.mean((2,3)))


class BatchProbeEngineTests(unittest.TestCase):
    def inputs(self):
        return torch.randn(16,3,224,224,generator=torch.Generator().manual_seed(42)),torch.arange(16,dtype=torch.long)

    def test_four_real_updates_and_three_measurements_without_input_mutation(self):
        m=Tiny();images,labels=self.inputs();original=images.clone();before=m.head.weight.detach().clone();events=[]
        record=api().run_batch_probe(m,images,labels,device='cpu',on_progress=events.append)
        self.assertTrue(record['completed']);self.assertFalse(record['formal_training_started'])
        self.assertEqual(record['optimizer_steps'],4);self.assertEqual(record['unique_images'],16)
        self.assertEqual(record['processed_images'],64);self.assertEqual(len(record['step_seconds']),3)
        self.assertTrue(all(math.isfinite(t) and t>0 for t in record['step_seconds']))
        self.assertAlmostEqual(record['compute_images_per_second'],16/record['median_step_seconds'])
        self.assertTrue(torch.equal(images,original));self.assertFalse(torch.equal(before,m.head.weight))
        self.assertEqual([v['optimizer_steps'] for v in events],[1,2,3,4])
        self.assertTrue(all(v['optimizer_step_completed'] for v in events))
        self.assertEqual(record['first_step_updated_groups'],{'head':True})
        self.assertIsNone(record['cuda_peak_allocated_bytes'])
        self.assertIsNone(record['accuracy_metrics']);self.assertFalse(record['checkpoint_saved'])
        self.assertEqual(record['input_scope'],'caller_provided_tensors')
        self.assertEqual(record['optimizer']['name'],'Adam')
        self.assertEqual(record['optimizer']['lr'],1e-5)
        self.assertEqual(record['optimizer']['weight_decay'],5e-5)

    def test_invalid_batches_rejected_before_parameter_update(self):
        images,labels=self.inputs()
        for x,y in [(images[:15],labels[:15]),(images.double(),labels),(images,labels.float()),
                    (images[:,:,:112,:112],labels),(images,labels+45)]:
            m=Tiny();before=m.head.weight.detach().clone();events=[]
            with self.subTest(shape=x.shape,dtype=x.dtype),self.assertRaises((ValueError,TypeError)):
                api().run_batch_probe(m,x,y,device='cpu',on_progress=events.append)
            self.assertTrue(torch.equal(before,m.head.weight));self.assertEqual(events,[])

    def test_fixed_small_budget_cannot_be_expanded_through_parameters(self):
        x,y=self.inputs()
        for kwargs in [{'batch_size':2},{'warmup_steps':0},{'warmup_steps':2},{'measured_steps':100},{'measured_steps':True}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                api().run_batch_probe(Tiny(),x,y,device='cpu',**kwargs)

    def test_nan_loss_does_not_report_a_successful_optimizer_step(self):
        class Broken(Tiny):
            def forward(self,images):return super().forward(images)*float('nan')
        x,y=self.inputs();events=[]
        with self.assertRaises(ValueError):api().run_batch_probe(Broken(),x,y,device='cpu',on_progress=events.append)
        self.assertEqual(events,[])

    def test_later_failure_retains_only_genuinely_completed_step_events(self):
        class Later(Tiny):
            calls=0
            def forward(self,images):
                self.calls+=1
                if self.calls==2:raise RuntimeError('second batch failed')
                return super().forward(images)
        x,y=self.inputs();events=[]
        with self.assertRaisesRegex(RuntimeError,'second batch failed'):
            api().run_batch_probe(Later(),x,y,device='cpu',on_progress=events.append)
        self.assertEqual(len(events),1);self.assertEqual(events[0]['optimizer_steps'],1)
        self.assertTrue(events[0]['optimizer_step_completed'])

    def test_frozen_or_non_fp32_model_cannot_understate_formal_memory(self):
        x,y=self.inputs()
        for mode in ['frozen','float64']:
            m=Tiny()
            if mode=='frozen':m.head.weight.requires_grad_(False)
            else:m.double()
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                api().run_batch_probe(m,x,y,device='cpu')

    def test_post_update_diagnostic_failure_does_not_erase_observed_update(self):
        engine=api();x,y=self.inputs();m=Tiny();before=m.head.weight.detach().clone();events=[]
        with patch.object(engine,'_finite_parameters',side_effect=[True,False]):
            with self.assertRaisesRegex(ValueError,'Updated probe parameters nonfinite'):
                engine.run_batch_probe(m,x,y,device='cpu',on_progress=events.append)
        self.assertFalse(torch.equal(before,m.head.weight));self.assertEqual(len(events),1)
        self.assertTrue(events[0]['optimizer_step_completed'])
        self.assertFalse(events[0]['post_update_checks_passed'])


if __name__=='__main__':unittest.main()
