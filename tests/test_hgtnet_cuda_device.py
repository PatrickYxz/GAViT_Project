"""Physical NVML capacity and CUDA-visible capacity have distinct meanings."""
import importlib.util
import os
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


def api():
    spec=importlib.util.spec_from_file_location('_hgt_cuda_check','tools/run_hgtnet_nwpu_smoke.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class CudaDeviceTests(unittest.TestCase):
    def gpu(self):
        return {'name':'NVIDIA GeForce RTX 3090','uuid':'GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1',
                'memory_total_mib':24576,'memory_free_mib':24123,'driver_version':'610.57.04'}

    def probe(self,**values):
        return dict({'available':True,'device_count':1,'name':self.gpu()['name'],
                     'total_memory_bytes':24123*1024**2},**values)

    def check(self,probe,gpu=None):
        module=api()
        self.assertTrue(hasattr(module,'_check_cuda_device'),'separate CUDA capacity check is missing')
        return module._check_cuda_device(probe,self.gpu() if gpu is None else gpu)

    def test_user_reported_453_mib_gap_keeps_both_observations(self):
        gpu=self.gpu();report=self.check(self.probe(),gpu)
        self.assertEqual(report['cuda_total_mib'],24123)
        self.assertEqual(report['nvml_total_mib'],24576)
        self.assertEqual(report['interface_gap_mib'],453)
        self.assertEqual(gpu['memory_total_mib'],24576)

    def test_exact_and_1gib_boundary_are_accepted(self):
        for mib in [24576,24123,23552]:
            with self.subTest(mib=mib):self.check(self.probe(total_memory_bytes=mib*1024**2))

    def test_bad_capacities_do_not_pass_identity_or_memory_gate(self):
        for value in [0,-1,True,None,24123.5,'24123',23552*1024**2-1,24576*1024**2+1]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.check(self.probe(total_memory_bytes=value))

    def test_wrong_name_count_and_unavailable_still_fail(self):
        for field,value in [('available',False),('available',1),('device_count',0),('device_count',2),
                            ('device_count',True),('name','NVIDIA GeForce RTX 3080')]:
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                self.check(self.probe(**{field:value}))

    def test_3080_default_capacity_policy_remains_bounded(self):
        gpu=dict(self.gpu(),name='NVIDIA GeForce RTX 3080',memory_total_mib=10240)
        self.check(self.probe(name=gpu['name'],total_memory_bytes=9874*1024**2),gpu)
        with self.assertRaises(ValueError):
            self.check(self.probe(name=gpu['name'],total_memory_bytes=8192*1024**2),gpu)

    def test_actual_runtime_adapter_accepts_reported_visible_capacity(self):
        module=api();gpu=self.gpu()
        with patch.dict(os.environ,{'HGTNET_TARGET_GPU':'3090'}):
            args=module.parser().parse_args(['run','--gpu-uuid',gpu['uuid']])
            torch=NS(__version__='2.2.2+cu121',float32='float32',
                cuda=NS(is_initialized=lambda:False,is_available=lambda:True,device_count=lambda:1,
                        get_device_name=lambda index:gpu['name'],get_device_properties=lambda index:NS(total_memory=24123*1024**2),
                        manual_seed_all=lambda seed:None),
                set_num_threads=lambda n:None,get_num_threads=lambda:8,set_default_dtype=lambda d:None,
                manual_seed=lambda seed:None,version=NS(cuda='12.1'),
                backends=NS(cuda=NS(matmul=NS(allow_tf32=True)),
                    cudnn=NS(allow_tf32=True,benchmark=True,deterministic=False,version=lambda:8902)))
            with patch.dict(sys.modules,{'torch':torch,'torchvision':NS(__version__='0.17.2+cu121'),
                'numpy':NS(__version__='1.26.4',random=NS(seed=lambda seed:None)), 'PIL':NS(__version__='10.3.0')}),\
                patch.object(module.sys,'version','3.11.8 (fixture CUDA boundary)'):
                try:actual,runtime=module._runtime_and_torch(args,gpu)
                except ValueError as exc:self.fail('reported healthy CUDA device was rejected: '+str(exc))
            self.assertIs(actual,torch)
            self.assertEqual(runtime['gpu']['memory_total_mib'],24576)
            self.assertEqual(runtime['cuda_device']['cuda_total_mib'],24123)
            self.assertFalse(runtime['amp']);self.assertFalse(runtime['matmul_tf32'])


if __name__=='__main__':unittest.main()
