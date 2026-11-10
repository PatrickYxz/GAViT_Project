"""Target profile must survive an actual source zipapp process boundary."""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from tools import package_hgtnet_nwpu_smoke as delivery


def runner():
    spec=importlib.util.spec_from_file_location('_test_hgt3090_runner','tools/run_hgtnet_nwpu_smoke.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class TargetDeliveryTests(unittest.TestCase):
    def test_direct_source_target_also_controls_scratch_namespace(self):
        module=runner()
        with patch.dict(os.environ,{},clear=True):
            args=module.parser().parse_args(['inspect','--target-gpu','3090'])
            self.assertIn('rtx3090',str(args.scratch_root))
            self.assertNotIn('3080',str(args.scratch_root))

    def test_runtime_checks_selected_3090_instead_of_fixed_3080(self):
        module=runner()
        with patch.dict(os.environ,{'HGTNET_TARGET_GPU':'3090'}):
            args=module.parser().parse_args(['run','--gpu-uuid','GPU-current3090'])
            self.assertTrue(hasattr(args,'target_gpu'),'runtime lacks explicit GPU profile')
            gpu={'name':'NVIDIA GeForce RTX 3090','uuid':args.gpu_uuid,'memory_total_mib':24576,
                 'memory_free_mib':24123,'driver_version':'610.57.04'}
            torch=NS(__version__='2.2.2+cu121',float32='float32',
                cuda=NS(is_initialized=lambda:False,is_available=lambda:True,device_count=lambda:1,
                        get_device_name=lambda index:gpu['name'],get_device_properties=lambda index:NS(total_memory=24576*1024**2),
                        manual_seed_all=lambda seed:None),
                set_num_threads=lambda n:None,get_num_threads=lambda:8,set_default_dtype=lambda d:None,
                manual_seed=lambda seed:None,version=NS(cuda='12.1'),
                backends=NS(cuda=NS(matmul=NS(allow_tf32=True)),
                    cudnn=NS(allow_tf32=True,benchmark=True,deterministic=False,version=lambda:8902)))
            with patch.dict(sys.modules,{'torch':torch,'torchvision':NS(__version__='0.17.2+cu121'),
                'numpy':NS(__version__='1.26.4',random=NS(seed=lambda seed:None)), 'PIL':NS(__version__='10.3.0')}),\
                patch.object(module.sys,'version','3.11.8 (fixture CUDA boundary)'):
                actual,runtime=module._runtime_and_torch(args,gpu)
            self.assertIs(actual,torch)
            self.assertEqual(runtime['physical_gpu_uuid'],args.gpu_uuid)
            self.assertEqual(runtime['gpu']['name'],'NVIDIA GeForce RTX 3090')
            self.assertFalse(runtime['matmul_tf32'])

    def test_parser_honors_delivery_target_and_command_binds_profile(self):
        module=runner()
        with patch.dict(os.environ,{'HGTNET_TARGET_GPU':'3090'}):
            args=module.parser().parse_args(['inspect'])
            self.assertTrue(hasattr(args,'target_gpu'),'runner has no target profile')
            self.assertEqual(args.target_gpu,'3090')
            self.assertIn('--target-gpu 3090',module.run_command(args,'GPU-current3090'))
            self.assertIn('3090',str(args.scratch_root))
            self.assertNotIn('3080',str(args.scratch_root))

    def test_build_actual_3090_bundle_and_target_reaches_bootstrap(self):
        self.assertIn('target_gpu',inspect.signature(delivery.build_bundle).parameters,
                      'builder cannot bind a3090 target')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'repository'
            fake='import json,os,sys;print(json.dumps({"target":os.environ.get("HGTNET_TARGET_GPU"),"action":sys.argv[1]}))\n'
            for name,text in {'external_baselines/__init__.py':'',
                'external_baselines/hgtnet/__init__.py':'',
                'configs/hgtnet_independent_v0.json':'{}',
                'tools/run_dfagcn_nwpu_smoke.py':'',delivery.RUNNER:fake}.items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
            bundle=Path(tmp)/'hgt3090.pyz'
            with patch.object(delivery,'ROOT',root):record=delivery.build_bundle(bundle,target_gpu='3090')
            self.assertEqual(record['manifest']['target_gpu'],'3090')
            env=dict(os.environ,HGTNET_TARGET_GPU='3080')
            result=subprocess.run([sys.executable,str(bundle),'inspect'],capture_output=True,text=True,env=env)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout),{'target':'3090','action':'inspect'})

    def test_unknown_profile_is_rejected_without_creating_files(self):
        self.assertIn('target_gpu',inspect.signature(delivery.build_bundle).parameters)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'invalid.pyz'
            for value in ['4090','3090Ti',3090,True,None]:
                with self.subTest(value=value),self.assertRaises(ValueError):
                    delivery.build_bundle(path,target_gpu=value)
                self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
