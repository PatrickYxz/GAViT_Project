"""Independent CPU/stdlib boundary fixtures; never a CUDA or model execution."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import types
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
GPU={'name':'NVIDIA GeForce RTX 3090','uuid':'GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1',
     'memory_total_mib':24576,'memory_free_mib':24123,'driver_version':'610.57.04'}
MIB=1024**2


def api(path):
    module=types.ModuleType('_independent_hgtnet_runtime_probe')
    module.__file__=str(ROOT/'tools/run_hgtnet_nwpu_smoke.py')
    exec(compile(path.read_text(),str(path),'exec'),module.__dict__)
    return module


def invoke(module,*,memory=24123*MIB,available=True,count=1,name=GPU['name'],
           gpu=None,requested_uuid=GPU['uuid'],torch_version='2.2.2+cu121'):
    """Execute the real runtime function with explicit fake import/CUDA seams."""
    cuda_backend=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True))
    cudnn=types.SimpleNamespace(allow_tf32=True,benchmark=True,deterministic=False,version=lambda:8902)
    cuda=types.SimpleNamespace(is_initialized=lambda:False,is_available=lambda:available,
         device_count=lambda:count,get_device_name=lambda index:name,
         get_device_properties=lambda index:types.SimpleNamespace(total_memory=memory,name=name),
         manual_seed_all=lambda value:None)
    torch=types.ModuleType('torch');torch.__version__=torch_version;torch.cuda=cuda
    torch.version=types.SimpleNamespace(cuda='12.1');torch.backends=types.SimpleNamespace(cuda=cuda_backend,cudnn=cudnn)
    torch.float32='float32';torch.set_default_dtype=lambda value:None
    torch.set_num_threads=lambda value:None;torch.get_num_threads=lambda:8;torch.manual_seed=lambda value:None
    vision=types.ModuleType('torchvision');vision.__version__='0.17.2+cu121'
    numpy=types.ModuleType('numpy');numpy.__version__='1.26.4';numpy.random=types.SimpleNamespace(seed=lambda value:None)
    pil=types.ModuleType('PIL');pil.__version__='10.3.0'
    args=types.SimpleNamespace(gpu_uuid=requested_uuid,target_gpu='3090')
    try:
        with patch.dict(sys.modules,{'torch':torch,'torchvision':vision,'numpy':numpy,'PIL':pil}),\
             patch.object(sys,'version','3.11.8 fixture'),patch.dict(os.environ,{},clear=False):
            _,runtime=module._runtime_and_torch(args,dict(GPU) if gpu is None else gpu)
        return {'accepted':True,'runtime':runtime}
    except (ValueError,TypeError,OverflowError,AttributeError) as exc:
        return {'accepted':False,'error_type':type(exc).__name__,'error':str(exc)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--before',action='store_true');args=parser.parse_args()
    source=Path(__file__).with_name('before_runner.py') if args.before else ROOT/'tools/run_hgtnet_nwpu_smoke.py'
    module=api(source)
    cases=[
      ('exact_memory',True,{'memory':24576*MIB}),
      ('reported_453mib_gap',not args.before,{}),
      ('max_1024mib_gap',not args.before,{'memory':(24576-1024)*MIB}),
      ('max_gap_plus_one_byte',False,{'memory':(24576-1024)*MIB-1}),
      ('above_nvml_plus_one_byte',False,{'memory':24576*MIB+1}),
      ('zero_memory',False,{'memory':0}),
      ('negative_memory',False,{'memory':-1}),
      ('bool_memory',False,{'memory':True}),
      ('float_memory',False,{'memory':float(24123*MIB)}),
      ('wrong_name',False,{'name':'NVIDIA GeForce RTX 3080'}),
      ('wrong_count_zero',False,{'count':0}),
      ('wrong_count_two',False,{'count':2}),
      ('unavailable',False,{'available':False}),
      ('wrong_nvml_physical_memory',False,{'memory':24575*MIB,'gpu':dict(GPU,memory_total_mib=24575)}),
      ('wrong_nvml_uuid',False,{'memory':24576*MIB,'gpu':dict(GPU,uuid='GPU-other')}),
      ('wrong_torch_version',False,{'memory':24576*MIB,'torch_version':'2.2.2+cpu'}),
    ]
    records=[]
    for label,expected,kwargs in cases:
        observed=invoke(module,**kwargs)
        # Original equality was itself known to accept fractional bytes above nominal.
        if args.before and label=='above_nvml_plus_one_byte':expected=True
        records.append({'case':label,'expected_accept':expected,'passed':observed['accepted']==expected,**observed})
    if not args.before:
        accepted=next(row for row in records if row['case']=='reported_453mib_gap')
        runtime=accepted.get('runtime',{});probe=runtime.get('cuda_device',{})
        checks={
          'physical_nvml_record_preserved':runtime.get('gpu')==GPU,
          'physical_uuid_preserved':runtime.get('physical_gpu_uuid')==GPU['uuid'],
          'cuda_visible_device_uuid_preserved':runtime.get('cuda_visible_devices')==GPU['uuid'],
          'cuda_total_separate':probe.get('cuda_total_mib')==24123,
          'nvml_total_separate':probe.get('nvml_total_mib')==24576,
          'interface_gap_observed':probe.get('interface_gap_mib')==453,
          'explicit_engineering_bound':probe.get('max_interface_gap_mib')==1024,
          'fp32_seed_amp_preserved':all(runtime.get(key)==value for key,value in [('dtype','float32'),('seed',42),('amp',False)]),
          'tf32_preserved':runtime.get('matmul_tf32') is False and runtime.get('cudnn_tf32') is False,
        }
    else:checks={}
    report={'scope':'Independent stdlib fake runtime/CUDA boundary probes. No ML libraries, CUDA, data, training or GPU executed.',
            'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'before_fix':args.before,'cases':records,'observation_checks':checks,
            'successful':all(row['passed'] for row in records) and all(checks.values())}
    destination=Path(__file__).with_name('before_probe.json' if args.before else 'after_probe.json')
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'result':str(destination),'cases':len(records),'passed':sum(row['passed'] for row in records),
           'observation_checks':checks,'successful':report['successful']}))
    return 0 if report['successful'] else 1

if __name__=='__main__':raise SystemExit(main())
