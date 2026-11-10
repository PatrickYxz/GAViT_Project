import ast,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
if sys.flags.optimize: raise RuntimeError('Disable PYTHONOPTIMIZE')
root=Path('/home/featurize/work/bigearth_soft42_handoff_20261013_01')
smoke=Path('/home/featurize/work/gpucheck_bigearth_soft42_4090_20261013_01')
formal=Path('/home/featurize/work/bigearth_soft42_20261013_01')
def require(ok,message):
    if not ok: raise RuntimeError(message)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name):
    spec=importlib.util.spec_from_file_location(name,root/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
require(not formal.exists(),'Formal output already exists: preserve it and return run.log/run.exit; do not relaunch')
expected={'run_bigearth_soft42.py': '970bdc3e3ef12229dc0d4a3c3a898db355bd9d5820d26d09d61d9882c3f28398', 'preflight_bigearth_soft42.py': 'fc16ecd41b46459b6ad15cab784aebdf33c7586723ac8271bed7d438cef5723f', 'handoff_bigearth_soft42.py': '2c7d7365c4722b6fd25973e6d54b385535db7a64b78011bfc5f40d4409b749b8', 'prepare_smoke_source.py': '1d86076351dd7f5290cf4d230cc002c89ee89c11cccbcc754778d0710d8748d4', 'train_reload_smoke_source.py': 'd67105cc77fd16f7d9e836e31668f7cd8f012516af057b339cc03a58fffbb2f9'}
wanted={'handoff_bigearth_soft42.py': 'f85e1f27587ebed92c056fb4c7c106d771134bbc506499ec08743d825c533041', 'run_bigearth_soft42.py': '09722f10542bed449d2e80df0041387e113cab58db403b3979e07f192194f767'}
manifest_path=root/'manifest.json';old_manifest=manifest_path.read_bytes();manifest=json.loads(old_manifest)
require(manifest['seed']==42 and manifest['files']==expected,'Handoff manifest differs or is already repaired; inspect without overwriting')
for name,sha in expected.items():require(digest(root/name)==sha,'Unexpected scheduler file: '+name)
require(not (root/'launch.claim').exists(),'Existing launch claim: inspect before repeating')
controller=load('handoff_bigearth_soft42');helper=load('run_bigearth_soft42');ready=json.loads((root/'ready.json').read_text())
receipt=controller.check_smoke(smoke,ready,helper)
require(receipt['checkpoint_sha256']=='4a6ae6e43001ffe5d729f7eb4e50ef9da4951e3b8422b2e5342b02f88a813398','Smoke differs from the reviewed receipt')
require(receipt['hardware']['gpu_identity'].startswith('GPU-44825cf0-bd98-bfdc-efbc-0100238a6789,'),'GPU differs from reviewed smoke')
replacements={'handoff_bigearth_soft42.py': ("    exec(compile(train,str(root/'train_reload_smoke_source.py'),'exec'), ns)\n    check_smoke(SMOKE,ready,helper)\n", "    exec(compile(train,str(root/'train_reload_smoke_source.py'),'exec'), ns)\n    # The frozen worker writes run.exit in atexit. Return so it can finish;\n    # launch() checks that marker and the saved artifacts in a new process.\n"), 'run_bigearth_soft42.py': ("            ns['b'].save(root/'preflight.json',ns['report'])", "            load_base().save(root/'preflight.json',ns['report'])")}
updated={}
for name,(before,after) in replacements.items():
    contents=(root/name).read_text();require(contents.count(before)==1,'Unexpected scheduler structure: '+name)
    updated[name]=contents.replace(before,after).encode();ast.parse(updated[name])
    require(hashlib.sha256(updated[name]).hexdigest()==wanted[name],'Replacement SHA mismatch: '+name)
backup=root/'scheduler_repair_20261016'
require(not backup.exists(),'Repair backup exists: inspect current scheduler before repeating')
for name in updated:require(not (root/(name+'.repair.tmp')).exists(),'Existing temporary file: preserve and inspect')
backup.mkdir()
(backup/'manifest.json').write_bytes(old_manifest)
for name in updated:(backup/name).write_bytes((root/name).read_bytes())
(backup/'smoke_summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
for name,data in updated.items():
    tmp=root/(name+'.repair.tmp')
    with tmp.open('xb') as f:f.write(data)
    tmp.replace(root/name)
manifest['files'].update(wanted);manifest['package_revision']='scheduler_exit_and_preflight_save_fix_20261016'
tmp=root/'manifest.json.repair.tmp'
with tmp.open('x') as f:f.write(json.dumps(manifest,indent=2)+'\n')
tmp.replace(manifest_path)
for name,sha in manifest['files'].items():require(digest(root/name)==sha,'Patched identity failed: '+name)
print('SOFT42_SCHEDULER_REPAIR_PASS backup='+str(backup),flush=True)
subprocess.run([sys.executable,'-B','-u',str(root/'handoff_bigearth_soft42.py'),'launch'],check=True)
