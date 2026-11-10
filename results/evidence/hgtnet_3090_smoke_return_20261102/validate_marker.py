"""Audit a user-pasted marker against immutable delivered source, not GPU IO."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[3];E=Path(__file__).resolve().parent
bundle=ROOT/'output/hgtnet_nwpu_smoke_3090_20261102_fix1.pyz'
expected_sha='90c60dabcd185ffd0a1b575f8544e7363a469c4e3616528955cb3645c2ae7c42'
raw=(E/'reported_output.txt').read_text();lines=raw.splitlines()
events=[json.loads(line.split('SMOKE_PROGRESS ',1)[1]) for line in lines if 'SMOKE_PROGRESS ' in line]
training=[event for event in events if event.get('stage')=='training']
checks={
    'delivered_bundle_sha':hashlib.sha256(bundle.read_bytes()).hexdigest()==expected_sha,
    'reported_step_sequence':[e['optimizer_steps'] for e in training]==[30,60,90,120,150,180],
    'reported_samples_sequence':[e['samples_seen'] for e in training]==[60,120,180,240,300,360],
    'batch2_progress':all(e['samples_seen']==2*e['optimizer_steps'] and e['training_started'] is True for e in training),
    'reported_complete_event':events[-1]=={'completed':True,'stage':'complete','training_started':True},
    'expected_fix1_output_path':lines[-1]=='HGTNET_NWPU_SMOKE_COMPLETE /home/featurize/work/hgtnet_nwpu_smoke_3090_20261102_fix1/runs/nwpu_smoke_seed42_01/summary.json',
    'truncated_first_prefix_preserved':lines[0].startswith('GTNET_SMOKE_PROGRESS '),
}
with zipfile.ZipFile(bundle) as archive:
    manifest=json.loads(archive.read('delivery_manifest.json'))
    for name,item in manifest['files'].items():
        data=archive.read(name)
        checks['payload:'+name]=hashlib.sha256(data).hexdigest()==item['sha256']
        if name!='bundled_weights_provenance.json':checks['local_matches_delivery:'+name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==item['sha256']
    runner=archive.read('tools/run_hgtnet_nwpu_smoke.py').decode()
    checks['marker_after_summary_exit_and_complete_event']=(
        runner.index("_write(args.output/'summary.json',result)") < runner.index("(args.output/'run.exit').write_text('0\\n')")
        < runner.index("progress({'stage':'complete'") < runner.index("print('HGTNET_NWPU_SMOKE_COMPLETE "))
record={'scope':'user-reported server marker/immutable source audit; no direct server files or GPU access',
        'state':'SERVER_REPORTED_HGTNET_REAL_NWPU_SMOKE_COMPLETE_MARKER_SUMMARY_PENDING',
        'passed':all(checks.values()),'checks':checks,'passed_checks':sum(checks.values()),'total_checks':len(checks),
        'bundle_sha256':expected_sha,'reported_optimizer_steps':180,'reported_training_samples_seen':360,
        'expected_counts_from_delivered_contract':{'train':360,'val':90,'test':0},
        'summary_file_returned':False,'direct_exit_file_read':False,
        'pending':['smoke validation accuracy/macroF1','measured CUDA allocator peaks','wall time','checkpoint SHA/exact reload error','actual runtime identities'],
        'formal_training_started':False}
print(json.dumps(record,indent=2,sort_keys=True));raise SystemExit(0 if record['passed'] else 1)
