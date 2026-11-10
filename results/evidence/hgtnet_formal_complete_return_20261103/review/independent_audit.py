"""Independent stdlib audit of delivered completion semantics and returned counts."""
import hashlib,json,math,re,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
bundle=ROOT/'output/hgtnet_nwpu_formal_3090_20261103_v1.pyz'
checks={}
checks['actual_v1_sha']=hashlib.sha256(bundle.read_bytes()).hexdigest()=='b2bc9077cf37e75c83fbb72e7c4047d2fb9ede04db8fb4e4a56344500ff572e1'
with zipfile.ZipFile(bundle) as z:
    manifest=json.loads(z.read('delivery_manifest.json'))
    engine=z.read('external_baselines/hgtnet/formal_engine.py').decode()
    entry=z.read('external_baselines/hgtnet/formal_entry.py').decode()
    protocol=z.read('external_baselines/hgtnet/formal_protocol.py').decode()
    checks['engine_current']=engine==(ROOT/'external_baselines/hgtnet/formal_engine.py').read_text()
    checks['entry_current']=entry==(ROOT/'external_baselines/hgtnet/formal_entry.py').read_text()
    checks['formal3090_manifest']=manifest['runner']=='external_baselines/hgtnet/formal_entry.py' and manifest['target_gpu']=='3090'
checks['100calibration_frozen']='max_epochs=100' in protocol
checks['full_test25200_frozen']='test=25200' in protocol
checks['test_order_count_gate']="labels!=[row['label'] for row in binding['records'][split]]" in engine
checks['equal_fresh_init_required']="if refit_initial!=initial:raise ValueError" in engine
checks['refit_before_reload_before_test']=engine.index("_notify(on_progress,state,'refit_complete'")<engine.index("reload_record=load_model(")<engine.index("_notify(on_progress,state,'before_test'")<engine.index("test,_=_evaluate(")
checks['reload_state_and_numeric_required']="if state_sha256(fresh)!=expected_state:raise ValueError" in engine and 'error>1e-6:raise ValueError' in engine
checks['full_test_file_before_engine_complete']=engine.index("atomic_json(output/'test.json',test)")<engine.index("return dict(schema='hgtnet-nwpu-formal-result-v1',completed=True")
checks['test_before_wrapper_final_input_recheck']=entry.index("result.update(_run_model(")<entry.index("_assert_unchanged(args,inputs,binding,include_test=True)")
checks['final_source_recipe_input_recheck']="_binding(args,inputs)!=binding" in entry and "COMMON._sha(args.manifest)!=DATA.MANIFEST_SHA" in entry and "COMMON._sha(args.weights)!=COMMON.WEIGHTS_SHA" in entry and 'DATA.validate_selected(args._test_root' in entry
summary=entry.index("_write(args.output/'summary.json',result);_write(args.output/'run.exit','0\\n')")
marker=entry.index("_write(args.output/'completion.marker',MARKER)",summary)
progress=entry.index("progress(dict(stage='complete'",marker)
complete=entry.index("print('HGTNET_NWPU_FORMAL_COMPLETE '",progress)
checks['summary_exit_marker_progress_COMPLETE_order']=summary<marker<progress<complete
checks['68930_updates_consistent']=100*math.ceil(5040/16)+95*math.ceil(6300/16)==68930
checkpoint_sha='716835556fe989739fc39e2d278188739b4d4454a14b5bf4d8d2b9efe060db1c'
state_sha='99c7a6a85176f75c37cdde268ec8d06dc318f4c2722db4e2f53bd61631eb3cea'
checks['returned_model_sha_format']=bool(re.fullmatch('[0-9a-f]{64}',checkpoint_sha))
checks['returned_state_sha_format']=bool(re.fullmatch('[0-9a-f]{64}',state_sha))
rate=6300/112.18204713500018
checks['fragment_train_rate_arithmetic']=math.isclose(rate,56.15871844822517,rel_tol=1e-12)
raw=HERE.parent/'reported_output.txt'
raw_record=None
if raw.exists():
    lines=raw.read_text().splitlines()
    events=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('HGTNET_FORMAL_PROGRESS ')]
    checks['reported_refit_complete']=any(e['stage']=='refit_complete' and e['epoch']==95 and e['optimizer_steps']==68930 for e in events)
    checks['reported_numeric_reload0']=any(e['stage']=='before_test' and e['checkpoint_reload']['reload_max_abs_error']==0.0 and e['checkpoint_reload']['sha256']==checkpoint_sha for e in events)
    checks['reported_complete_event']=any(e['stage']=='complete' and e['completed'] is True and e['optimizer_steps']==68930 for e in events)
    checks['reported_complete_path']=any(line=='HGTNET_NWPU_FORMAL_COMPLETE /home/featurize/work/hgtnet_nwpu_formal_3090_20261103_v1/runs/nwpu_seed42_01/summary.json' for line in lines)
    raw_record=dict(path=str(raw.relative_to(ROOT)),sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
result=dict(schema='hgtnet-formal-completion-marker-independent-review-v1',checks=checks,checks_count=len(checks),passed=all(checks.values()),
    raw_original_if_available=raw_record,
    derived=dict(calibration_updates=31500,refit_updates=37430,total_updates=68930,last_refit_train_images_per_second=rate),
    disposition='server-reported normal-program completion accepted; no repeat training or test required',
    boundaries=['first line truncated; no complete epoch-wall reconstructed','last refit trainaccuracy100% is not testaccuracy','no finaltestOA/F1, overallwall or peak-memory value in this excerpt','normal delivered source establishes completion requirements; remote source/runtime/checkpoint/predictions/exit bytes not directly retrieved'])
(HERE/'validation.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(checks=len(checks),passed=result['passed'],failed=[k for k,v in checks.items() if not v],derived=result['derived'],raw_available=raw_record is not None),indent=2))
if not result['passed']:raise SystemExit(1)
