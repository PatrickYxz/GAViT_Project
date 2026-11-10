"""Offline readback validation with synthetic saved predictions; no server/GPU."""
import ast, contextlib, csv, hashlib, importlib.util, io, json, shlex, subprocess, tempfile
from pathlib import Path
E=Path(__file__).resolve().parent
source=(E/'compact_closeout_payload.py').read_text()
flat='; '.join(source.splitlines())
assert ast.dump(ast.parse(source))==ast.dump(ast.parse(flat))
spec=importlib.util.spec_from_file_location('reference_metrics',E.parents[2]/'scene_classification/metrics.py')
metrics=importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics)
checks=[]
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp); names=('old_swin','old_soft','swin_ffn','soft_ffn')
    paths=dict(zip(names,[root/'nwpu_ablation42_20260928_01'/x for x in ('swin','soft')]+[root/'nwpu_ffn42_20261003_01'/x for x in names[2:]]))
    counts=(1187,1181,1185,1174); bests=(29,28,29,22); rows={}
    for idx,n in enumerate(names):
        p=paths[n]; p.mkdir(parents=True)
        truth=[i//28 for i in range(1260)]; pred=truth.copy()
        for i in range(idx*20,idx*20+1260-counts[idx]): pred[i]=(truth[i]+1)%45
        metric=metrics.classification_metrics(truth,pred,[f'class_{i}' for i in range(45)])
        rows[n]=[{'path':f'image_{i}.jpg','label':y,'prediction':pred[i]} for i,y in enumerate(truth)]
        with (p/'validation_predictions.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['path','label','prediction']); writer.writeheader(); writer.writerows(rows[n])
        (p/'best.pth').write_bytes(n.encode())
        (p/'run.exit').write_text('exit_code=0\n')
        report={'status':'complete','phase':'calibrate','seed':42,'counts':{'train':5040,'val':1260},'final_epoch':30,'best_epoch':bests[idx], 'best_val_oa_percent':metric['oa_percent'],'manifest_sha256':'fixture_manifest','pretrained_sha256':'fixture_pretrained','training':{'lr':0.0003},'total_parameters':123,'code':{'git':{'commit':'fixture','dirty':False}},'hardware':{'device_name':'fixture'},'checkpoint_sha256':hashlib.sha256(n.encode()).hexdigest()}
        history=[{'epoch':i,'validation':dict(metric,oa_percent=metric['oa_percent'] if i==bests[idx] else metric['oa_percent']-1),'train_loss':0.01,'train_seconds':1.2,'validation_seconds':0.3} for i in range(1,31)]
        for name,obj in [('run.json',report),('validation_metrics.json',metric),('history.json',history)]: (p/name).write_text(json.dumps(obj))
    class Remap(ast.NodeTransformer):
        def visit_Constant(self,node):
            return ast.copy_location(ast.Constant(str(root)),node) if node.value=='/home/featurize/work' else node
    code=compile(ast.fix_missing_locations(Remap().visit(ast.parse(flat))),'<fixture>','exec')
    def run():
        out=io.StringIO()
        with contextlib.redirect_stdout(out): exec(code,{})
        return out.getvalue()
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    output=run(); lines=output.splitlines(); assert len(lines)==9 and lines[-1].startswith('COMPACT_CLOSEOUT_PASS')
    for n,target,line in zip(names,counts,lines): assert json.loads(line[len(n)+1:])['correct']==target
    for line,(b,q) in zip(lines[4:8],[(names[1],names[3]),(names[2],names[3]),(names[0],names[3]),(names[0],names[2])]):
        data=json.loads(line.split(' ',3)[3]); bp=rows[b]; qp=rows[q]
        expected=sum(a['label']==a['prediction'] and z['label']==z['prediction'] for a,z in zip(bp,qp))
        assert data['both_correct']==expected
        assert sum(data[k] for k in ('both_correct','corrects','new_errors','both_wrong'))==1260
        assert data['net']==data['corrects']-data['new_errors']==counts[names.index(q)]-counts[names.index(b)]
        assert sum(data['class_net'].values())==data['net']
    assert all(p.read_bytes()==b for p,b in before.items()) and 'confusion_matrix' not in output and len(output)<8000
    checks+=['four_model_metrics_match_reference','four_paired_tables_and_all_nonzero_class_deltas','read_only_and_nine_output_lines']
    def reject(name,path,content):
        previous=path.read_bytes(); path.write_bytes(content)
        try:
            try: run()
            except AssertionError: pass
            else: raise RuntimeError('not rejected: '+name)
        finally: path.write_bytes(previous)
        checks.append(name)
    reject('bad_weight_hash',paths['soft_ffn']/'best.pth',b'corrupt')
    reject('failed_run',paths['soft_ffn']/'run.exit',b'exit_code=1')
    p=paths['soft_ffn']/'run.json'; report=json.loads(p.read_text()); report['best_epoch']=30
    reject('wrong_best_epoch',p,json.dumps(report).encode())
    report=json.loads(p.read_text()); report['training']={'lr':0.003}
    reject('mismatched_training',p,json.dumps(report).encode())
    p=paths['soft_ffn']/'validation_metrics.json'; metric=json.loads(p.read_text()); metric['macro_f1_percent']+=1
    reject('wrong_cached_macro_f1',p,json.dumps(metric).encode())
    p=paths['soft_ffn']/'validation_predictions.csv'; lines=p.read_text().splitlines(); lines[2]=lines[1]
    reject('duplicate_prediction_paths',p,('\n'.join(lines)+'\n').encode())
command="python -c '"+flat+"'"
assert shlex.split(command)[2]==flat
for shell in ('bash','zsh'): subprocess.run([shell,'-n','-c',command],check=True)
checks.append('bash_zsh_python_syntax')
print(json.dumps({'status':'pass','checks':checks,'synthetic_output_bytes':len(output),'synthetic_output_lines':9,'scope':'Offline synthetic fixture and reference metrics only; no server, CUDA or new experiment'},indent=2))
