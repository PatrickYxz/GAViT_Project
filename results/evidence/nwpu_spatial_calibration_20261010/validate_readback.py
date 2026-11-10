import ast,contextlib,csv,hashlib,io,json,shlex,subprocess,tempfile
from pathlib import Path
root=Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2');e=root/'results/evidence/nwpu_spatial_calibration_20261010'
cmd=(root/'docs/read_nwpu_spatial_closeout_20261010.txt').read_text().strip();source=(e/'readback.py').read_text()
assert ast.dump(ast.parse(shlex.split(cmd)[2]))==ast.dump(ast.parse(source))
for shell in ('/bin/bash','/bin/zsh'):subprocess.run([shell,'-n','-c',cmd],check=True)
checks=[]
with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
    b=Path(tmp);paths={'swin':b/'nwpu_ablation42_20260928_01/swin','soft':b/'nwpu_ablation42_20260928_01/soft','spatial':b/'nwpu_spatial42_calibrate_20261009_01'}
    counts={'swin':1187,'soft':1181,'spatial':1164};metrics={}
    for key,p in paths.items():
        p.mkdir(parents=True);(p/'run.exit').write_text('exit_code=0\n')
        rows=[{'path':f'image_{i}.jpg','label':i//28,'prediction':(i//28 if i<counts[key] else (i//28+1)%45)} for i in range(1260)]
        with (p/'validation_predictions.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=['path','label','prediction']);w.writeheader();w.writerows(rows)
        cm=[[sum(x['label']==i and x['prediction']==j for x in rows) for j in range(45)] for i in range(45)]
        f1=sum(200*cm[i][i]/(sum(cm[i])+sum(v[i] for v in cm)) for i in range(45))/45
        metrics[key]={'samples':1260,'oa_percent':counts[key]/1260*100,'macro_f1_percent':f1}
        (p/'validation_metrics.json').write_text(json.dumps(metrics[key]))
        (p/'best.pth').write_bytes(b'fixture-checkpoint')
        r={'status':'complete','phase':'calibrate','seed':42,'counts':{'train':5040,'val':1260},'training':{},'manifest_sha256':'ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e','pretrained_sha256':'f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36','code':{'git':{'commit':'67fe84d','dirty':False}},'architecture':{'soft_assignment':{'spatial_prior':'anchor_gaussian_v1'}},'best_epoch':24,'final_epoch':30,'best_val_oa_percent':metrics[key]['oa_percent'],'checkpoint_sha256':hashlib.sha256(b'fixture-checkpoint').hexdigest(),'hardware':{},'graph_capture':{}}
        (p/'run.json').write_text(json.dumps(r))
    h=[{'epoch':i,'validation':dict(metrics['spatial'],oa_percent=metrics['spatial']['oa_percent'] if i>=24 else 90.0),'train_loss':.1,'train_seconds':40.,'validation_seconds':5.} for i in range(1,31)]
    p=paths['spatial'];(p/'history.json').write_text(json.dumps(h))
    src=source.replace('/home/featurize/work',tmp).replace('a1c7f3b173e369ec3d40e2122fec91fa9ebc70d39ac6a96f5db90017c76fb579',hashlib.sha256(b'fixture-checkpoint').hexdigest())
    def check(name,error=None):
        with contextlib.redirect_stdout(io.StringIO()) as f:
            try:exec(compile(src,'<fixture>','exec'),{})
            except AssertionError as exc:
                assert error and error in str(exc),(name,str(exc))
            else:
                assert error is None and 'SPATIAL_CACHED_CLOSEOUT_PASS' in f.getvalue()
        checks.append(name)
    check('aligned three-model prediction metrics and history')
    (p/'best.pth').write_bytes(b'corrupted');check('changed checkpoint rejected','Checkpoint mismatch');(p/'best.pth').write_bytes(b'fixture-checkpoint')
    target=p/'validation_predictions.csv';original=target.read_bytes();target.write_bytes(original.replace(b'image_0.jpg',b'different_path.jpg'));check('different paths rejected','Prediction paths/labels mismatch');target.write_bytes(original)
    h[22]['validation']['oa_percent']=metrics['spatial']['oa_percent'];(p/'history.json').write_text(json.dumps(h));check('first-highest epoch mismatch rejected','Best epoch/curve mismatch')
validation={'scope':'Synthetic cached CSV/JSON files and dummy checkpoint; no training/source import/GPU/server access','syntax':['Python AST equality','bash -n','zsh -n'],'passed_cases':checks,'command_sha256':hashlib.sha256(cmd.encode()).hexdigest(),'server_readback_complete':False}
(e/'readback_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation,indent=2))
