"""Build the pinned, single-line inference-only Featurize handoff."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import shlex
import sys
import zlib

TARGET = '/home/featurize/work/ben_gridpair42_visual_20261020_01'
ROWS = [97389,14680,28450,4439]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_payload(repo):
    repo = Path(repo)
    wt = repo/'.worktrees/sparse-hybrid-4n-top2'
    source = repo/'results/evidence/ben_grid3_seed42_prepare_20261018/source'
    manifest = json.loads((source/'snapshot.json').read_text())
    assert manifest['snapshot_sha256'] == 'f82d37a9cc1d18920a553fa1e4fd516e9d9ebda9d9abddb6595e12a69a944f58'
    for name,digest in manifest['files'].items():
        assert sha((source/name).read_bytes()) == digest, name
    archive = wt/'results/evidence/pair44_feedback_k_7ehbne/case_sheet_return_20260919'
    original = json.loads((archive/'evidence.json').read_text())
    cases=[]
    for row in ROWS:
        stem=f'row_{row}'; jp,ip=archive/(stem+'.json'),archive/(stem+'.png')
        assert sha(jp.read_bytes()) == original['source_file_sha256'][jp.name]
        assert sha(ip.read_bytes()) == original['source_file_sha256'][ip.name]
        record = json.loads(jp.read_text())
        assert record['row_index'] == row
        cases.append(dict(row_index=row,labels=record['labels'],patch_path=record['patch_path'],
                          png_path=original['source']+'/'+ip.name,png_sha256=sha(ip.read_bytes()),
                          historical_case_json_sha256=sha(jp.read_bytes())))
    tree=ast.parse((source/'models/bigearth_dataset.py').read_text())
    classes=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id == 'CLASSES_19' for t in n.targets))
    control='/home/featurize/work/GAViT_Project/checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.pth'
    cfg=dict(dataset='BigEarthNet-19',seed=42,device='cpu',classes=classes,
             training_root='/home/featurize/work/ben_grid3_seed42_20261018_01',snapshot=manifest,
             checkpoints={'9':dict(path='/home/featurize/work/ben_grid3_seed42_20261018_01/train/gavit.pth',
                                   sha256='c4b77953d6dd2908d7a61ebe7a34f4004e8c219a0843de6ab05aaf207460feff'),
                          '16':dict(path=control,sha256='9b9acb76d93fa8799e979a227ce1d2496573da937b9329ab0ce3ca1c3f29859e')},
             architecture=dict(num_classes=19,knn_k=5,gat_hidden=256,gat_heads=4,gat_layers=2,
                               dropout=.1,grouping='attentive_spatial',edge_type='sparse_hybrid',integration='token_feedback'),
             cases=cases,selection='All four locally archived original-input historical diagnostic validation cases; fixed before this K9/K16 capture; no population inference.',
             val_csv='/home/featurize/work/GAViT_Project/bigearth_files/splits/val.csv',
             val_csv_sha256='1c6627cb48e33588538c4e411e4f1b94560827f27d59fb7aeb0ea2b5d30c2a45',
             reported_metrics=json.loads((repo/'results/evidence/ben_grid3_seed42_metrics_20261020/metrics.json').read_text()))
    files={'ben_grid_pair_visual.py':(wt/'tools/ben_grid_pair_visual.py').read_text(),
           'config.json':json.dumps(cfg,ensure_ascii=False,indent=2)+'\n'}
    return dict(files=files,sha256={name:sha(text.encode()) for name,text in files.items()})


def make_command(payload):
    packed=base64.b64encode(zlib.compress(json.dumps(payload,ensure_ascii=False).encode(),9)).decode()
    bootstrap=f'''import base64,json,zlib,subprocess,sys
from pathlib import Path
payload=json.loads(zlib.decompress(base64.b64decode("{packed}")))
ns={{"__name__":"frozen_ben_grid_pair_installer"}}
exec(compile(payload["files"]["ben_grid_pair_visual.py"],"<frozen_ben_grid_pair_visual>","exec"),ns)
target=ns["install"](payload,Path("{TARGET}"))
print("BEN_GRID_PAIR_VISUAL_INSTALLED "+str(target),flush=True)
sys.exit(subprocess.run([sys.executable,"-B","-u",str(target/"ben_grid_pair_visual.py"),str(target),"cpu"]).returncode)
'''
    code=bootstrap.encode()
    encoded=base64.b64encode(zlib.compress(code,9)).decode()
    arg=f'import base64,zlib,hashlib; s=zlib.decompress(base64.b64decode("{encoded}")); assert hashlib.sha256(s).hexdigest()=="{sha(code)}"; exec(compile(s,"<ben_grid_pair_visual_handoff>","exec"))'
    return 'python -B -c '+shlex.quote(arg)


def main():
    repo=Path(__file__).resolve().parents[3]
    out=repo/'results/evidence/ben_gridpair42_visual_prepare_20261020'
    out.mkdir(parents=True,exist_ok=True)
    payload=make_payload(repo); command=make_command(payload)
    assert len(command.encode()) < 64000 and '\n' not in command
    (out/'release_payload.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    (out/'install_and_capture_command.txt').write_text(command+'\n')
    (out/'config.json').write_text(payload['files']['config.json'])
    print(json.dumps(dict(status='PREPARED_ONLY',target=TARGET,command_bytes=len(command.encode()),
                         files_sha256=payload['sha256'],new_training=False,server_capture_executed=False),indent=2))


if __name__ == '__main__':
    main()
