"""Export a hash-identified K9 extension of the historical fixed42 source.

No working checkout is edited. K16 computation stays unchanged. The exported
source is explicitly identified as modified, rather than a clean Git commit.
"""
import hashlib
import json
from pathlib import Path
import subprocess

BASE = 'cd39be91a013bcf98099808c451300f87d3898b7'
FILES = ('experiment_identity.py', 'utils.py', 'train_bigearth.py', 'test_bigearth.py',
         'models/__init__.py', 'models/bigearth_dataset.py', 'models/gavit.py',
         'models/graph_construction.py', 'models/graph_reasoning.py',
         'models/region_grouping.py', 'models/swin_backbone.py',
         'models/swin_features.py', 'models/swin_region_only.py')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def topology(k):
    if type(k) is not int or k not in (9,16):
        raise ValueError('Only fixed K9 and K16 are approved for this ablation')
    side=3 if k==9 else 4
    spatial=4*side*(side-1)
    return dict(name='sparse_hybrid_4n_top2' if k==16 else 'sparse_hybrid_grid3_4n_top2',
                spatial_connectivity=4,feature_k=2,spatial_directed_edges=spatial,
                feature_directed_edges=2*k,total_directed_edges=spatial+2*k,
                message_direction='neighbor_to_query',cosine_role='topology_only')


def replace_once(text,old,new):
    if text.count(old)!=1:
        raise RuntimeError('Frozen source anchor changed: '+old[:90])
    return text.replace(old,new,1)


IDENTITY_SHIM = '''

# This export is modified source, not a clean checkout. Formal gating uses
# the complete snapshot manifest; the outer runner pins its digest as well.
def _verified_snapshot_state():
    import hashlib
    from pathlib import Path
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'snapshot.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen snapshot changed: '+name)
    return dict(commit='cd39be9',dirty=True,base_commit=manifest['base_commit'],
                snapshot_sha256=manifest['snapshot_sha256'],provenance='modified_frozen_snapshot')

def get_git_state(cwd=None):
    return _verified_snapshot_state()

def assert_clean_git_state(state):
    if state!=_verified_snapshot_state():
        raise RuntimeError('Unverified experimental snapshot identity')
'''


def patched_sources(original):
    result=dict(original)
    text=result['models/graph_construction.py'].decode()
    start=text.index('def build_sparse_hybrid_graph(')
    end=text.index('\ndef ',start+5) if '\ndef ' in text[start+5:] else len(text)
    old=text[start:end]; new=old
    new=replace_once(new,'if num_regions != 16:', 'if num_regions not in (9, 16):')
    new=replace_once(new,'    spatial_sources: list[int] = []','    side = 3 if num_regions == 9 else 4\n    spatial_sources: list[int] = []')
    new=replace_once(new,'divmod(query, 4)','divmod(query, side)')
    new=replace_once(new,'0 <= neighbor_row < 4 and 0 <= neighbor_col < 4','0 <= neighbor_row < side and 0 <= neighbor_col < side')
    new=replace_once(new,'neighbor_row * 4 + neighbor_col','neighbor_row * side + neighbor_col')
    new=replace_once(new,'torch.ones(48,','torch.ones(len(spatial_sources),')
    new=new.replace('fixed-budget K=16','fixed K9/K16').replace('in the 4x4 grid','in the selected square grid')
    new=new.replace('shape (B, 16, D)','shape (B, K, D)').replace('requires the approved 4x4 grid','requires K9 or K16')
    result['models/graph_construction.py']=(text[:start]+new+text[end:]).encode()
    text=result['models/gavit.py'].decode()
    text=replace_once(text,'edge_type == "sparse_hybrid" and num_regions != 16','edge_type == "sparse_hybrid" and num_regions not in (9, 16)')
    result['models/gavit.py']=text.encode()
    text=result['experiment_identity.py'].decode()
    text=replace_once(text,'def graph_topology_identity(edge_type: str, *, knn_k: int) -> dict:',
                      'def graph_topology_identity(edge_type: str, *, knn_k: int, num_regions: int = 16) -> dict:')
    start=text.index('    if edge_type == "sparse_hybrid":')
    end=text.index('    if edge_type == "knn":',start)
    block='''    if edge_type == "sparse_hybrid":
        if type(num_regions) is not int or num_regions not in (9,16):
            raise ValueError('Expected K9 or K16')
        side=3 if num_regions==9 else 4
        spatial=4*side*(side-1)
        return dict(name='sparse_hybrid_4n_top2' if num_regions==16 else 'sparse_hybrid_grid3_4n_top2',
                    spatial_connectivity=4,feature_k=2,spatial_directed_edges=spatial,
                    feature_directed_edges=2*num_regions,total_directed_edges=spatial+2*num_regions,**shared)
'''
    result['experiment_identity.py']=(text[:start]+block+text[end:]+IDENTITY_SHIM).encode()
    text=result['train_bigearth.py'].decode()
    text=replace_once(text,'graph_topology_identity(args.edge_type, knn_k=args.knn_k)',
                      'graph_topology_identity(args.edge_type, knn_k=args.knn_k, num_regions=args.num_regions)')
    text=replace_once(text,'for epoch in range(start_epoch, args.epochs + 1):',
                      'first_step_logged = False\nfor epoch in range(start_epoch, args.epochs + 1):')
    text=replace_once(text,'        optimizer.step()','''        optimizer.step()
        if not first_step_logged:
            print(f'TRAIN_FIRST_BATCH_OK K={args.num_regions} device={DEVICE} loss={float(loss.detach()):.6f}',flush=True)
            first_step_logged = True''')
    # IO-only historical host-memory fix, shared by K9 and K16 validation.
    text=replace_once(text,'val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,\n                          num_workers=4, pin_memory=True)',
                      'val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,\n                          num_workers=4, pin_memory=False)')
    result['train_bigearth.py']=text.encode()
    text=result['test_bigearth.py'].decode()
    text=replace_once(text,'num_workers=4, pin_memory=True)','num_workers=4, pin_memory=False)')
    result['test_bigearth.py']=text.encode()
    return result


def export_snapshot(repo,target):
    repo,target=Path(repo),Path(target)
    original={name:subprocess.check_output(['git','-C',str(repo),'show',BASE+':'+name]) for name in FILES}
    patched=patched_sources(original)
    for name,data in patched.items():
        compile(data,name,'exec')
    target.mkdir(parents=True,exist_ok=False)
    for name,data in patched.items():
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    manifest=dict(base_commit=BASE,patch_sha256=digest(__file__),
                  original_files={k:hashlib.sha256(v).hexdigest() for k,v in original.items()},
                  files={k:hashlib.sha256(v).hexdigest() for k,v in patched.items()})
    manifest['snapshot_sha256']=hashlib.sha256(canonical(manifest)).hexdigest()
    (target/'snapshot.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (target/'legacy_graph.py').write_bytes(original['models/graph_construction.py'])
    return manifest


def verify_snapshot(root,manifest):
    root=Path(root)
    value=dict(manifest);expected=value.pop('snapshot_sha256')
    if hashlib.sha256(canonical(value)).hexdigest()!=expected:
        raise RuntimeError('Snapshot manifest digest mismatch')
    if json.loads((root/'snapshot.json').read_text())!=manifest:
        raise RuntimeError('Snapshot manifest changed')
    if any(digest(root/name)!=sha for name,sha in manifest['files'].items()):
        raise RuntimeError('Snapshot source changed')
    if digest(root/'legacy_graph.py')!=manifest['original_files']['models/graph_construction.py']:
        raise RuntimeError('Legacy graph reference changed')
