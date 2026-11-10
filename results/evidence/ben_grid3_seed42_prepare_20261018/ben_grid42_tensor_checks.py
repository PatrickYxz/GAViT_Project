"""Real torch/model gates; absence of torch is an error, never a skipped pass."""
import importlib.util
import math
from pathlib import Path


def check_logit_difference(value):
    # Repeated CUDA scatter reductions may differ in summation order.
    if not math.isfinite(value) or not 0<=value<=1e-5:
        raise RuntimeError('K16 forward comparison exceeds finite 1e-5 tolerance')


def verify_edges(features,edge_index,edge_weight,batch,k):
    import torch
    side=3 if k==9 else 4
    spatial=24 if k==9 else 48
    total=42 if k==9 else 80
    size=features.shape[0]
    assert tuple(edge_index.shape)==(2,size*total)
    assert tuple(edge_weight.shape)==(size*total,) and torch.isfinite(edge_weight).all()
    assert batch.tolist()==sum(([i]*k for i in range(size)),[])
    for i in range(size):
        pairs=edge_index[:,i*total:(i+1)*total].t().cpu().tolist()
        pairs=[(s-i*k,t-i*k) for s,t in pairs]
        assert len(set(pairs))==total and all(0<=s<k and 0<=t<k and s!=t for s,t in pairs)
        distance=lambda s,t:abs(s//side-t//side)+abs(s%side-t%side)
        assert all(distance(s,t)==1 for s,t in pairs[:spatial])
        assert all(distance(s,t)>1 for s,t in pairs[spatial:])
        assert all(sum(t==q for s,t in pairs[spatial:])==2 for q in range(k))


def install_capture(k):
    from models import gavit
    original=gavit.build_sparse_hybrid_graph
    result={}
    def capture(features,*args,**kwargs):
        outputs=original(features,*args,**kwargs)
        if not result:
            assert features.shape[1]==k
            verify_edges(features,*outputs,k)
            total=42 if k==9 else 80
            result.update(actual_forward_checked=True,num_regions=k,batch_size=features.shape[0],
                          spatial_edges=24 if k==9 else 48,feature_edges=2*k,edges_per_image=total,
                          first_image_edge_index=outputs[0][:,:total].detach().cpu().tolist(),
                          scope='first actual model forward in this worker')
        return outputs
    gavit.build_sparse_hybrid_graph=capture
    return result


def run_checks(code):
    import torch
    from models.graph_construction import build_sparse_hybrid_graph
    from models import gavit
    from models.gavit import GAViT
    code=Path(code)
    spec=importlib.util.spec_from_file_location('frozen_cd39_graph_reference',code/'legacy_graph.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    checks=[]
    for k in (9,16):
        # Orthogonal rows force cosine ties; the expected sources are hand checked.
        x=torch.eye(k).unsqueeze(0).repeat(2,1,1)
        outputs=build_sparse_hybrid_graph(x)
        verify_edges(x,*outputs,k)
        start=24 if k==9 else 48
        selected=outputs[0][0,start:(42 if k==9 else 80)][outputs[0][1,start:(42 if k==9 else 80)]==0]
        assert selected.tolist()==[2,4] if k==9 else selected.tolist()==[2,3]
        checks.append('K'+str(k)+'_budget_direction_batch_ties')
    # Independent high-cosine, non-spatial candidates for query0 on a 3x3 grid.
    x=torch.full((1,9,2),-1.)
    x[0,0]=torch.tensor([1.,0.]);x[0,5]=torch.tensor([1.,0.]);x[0,8]=torch.tensor([.8,.6])
    edges,_,_=build_sparse_hybrid_graph(x)
    assert edges[0,24:][edges[1,24:]==0].tolist()==[5,8]
    checks.append('K9_feature_cosine_selection')
    torch.manual_seed(42)
    for x in (torch.randn(2,16,17),torch.ones(2,16,7),torch.eye(16).unsqueeze(0)):
        before=old.build_sparse_hybrid_graph(x);after=build_sparse_hybrid_graph(x)
        assert all(torch.equal(a,b) for a,b in zip(before,after))
    checks.append('K16_exact_graph_tensor_equivalence_to_cd39')
    model=GAViT(num_classes=19,num_regions=16,grouping='attentive_spatial',edge_type='sparse_hybrid',
                integration='token_feedback',pretrained=False).to('cuda').eval()
    x=torch.randn(2,3,224,224,device='cuda')
    with torch.no_grad():
        patched=model(x)
        current=gavit.build_sparse_hybrid_graph
        try:
            gavit.build_sparse_hybrid_graph=old.build_sparse_hybrid_graph
            legacy=model(x)
        finally:
            gavit.build_sparse_hybrid_graph=current
    diff=float((patched-legacy).abs().max())
    check_logit_difference(diff)
    assert torch.isfinite(patched).all()
    checks.append('K16_full_forward_equivalence_to_cd39_graph_within_1e-5')
    del model
    captured=install_capture(9)
    model=GAViT(num_classes=19,num_regions=9,grouping='attentive_spatial',edge_type='sparse_hybrid',
                integration='token_feedback',pretrained=False).to('cuda').eval()
    with torch.no_grad():
        logits=model(x)
    assert tuple(logits.shape)==(2,19) and torch.isfinite(logits).all() and captured['actual_forward_checked']
    assert sum(p.numel() for p in model.parameters())==31366926
    checks.append('K9_actual_model_forward_and_parameters')
    del model
    torch.cuda.synchronize()
    return dict(status='pass',checks=checks,k16_max_logit_diff=diff,k16_logit_tolerance=1e-5,k9_capture=captured,
                pretrained=False,scope='graph tensors and random-initialized full forwards; training/reload follow in independent smoke stage')
