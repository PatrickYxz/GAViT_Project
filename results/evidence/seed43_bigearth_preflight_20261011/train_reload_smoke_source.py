import os,sys,json,time,atexit,hashlib,subprocess,runpy,resource; from pathlib import Path; repo=Path("/home/featurize/work/GAViT_SOFT_20260927"); out=Path("/home/featurize/work/gpucheck_bigearth_soft43_4090_20261011_01"); h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); info=json.loads((out/"preflight.json").read_text()); assert info["status"]=="ready"; assert subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()==info["commit"]; assert not subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip(); assert all(h(out/"splits"/(s+".csv"))==v for s,v in info["subset_sha256"].items()); assert h(Path("/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors"))==info["pretrained_sha256"]; (out/"run.started").open("x").close(); status={"exit_code":1}; atexit.register(lambda:(out/"run.exit").write_text("exit_code="+str(status["exit_code"])+"\n")); os.chdir(repo); sys.path.insert(0,str(repo)); import torch; assert torch.cuda.is_available(),"CUDA unavailable"; torch.set_num_threads(8); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter(); print("DEVICE",torch.cuda.get_device_name(0),"torch",torch.__version__,"CUDA",torch.version.cuda,"matmul_tf32",torch.backends.cuda.matmul.allow_tf32,"cudnn_tf32",torch.backends.cudnn.allow_tf32,flush=True); sys.argv=[str(repo/"train_bigearth.py")]+ ["--model","gavit","--grouping","soft_prototype","--edge_type","sparse_hybrid","--integration","token_feedback","--num_regions","16","--knn_k","5","--gat_hidden","256","--gat_heads","4","--gat_layers","2","--dropout","0.1","--seed","43","--epochs","1","--batch_size","32","--lr","0.0003","--run_stage","smoke","--run_tag","soft_prototype_seed43_4090_20261011_01","--data_dir","/home/featurize/work/gpucheck_bigearth_soft43_4090_20261011_01/splits","--pretrained_path","/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors","--checkpoint_path","/home/featurize/work/gpucheck_bigearth_soft43_4090_20261011_01/gavit.pth"]; ns=runpy.run_path(str(repo/"train_bigearth.py"),run_name="__main__"); torch.cuda.synchronize(); train_wall=time.perf_counter()-start; assert ns["DEVICE"]=="cuda" and len(ns["train_set"])==96 and len(ns["val_set"])==64; assert ns["epoch"]==1 and ns["probe"].last_epoch==1; from models.gavit import GAViT; from scene_classification.modeling import capture_graph_once; from utils import load_checkpoint_metadata; meta=load_checkpoint_metadata(str(out/"gavit.pth")); assert h(out/"gavit.pth")==meta["checkpoint_sha256"] and meta["best"]["epoch"]==1; m=ns["model"].eval(); x=ns["probe"].inputs[:2].to("cuda"); fresh=GAViT(**meta["architecture"],pretrained=False).to("cuda").eval(); fresh.load_state_dict(torch.load(out/"gavit.pth",map_location="cpu",weights_only=True),strict=True); graph,hook=capture_graph_once(fresh); torch.set_grad_enabled(False); delta=float((m(x)-fresh(x)).abs().max()); hook.remove(); assert delta<=1e-5 and graph.get("actual_forward_checked"),(delta,graph); probe=out/"gavit.assignment_probe"; assert all((probe/n).is_file() for n in ("epoch_000.pt","epoch_001.pt","best.pt","final.pt")); assert (out/"gavit.last.train_state.pth").is_file(); torch.cuda.synchronize(); summary={"status":"training_and_reload_complete","counts":info["counts"],"device":torch.cuda.get_device_name(0),"epochs":1,"reload_max_logit_diff":delta,"graph":graph,"best":meta["best"],"checkpoint_sha256":meta["checkpoint_sha256"],"training_entry_seconds":train_wall,"total_seconds_including_reload":time.perf_counter()-start,"cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(),"cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(),"main_process_max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"precision":{"amp":False,"matmul_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_tf32":torch.backends.cudnn.allow_tf32},"probe_summary":ns["probe"].summary()}; (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); 

import math
assert meta['architecture']==helper_ns['ARCH']
expected_training={**helper_ns['TRAINING'],'epochs':1}
assert meta['training']==expected_training and meta['execution']['run_stage']=='smoke'
assert meta['git']=={'commit':'ed480e6','dirty':False}
assert ns['total_params']==31466766
assert all(math.isfinite(ns[k]) for k in ('avg_loss','val_mAP','val_f1'))
state=torch.load(out/'gavit.last.train_state.pth',map_location='cpu',weights_only=True)
assert state['epoch']==1
del state
graph_probe=helper_ns['check_probe'](out/'gavit.assignment_probe',torch,
    ns['model'].region_grouping.candidate_mask.cpu(),ns['val_patches'][:8],1,final_epoch=1)
hardware_after,precision_after=helper_ns['environment'](torch,b)
assert hardware_after==ready['hardware'] and precision_after==ready['precision']
summary.update(seed=43,commit=helper_ns['COMMIT'],hardware=hardware_after,precision=precision_after,
    metadata=meta,parameters=ns['total_params'],train_loss=ns['avg_loss'],val_mAP_percent=ns['val_mAP'],
    val_macro_f1_percent=ns['val_f1'],graph_probe=graph_probe,
    source_preflight='approved_preflight.json',resource_scope='whole smoke including reload and artifact checks',
    status='complete')
torch.cuda.synchronize()
summary['total_seconds_including_reload']=time.perf_counter()-start
summary['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
summary['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
status['exit_code']=0
print('BIGEARTH_SOFT43_SMOKE_COMPLETE',json.dumps(summary),flush=True)
