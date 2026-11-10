# NWPU短测剩余张量与CUDA梯度检查

2026-09-28。_02的训练/重载及JSON诊断已返回。两条命令分别读取已保存快照、用同一checkpoint和同一8图探针中的前2图检查分配专属梯度；不更新参数，不写训练输出，不重新生成划分。

这些命令先在本地真实Swin/PyG、合成8图和随机权重上做CPU验证：有效快照/梯度通过；非归一化S和人为断开S到模型的梯度都被拒绝。该验证只覆盖命令接口/断言，不代表服务器快照、预训练权重或CUDA已通过。原始验证结果和本地验证脚本见`results/evidence/nwpu_soft_smoke42_20260927_02/final_checks_local_validation.json`及同目录脚本。

## 1. 检查缓存快照

核对epoch0/1与best/final身份、形状、有限非负值，S按行/W按列归一化，候选区域外严格为0，best/final与epoch1文件一致，以及张量实际变化与诊断JSON相符。最低区域质量只作为数值输出，不等同于语义纯净或总体无塌缩证明。

```bash
python -c 'import sys,json,hashlib; from pathlib import Path; sys.path.insert(0,"/home/featurize/work/GAViT_SOFT_20260927"); import torch; p=Path("/home/featurize/work/nwpu_soft_smoke42_20260927_02"); q=p/"assignment_probe"; ident=json.loads((q/"probe_manifest.json").read_text()); assert ident["split"]=="val" and len(ident["sample_ids"])==8; from models.soft_region_grouping import SoftPrototypeGrouping; names=("epoch_000.pt","epoch_001.pt","best.pt","final.pt"); a=[torch.load(q/n,map_location="cpu",weights_only=True) for n in names]; assert [v["epoch"] for v in a]==[0,1,1,1] and all(v["identity"]==ident for v in a); mask=SoftPrototypeGrouping(soft_assignment=ident["soft_assignment"]).candidate_mask; assert all(tuple(v["tensors"][k].shape)==(8,49,16) and bool(torch.isfinite(v["tensors"][k]).all()) and bool((v["tensors"][k]>=0).all()) for v in a for k in ("S","W")); errors={"S_row":max(float((v["tensors"]["S"].sum(-1)-1).abs().max()) for v in a),"W_column":max(float((v["tensors"]["W"].sum(1)-1).abs().max()) for v in a)}; assert max(errors.values())<1e-5,errors; assert all(int(torch.count_nonzero(v["tensors"][k][:,~mask]))==0 for v in a for k in ("S","W")); assert len({hashlib.sha256((q/n).read_bytes()).hexdigest() for n in names[1:]})==1; d=[json.loads(s) for s in (q/"assignment_diagnostics.jsonl").read_text().splitlines()]; assert [r["epoch"] for r in d]==[0,1] and all(r["sample_ids"]==ident["sample_ids"] for r in d); delta=float((a[1]["tensors"]["S"]-a[0]["tensors"]["S"]).abs().mean()); assert abs(delta-d[1]["mean_abs_delta_S"])<1e-6; print("SNAPSHOT_CHECK_PASS",json.dumps({**errors,"delta_S":delta,"min_region_mass":min(float(v["tensors"]["S"].sum(1).min()) for v in a)}))'
```

应输出`SNAPSHOT_CHECK_PASS`。失败时保留报错，停止后续依赖步骤。

## 2. 检查真实CUDA上的分配梯度

核对现有best权重SHA256，严格加载原配置；重新读取原8张val探针并校验预处理输入SHA256。只用前2张做一次backbone→S的前向及随机线性S目标反向；投影、原区域注意力、骨干三组梯度必须有限且非零，GAT/feedback/classifier不应接收梯度。没有optimizer或保存调用；这一计算验证可微路径，不衡量分类收益或等同于正式训练收敛。

```bash
python -c 'import sys,json,hashlib; from pathlib import Path; sys.path.insert(0,"/home/featurize/work/GAViT_SOFT_20260927"); import torch; p=Path("/home/featurize/work/nwpu_soft_smoke42_20260927_02"); q=p/"assignment_probe"; ident=json.loads((q/"probe_manifest.json").read_text()); assert ident["split"]=="val" and len(ident["sample_ids"])==8; import math; from PIL import Image; from scene_classification.modeling import build_model,image_transform; assert torch.cuda.is_available(); device=torch.device("cuda"); torch.set_num_threads(8); torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; r=json.loads((p/"run.json").read_text()); assert r["status"]=="complete" and r["architecture"]["grouping"]=="soft_prototype"; assert hashlib.sha256((p/"best.pth").read_bytes()).hexdigest()==r["checkpoint_sha256"]; model=build_model("gavit",45,architecture_config=r["architecture"]).to(device).eval(); model.load_state_dict(torch.load(p/"best.pth",map_location="cpu",weights_only=True),strict=True); tf=image_transform(False); x=torch.stack([tf(Image.open(Path("/home/featurize/data/NWPU-RESISC45")/s).convert("RGB")) for s in ident["sample_ids"]]); assert hashlib.sha256(x.numpy().tobytes()).hexdigest()==ident["input_sha256"]; s=model.region_grouping(model.backbone(x[:2].to(device)))[1]; (s*torch.randn_like(s)).sum().backward(); modules={"projection":model.region_grouping.assignment_projection,"attention":model.region_grouping.attn,"backbone":model.backbone}; g={n:sum(float(t.grad.abs().sum()) for t in m.parameters() if t.grad is not None) for n,m in modules.items()}; assert all(math.isfinite(v) and v>0 for v in g.values()),g; assert all(t.grad is None for m in (model.graph_reasoning,model.feedback_proj,model.classifier) for t in m.parameters()); print("ASSIGNMENT_GRAD_PASS",json.dumps({"device":str(device),"gradient_abs_sum":g}))'
```

应输出`ASSIGNMENT_GRAD_PASS`且device为cuda。若原始图片因实例重启消失，输入读取会失败；保留结果，不能替换为随机输入后称真实探针验证成功。

2026-09-28两项已回传通过：归一化最大误差2.98e-7，真实CUDA分配专属梯度三组有限非零；详见[验收记录](../results/nwpu_soft_smoke_validation.md)。本页命令保留为证据，无需重复。NWPU预定工程检查收尾；下一步[检查BigEarthNet96/64原图](check_bigearth_soft_data.md)，其短测及正式三模型/多seed仍未完成。
