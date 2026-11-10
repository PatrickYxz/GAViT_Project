# BigEarthNet soft_prototype GPU短测 — 2026-09-28

运行`/home/featurize/work/bigearth_soft_smoke42_20260928_01`返回退出码0（用户粘贴首行省略了首字母e）、`training_and_reload_complete`。这是用户终端证据，未直接连接服务器或下载checkpoint。[回传摘要](evidence/bigearth_soft_preflight_20260928/server_summary_user_return.json)、[本地数值核对](evidence/bigearth_soft_preflight_20260928/server_validation.json)与[已交付完整命令](../docs/run_bigearth_soft_smoke_20260928.md)共同确定验收范围。

| 项目 | 返回结果 |
|---|---|
| GPU / 数据 | RTX2080 Ti；96train/64val、19类、1轮、seed42 |
| 入口及配方 | 发布ed480e6；原train_bigearth、soft_prototype、sparse_hybrid、token_feedback；BCE、batch32、workers4、train pin=True/val=False、AdamW3e-4/wd1e-4、Cosine T_max1；身份来自启动包装器的通过断言，完整preflight/meta尚未单独回传 |
| 实际图边 | actual_forward_checked=true，48空间+32特征=80条/图 |
| 重建最大logit差 | 4.76837158203125e-7，低于1e-5；相同2张无增强val探针 |
| best epoch / 小样本val mAP | 1 / 37.0547%；仅工程短测，入口对无正例类别跳过AP，不代表完整19类正式结果 |
| checkpoint SHA256（报告值） | fbde01de1bea308b981623500274bdfea9799a7582abbd42c12688498944daef |
| 入口墙钟 / 含重建总墙钟 | 24.338881547 / 26.110388833秒；入口墙钟包含初始化、probe、验证和存盘，不是纯训练时间 |
| CUDA峰值allocated / reserved | 3929639936 / 4334813184 bytes = 3.6598 / 4.0371 GiB，统计包含重建 |
| 主进程RSS峰值 | 1865384 KiB = 1.7790 GiB，不包含全部worker内存 |
| 精度 | FP32无AMP；matmul TF32 false、cuDNN TF32 true，沿用BigEarth入口；不同于NWPU两项TF32均false，不当成同精度效率对比 |
| 探针 | 固定8val、last epoch1；累计1.713090071秒；四快照和last state存在由包装器检查 |

## 结论与工程收尾

新分组在真实BigEarth原RGB、19标签BCE和CUDA多worker下的训练/验证/保存/重建及图边路径已获回传证据；与此前NWPU的CE和CUDA分配专属梯度检查合并，证明两数据集入口可运行。不能据单轮37.05%评价效果或外推正式训练费用。

**09-28工程收尾已回传通过**：[结果摘录](evidence/bigearth_soft_preflight_20260928/final_artifact_user_return.json)。用户回传的是所交付命令所有断言之后的JSON，未包含前置PASS行；据命令执行顺序确认快照身份/非负有限/mask/别名/JSON一致性以及ZIP与480成员逐文件哈希均走过检查。S/W最大归一化误差均2.3841858e-7。备份9620902 bytes（9.62 MB/9.175 MiB），位于运行目录rgb_subset.zip，仅恢复短测原图。

精确loss0.5011217395、val mAP37.0546909675%、macro-F1 2.6737323859%，epoch wall（排除probe、包含train/val/保存）7.815954543秒。train缺类16海岸湿地正例；val缺类14沙滩和类16海岸湿地正例，因此本次mAP只平均17个有正例的类，macro-F1仍按19列计算。保留这些短测原值，不改阈值/挑样本/据其评价正式模型。

固定8val探针epoch0→1：熵0.772952→1.110026 nats，S平均绝对变化0→0.05066226，低范数原型均0，最小范数0.522190→0.596006，反馈/token比均值1.364840→1.495322。统计有限，无这组检查覆盖到的数值异常；不据原型范数、S变化或反馈量级证明语义无塌缩或收益。

NWPU哨兵图存在性false，仅说明该文件不在预期路径，不能断言全部31500张都缺失。下一步用已保存ZIP恢复NWPU，再按[三模型seed42校准](../docs/run_nwpu_ablation42_20260928.md)执行冻结协议。以下只读检查命令保留供追溯，无需重复。

```bash
python -c 'import sys,json,hashlib,zipfile,math; from pathlib import Path; sys.path.insert(0,"/home/featurize/work/GAViT_SOFT_20260927"); import torch; from models.soft_region_grouping import SoftPrototypeGrouping; p=Path("/home/featurize/work/bigearth_soft_smoke42_20260928_01"); q=p/"gavit.assignment_probe"; h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest(); assert (p/"run.exit").read_text().strip()=="exit_code=0"; info=json.loads((p/"preflight.json").read_text()); meta=json.loads((p/"gavit.meta.json").read_text()); ident=json.loads((q/"probe_manifest.json").read_text()); assert meta["architecture"]["grouping"]=="soft_prototype" and ident["split"]=="val" and len(ident["sample_ids"])==8; names=("epoch_000.pt","epoch_001.pt","best.pt","final.pt"); a=[torch.load(q/n,map_location="cpu",weights_only=True) for n in names]; assert [v["epoch"] for v in a]==[0,1,1,1] and all(v["identity"]==ident for v in a); mask=SoftPrototypeGrouping(soft_assignment=ident["soft_assignment"]).candidate_mask; assert all(tuple(v["tensors"][k].shape)==(8,49,16) and bool(torch.isfinite(v["tensors"][k]).all()) and bool((v["tensors"][k]>=0).all()) and int(torch.count_nonzero(v["tensors"][k][:,~mask]))==0 for v in a for k in ("S","W")); errors={"S_row":max(float((v["tensors"]["S"].sum(-1)-1).abs().max()) for v in a),"W_column":max(float((v["tensors"]["W"].sum(1)-1).abs().max()) for v in a)}; assert max(errors.values())<1e-5 and len({h(q/n) for n in names[1:]})==1; d=[json.loads(s) for s in (q/"assignment_diagnostics.jsonl").read_text().splitlines()]; assert [r["epoch"] for r in d]==[0,1] and all(r["sample_ids"]==ident["sample_ids"] for r in d); assert abs(float((a[1]["tensors"]["S"]-a[0]["tensors"]["S"]).abs().mean())-d[1]["mean_abs_delta_S"])<1e-6; z=zipfile.ZipFile(p/"rgb_subset.zip"); assert h(p/"rgb_subset.zip")==info["archive_sha256"] and len(z.infolist())==480 and set(z.namelist())==set(info["band_sha256"]); assert all(hashlib.sha256(z.read(n)).hexdigest()==v for n,v in info["band_sha256"].items()); z.close(); log=Path(str(p)+".launch.log").read_text(errors="replace"); epochs=[json.loads(s[len("SOFT EPOCH "):]) for s in log.splitlines() if s.startswith("SOFT EPOCH ")]; assert len(epochs)==1 and epochs[0]["epoch"]==1 and all(math.isfinite(epochs[0][k]) for k in ("train_loss","val_mAP_percent","val_macro_f1_percent")); print("BIGEARTH_SMOKE_ARTIFACTS_PASS"); print(json.dumps({"normalization_errors":errors,"archive_bytes":info["archive_bytes"],"positive_support":info["positive_support"],"epoch":{k:epochs[0][k] for k in ("train_loss","val_mAP_percent","val_macro_f1_percent","epoch_wall_seconds_excluding_probe")},"probe":[{k:r[k] for k in ("epoch","entropy_mean","mean_abs_delta_S","low_norm_prototypes","prototype_norm_min","feedback_to_token_ratio")} for r in d],"nwpu_data_exists":Path("/home/featurize/data/NWPU-RESISC45/airplane/airplane_001.jpg").is_file()},indent=2))'
```

该次工程检查已收尾，按既定同manifest/预训练/seed42协议推进NWPU三模型30轮校准（5040train/1260val；test25200不参与候选选择）。历史NWPU近似成绩不作为新协议baseline；正式候选比较和多seed仍未完成。当前无需改架构、重训smoke或合并PR。
