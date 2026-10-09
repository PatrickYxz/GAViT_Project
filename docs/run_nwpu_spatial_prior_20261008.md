# NWPU 单候选：保留 attention，给 S 加固定空间先验

本分支是空间先验单候选的独立发布版本，服务器尚待一轮短测。发布分支：`codex/soft-spatial-prior-nwpu-20261008`。先按发布消息中固定完整 commit 的 Git 命令同步，再执行下文；不要在旧服务器目录手工修改模型。

## 已冻结的研究问题

在原 soft GAViT 的内容相似度中加入与原网格一致的空间偏好，能否改善完整验证集分类，同时减少不必要的跨区归属？原 pooling 干预的 OA 从 93.730159% 降至 93.174603%，所以保留 attention 汇总。该结果没有证明空间距离就是失败根因，也没有证明新候选一定有益。

只使用 `--spatial-prior anchor_gaussian_v1` 一个候选，不扫描强度。定义：

```text
x_i = token 的行、列索引（0..6）
c_k = 原硬分区内 token 坐标的无权平均
      单轴中心：[0.5, 2.5, 4.5, 6.0]
d²_ik = ||(x_i - c_k) / 1.75||²
logit_ik = cosine(z_i, prototype_k) / 0.1 - 0.5 * d²_ik
S_i = softmax_region(mask_radius_1(logit_i))
W_ik = softmax_token(log(S_ik) + attention_i)
```

距离项放在温度缩放之后，单位为 logit；一名义区域宽度的惩罚为 0.5，相应空间因子 exp(-0.5)≈0.607。实际区域不是等大的，故使用现有硬分区的质心，而非理想均匀网格中心。所有位置固定，不学习中心，不迭代更新，不加新损失或日程。

其余维度保持：16 节点、128 维投影、温度 0.1、半径 1、原型 attention、最终 attention、48 条空间边 + 32 条特征边、GAT、feedback、分类头、优化器、数据增强和预训练来源。新版本不增加学习参数。S 改变会连带改变 W 数值、区域表示和特征边；不能称后者数值不变。

旧默认 `local_prototype_cosine_v1` 不变，新版本为 `local_prototype_cosine_spatial_v1`，完整先验字段写入 architecture 和 probe。新版本独有持久 buffer `spatial_log_prior`（49×16 常量），strict load 会拒绝新旧 checkpoint 混用；旧 checkpoint key 不变。

## 范围、对照与停止条件

- 当前只交付 NWPU seed42 **一轮工程短测**：每类 train8/val2，共 360/90。完整校准须先收到短测成功、资源和重载证据，不与短测自动串行。
- 后续拟定校准：5040 train / 1260 val，30 epoch，batch32、workers2、AdamW lr3e-4/wd1e-4、cosine T_max30、FP32、无 AMP、两种 TF32 均 false。从原 ImageNet 权重重新初始化，不接旧 soft best。
- 主指标为完整 val OA；同时记录 macro-F1、逐类准确率、混淆矩阵、逐图预测及全部类别概率、best/final/曲线、吞吐和显存。与已完成旧 soft 和 Swin 的相同 val 配对比较，不读 test 选配置。
- 若校准 best OA ≤ 旧 soft 的 1181/1260（93.730159%），结束该候选，不追加 seeds 或换强度重跑。若仅优于旧 soft 但没有超过 Swin 的 1187/1260（94.206349%），最多作为局部消融证据，不宣称 baseline 提升。
- 超过 Swin 时仍需核对 macro-F1（参考 94.171552%）、逐类代价、配对错误和曲线；一轮 seed42 的小幅差值不等于稳定性，不自动启动多 seeds。是否保留为最终候选另行依据完整证据决定。
- S/W 几何统计仅为辅助解释；不以漂移变小或熵变小替代分类指标，也不把预期方向当工程必过条件。当前不运行 BigEarthNet、refit、test、FFN、额外 seed。

当前 [结果状态表](../results/nwpu_spatial_prior_trial.md) 和 [可审查代码补丁](../results/evidence/nwpu_spatial_prior_20261008/spatial_prior_code.patch)。补丁基于已发布 `058f1d8b79d0be538ec358eb742d2cd3992921d4`，该版本中的 FFN 保持关闭。5 个运行文件 + 2 个测试文件；其余17个共享运行源文件与基底一致。`release_identity.json` 保留10月8日未发布时的代码哈希快照，后续发布核验另见 `release_validation.json`。

## 服务器操作

1. 使用发布消息中的完整 commit 同步到 `/home/featurize/work/GAViT_SPATIAL_20261008`。已有目录时先检查，不强行覆盖或删除。
2. 在服务器复制下一整行：先检查已获取分支的 HEAD、工作区无改动、新输出不存在、单张2080Ti空闲且至少8GiB可用、全部图片路径存在，然后运行一轮前台短测。检查失败不会进入训练。若GPU已变，请回传信息，不修改检查来强行启动。

```bash
python -c 'import csv,io,json,os,subprocess; from pathlib import Path; p=Path("/home/featurize/work/GAViT_SPATIAL_20261008"); o=Path("/home/featurize/work/nwpu_spatial42_smoke_20261008_01"); g=lambda *a: subprocess.check_output(["git","-C",str(p),*a],text=True).strip(); assert g("rev-parse","HEAD")==g("rev-parse","origin/codex/soft-spatial-prior-nwpu-20261008"),"Release HEAD differs from fetched branch"; assert not g("status","--porcelain","--untracked-files=all"),"Source has local modifications"; assert not o.exists(),"Output exists; preserve it and inspect before retrying"; assert os.environ.get("CUDA_VISIBLE_DEVICES") in (None,"0"),"Unexpected CUDA_VISIBLE_DEVICES; inspect before running"; t=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True); rows=list(csv.reader(io.StringIO(t))); assert len(rows)==1 and "2080 Ti" in rows[0][0],"Expected one RTX 2080 Ti; report current GPU before running"; assert int(rows[0][1])>=8192 and int(rows[0][2])<=5,"GPU is busy or has less than 8 GiB free"; assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip(),"Existing GPU process; do not stop or overlap it"; m=json.loads(Path("/home/featurize/work/nwpu_protocol_20260927_02.json").read_text()); d=Path("/home/featurize/data/NWPU-RESISC45"); missing=[r["path"] for r in m["records"] if not (d/r["path"]).is_file()]; assert not missing,"NWPU images missing; restore the verified ZIP first: "+repr(missing[:3]); print("PREFLIGHT PASS",g("rev-parse","HEAD"),rows[0],"image paths",len(m["records"]),flush=True)' && env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 python -u /home/featurize/work/GAViT_SPATIAL_20261008/run_scene.py train --model gavit --grouping soft_prototype --spatial-prior anchor_gaussian_v1 --phase smoke --seed 42 --train-per-class 8 --val-per-class 2 --manifest /home/featurize/work/nwpu_protocol_20260927_02.json --data-root /home/featurize/data/NWPU-RESISC45 --output /home/featurize/work/nwpu_spatial42_smoke_20261008_01 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --device cuda --batch-size 32 --workers 2
```

入口会完整解码并检查 **31,500 张图片**的文件/像素身份，再选择360train/90val；不只是扫描450张。manifest SHA为 `ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`，ImageNet预训练 SHA为 `f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`，入口核对。线程固定8；无AMP，双TF32关闭。输出独占创建，已存在时先审阅，不自动重试。

仅预检报原图缺失时，验证持久ZIP并补解压，再执行上述启动行；不重建manifest。ZIP损坏时不会解压，已有文件由 `unzip -n` 保留。

```bash
printf '%s\n' 'beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53  /home/featurize/work/NWPU-RESISC45_a826b44.zip' | sha256sum -c - && unzip -n -q /home/featurize/work/NWPU-RESISC45_a826b44.zip -d /home/featurize/data
```

3. 启动成功需要模型/设备、`TRAIN FIRST BATCH OK`、实际进程和GPU活动，可在另一终端查看 `nvidia-smi`。完成需要 `SCENE SMOKE COMPLETE`；前台运行不要按Ctrl+C中断。结束后复制下面一行，回传简短摘要（不展开混淆矩阵）：

```bash
python -c 'import json,hashlib; from pathlib import Path; p=Path("/home/featurize/work/nwpu_spatial42_smoke_20261008_01"); r=json.loads((p/"run.json").read_text()); exit_text=(p/"run.exit").read_text().strip(); assert exit_text=="exit_code=0" and r["status"]=="complete","Run is incomplete or failed"; assert r["phase"]=="smoke" and r["seed"]==42 and r["counts"]=={"train":360,"val":90} and r["final_epoch"]==1,"Scope mismatch"; assert r["architecture"]["soft_assignment"]["spatial_prior"]=="anchor_gaussian_v1" and "token_refinement" not in r["architecture"],"Candidate mismatch"; assert r["reload_max_logit_diff"]<=1e-5 and r["graph_capture"]=={"actual_forward_checked":True,"edges_per_image":80,"feature_edges":32,"spatial_edges":48},"Reload or graph mismatch"; assert r["hardware"]["torch_num_threads"]==8 and r["hardware"]["device"]=="cuda","Device or thread mismatch"; assert r["code"]["git"]["dirty"] is False,"Dirty source"; digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest(); assert r["checkpoint_sha256"]==digest(p/"best.pth"),"Best checkpoint hash mismatch"; metas={n:json.loads((p/(n+".meta.json")).read_text()) for n in ("best","last")}; assert all(v["checkpoint_sha256"]==digest(p/(n+".pth")) and v["architecture"]==r["architecture"] for n,v in metas.items()),"Checkpoint sidecar mismatch"; required=("last.train_state.pth","config.json","best.meta.json","last.meta.json","history.json","selected_samples.json","validation_metrics.json","validation_predictions.csv","validation_per_class.csv","assignment_probe/probe_manifest.json","assignment_probe/assignment_diagnostics.jsonl","assignment_probe/epoch_000.pt","assignment_probe/epoch_001.pt","assignment_probe/best.pt","assignment_probe/final.pt"); assert all((p/f).is_file() and (p/f).stat().st_size>0 for f in required),"Missing artifact"; keys=("status","phase","seed","counts","final_epoch","reload_max_logit_diff","graph_capture","assignment_probe","elapsed_seconds","cuda_peak_allocated_bytes","cuda_peak_reserved_bytes","checkpoint_sha256","manifest_sha256","pretrained_sha256"); print(json.dumps({"exit":exit_text,**{k:r.get(k) for k in keys},"training_state_sha256_observed":digest(p/"last.train_state.pth"),"git":r["code"]["git"],"hardware":r["hardware"],"history": [{k:v for k,v in e.items() if k in ("epoch","train_loss","train_seconds","train_images_per_second","validation_seconds")} for e in json.loads((p/"history.json").read_text())]},indent=2))'
```

摘要同时核对退出、样本数/轮数、候选身份、图边、重载、源代码干净状态、线程、实际best/last checkpoint与sidecar SHA、所需产物存在性，并计算训练状态文件SHA（无既存摘要可供比对）。它不逐元素检查probe快照，不提供单参数梯度审计；训练入口每batch检查梯度有限性。回传后再检查probe及源码/输入身份是否与发布一致，工程短测指标不用于挑选模型。此命令不会自动进入完整校准、refit或test。
