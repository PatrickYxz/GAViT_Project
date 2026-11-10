# 局部原型软分配：服务器短测交付

2026-09-26建立，09-28更新。本地实现与 CPU 验收见 [验证记录](../results/soft_assignment_local_validation.md)。本文件定义执行顺序和验收条件；新版本已独立发布，服务器仍需按固定提交同步。本文未启动 GPU 任务。

**09-28双数据集工程检查已收尾**：BigEarth最终快照/日志/备份校验回传通过。两轮工程smoke无需重复，后续为[恢复NWPU并启动三模型seed42校准](run_nwpu_ablation42_20260928.md)；目前没有正式效果结果。

**09-28 BigEarth训练/重建回传通过**：RTX2080Ti、96/64、1轮、exit0、48+32图边、重建差4.77e-7。两数据集训练路径已有真实证据；BigEarth缓存探针/备份与日志摘要只读收尾仍待返回，命令见[本次验收](../results/bigearth_soft_smoke_validation.md)。不重训工程短测，后续为NWPU三模型seed42校准。

**09-28 NWPU工程检查已收尾**：快照归一化/mask/身份/别名及原图CUDA分配专属梯度均返回PASS，证据见[工程验收](../results/nwpu_soft_smoke_validation.md)。下一步运行[BigEarthNet160图存在检查](check_bigearth_soft_data.md)，再做96/64的1轮BCE短测。以下检查前记录保留为历史，不要重跑已通过的NWPU步骤。

**09-28探针详情已返回**：纯train3.236秒、111.25图/秒、val0.525秒；8图S变化0.03925、低范数0、四快照存在。下一步直接执行[缓存张量与CUDA梯度两条检查](check_nwpu_soft_final.md)，无需再读取同一摘要或重训。服务器结果仍待回传，BigEarth门槛继续保留。

**09-28_02训练/重载回传成功**：用户运行360train/90val、1轮、RTX2080 Ti/双worker，exit0、complete；实际48+32图边、重载差3.58e-7，20源文件SHA与ed480e6匹配。总131.21秒，峰值allocated/reserved3.66/4.04GiB；val8图probe摘要完成。详见[工程验收及下一条只读命令](../results/nwpu_soft_smoke_validation.md)。无需重跑该短测；诊断张量、分配专属CUDA梯度及BigEarth原图短测仍待后续，不能把此状态扩大为全部GPU门槛通过。

**09-28失败后恢复**：`_01`的旧缺图报错与重试目录碰撞混在追加launch日志中；最新调用在输出目录检查退出，未重新检查图片。保留_01；使用[31500文件路径核对及_02重试](retry_nwpu_soft_smoke_02.md)，协议不变。恢复后的文件检查和GPU短测成功仍待回传。

**09-28已发布**：`codex/soft-assignment-20260927`提交`ed480e60fea03ed786fdba70c144de5e815c34f5`，[草稿PR#1](https://github.com/PatrickYxz/GAViT_Project/pull/1)，base为`codex/aid-entry-20260920`，尚未合并。23文件中18项是已通过133项测试的代码/测试，5项为文档和证据。用户在服务器复制下面一整行完成隔离同步，再按[发布执行说明](../results/soft_assignment_release_validation.md#发布后下一步)运行1轮短测；固定manifest与预训练已在本轮回传确认，无需重建。

```bash
git -C /home/featurize/work/GAViT_AID_20260920 fetch origin codex/soft-assignment-20260927 && git -C /home/featurize/work/GAViT_AID_20260920 worktree add --detach /home/featurize/work/GAViT_SOFT_20260927 ed480e60fea03ed786fdba70c144de5e815c34f5
```

以下09-27条目保留为准备阶段历史，当前发布状态以上方09-28记录为准；所有GPU验收要求继续适用。

**09-27发布候选验收完成**：基于服务器同版`deb2b45`，18个代码/测试文件，候选完整回归133项通过/exit0；[补丁、源码身份、测试与待发布命令](../results/soft_assignment_release_validation.md)已可审阅。拟发布`codex/soft-assignment-20260927`，当前尚未提交/推送，服务器同步命令待真实发布commit确定。

**09-27服务器身份已返回**：RTX2080 Ti，total22528/free22009 MiB（查询时瞬时值）；`codex/aid-entry-20260920`在`deb2b45`、工作区干净。manifest SHA256为`ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`，预训练SHA256与历史`f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`一致。[回传证据](../results/evidence/nwpu_prepare_20260927/server_identity_return.txt)。下面四条只读命令已完成，无需再次运行；下一步整理独立发布版本及1轮短测。

**09-27当前进度**：用户group prepare回传成功，扫描31500图/47.6秒，train5040/val1260/test25200。8组同类重复（train2/val1/test5组）均未跨划分，保留全部文件，共31492个唯一像素图像。[完整回传摘录](../results/evidence/nwpu_prepare_20260927/group_prepare_return.txt)。固定`/home/featurize/work/nwpu_protocol_20260927_02.json`，不再运行下面保留的prepare历史命令。下一步只读回传下列四行的结果，核对GPU、现有服务器代码及清单/预训练身份，再安排独立代码发布和短测；这些命令不训练、不修改文件。

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
git -C /home/featurize/work/GAViT_AID_20260920 status --short --branch
git -C /home/featurize/work/GAViT_AID_20260920 --no-pager log -1 --oneline
sha256sum /home/featurize/work/nwpu_protocol_20260927_02.json /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors
```

## 目的与固定范围

短测回答：原图读取、CUDA 前后向、真实图构造、分配梯度、保存与重载能否完整工作，以及单轮耗时和显存是多少。短测分数不用于证明收益。通过之后才安排 NWPU seed42 的 Swin / 固定4×4 / soft_prototype 三模型校准；BigEarthNet 完整训练和多 seed 仍需据结果与预算安排。

新版本固定为16区域、128维投影、切比雪夫半径1、温度0.1、单次分配、sparse_hybrid、token_feedback；只变更分组及其对应软反馈。边构造保留每图48条锚点空间边加32条特征边，这80条指进入图推理模块的边，不包括 GAT 内部可能添加的自环。

## 1. 先核对服务器现状并发布独立版本

用户继续复制单行命令执行。先确认当前 GPU、实例是否重启、已解压 NWPU 目录、BigEarthNet RGB 文件、持久化 split CSV 和预训练文件。不要为了检查目录重新下载或全量解压。

已知历史资源位置（本轮尚未在服务器重新确认）：

- BigEarthNet 原图：`/home/featurize/data/BigEarthNet-S2`。
- 冻结 split：`/home/featurize/work/GAViT_Project/bigearth_files/splits`。
- 预训练：`/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors`；历史 SHA256 为 `f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`。
- NWPU 原图目录：`/home/featurize/data/NWPU-RESISC45`；09-27用户回传ZIP SHA256校验OK、45类31500图每类700，见[下载记录](download_nwpu.md)。逐图解码、重复检查和group manifest已回传成功，清单SHA256仍待返回。

只读的 GPU 检查可直接执行：

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
ls -lah /home/featurize/data /home/featurize/work
```

这两行是早期定位命令；NWPU已确认，本轮直接使用顶部四行核对身份即可。

正式发布时只提交本功能清单内的代码、测试和记录，保留其余已有未提交研究材料；按用户授权推送后，服务器用独立目录固定到该 commit。记录 commit、dirty 状态、源码 SHA256、Python/包版本、GPU/驱动、CUDA、TF32/AMP。不要在服务器旧实验目录直接改代码。当前没有可填写的新发布 commit，因此此处不提供指向假定发布目录的训练命令。

## 2. NWPU 数据与1轮短测

用 `run_scene.py prepare` 对真实 NWPU-RESISC45 生成独立 manifest：45类、31,500张；split seed42、val split seed4242。预期 train5040 / val1260 / test25200，train+val训练池6300。进行原图尺寸、文件/像素哈希和重复检查；若发现重复先报告，不能删图、换seed或跳过校验。manifest 固定后三个模型共用。

09-27已核对发布版本`deb2b45`：其`run_scene.py prepare`已支持NWPU，`scene_classification/data.py`、包初始化和`experiment_identity.py`与当前本地版本逐字节一致。因此数据检查可先使用服务器已有的AID统一入口目录；这不意味着该旧版本已支持新soft训练。先回传服务器当前commit，再执行prepare并回传GPU型号：

```bash
git -C /home/featurize/work/GAViT_AID_20260920 --no-pager log -1 --oneline
python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py prepare --dataset NWPU-RESISC45 --duplicate-policy reject --split-seed 42 --val-split-seed 4242 --data-root /home/featurize/data/NWPU-RESISC45 --output /home/featurize/work/nwpu_protocol_20260927_01.json
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

上述是首次prepare的历史命令，结果如下；当前成功manifest为后续`_02`，尚未运行原图短测。输出路径独立且prepare拒绝覆盖已存在的manifest；本地旧表中的历史NWPU划分不自动混入新协议。

**09-27首次prepare返回**：31500/31500解码与文件/像素哈希完成，扫描46.4秒；默认reject发现首组同类像素重复`airport/airport_050.jpg`和`airport/airport_420.jpg`，在写manifest前终止。并非图片缺失或解压失败，报告只展示第一组，不能据此推定重复总数。

下一次使用已有`group`策略，按像素相同且标签一致的组进行分配，保留全部原图，同组不得跨train/val/test；跨类别冲突仍报错，精确划分数量不可静默放宽。保持seed42/4242，用新输出`_02`，成功后Swin/固定GAViT/soft GAViT统一使用该manifest。尚无已成功的`_01`划分或本轮训练结果，因此这属于训练前数据质量处理，不是根据模型表现换split。

```bash
python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py prepare --dataset NWPU-RESISC45 --duplicate-policy group --split-seed 42 --val-split-seed 4242 --data-root /home/featurize/data/NWPU-RESISC45 --output /home/featurize/work/nwpu_protocol_20260927_02.json
```

**09-27第二次prepare完成**：该命令扫描47.6秒，返回`MANIFEST COMPLETE`，数量5040/1260/25200；8组/8额外副本/31492唯一图像、同组不跨split。组归属为train的airport118/315与361/509，val的airport050/420，test的airport064/527、243/397、289/635及palace224/439、595/669。保留重复原图，不重新生成manifest。上述阶段仅完成数据准备，GPU工程门槛仍待执行。

新模式训练参数：

| 参数 | 值 |
|---|---|
| command / model / grouping | train / gavit / soft_prototype |
| phase / seed | smoke / 42 |
| train-per-class / val-per-class | 8 / 2（360训练、90验证） |
| device / batch-size / workers | cuda / 32 / 2 |
| pretrained-path | 经 SHA256 确认的本地 Swin-T 文件 |
| manifest / data-root / output | 已确认的绝对路径；全新独立输出目录 |

scene 的 smoke 固定1轮，保留正式增强、CE、AdamW和Cosine配方；不得使用 `--random-init` 做服务器预训练验收。这个短测是工程门槛，不是30轮校准的替代品。

启动验收必须同时看到进程、GPU 显存、`TRAIN FIRST BATCH OK`，不能只凭 PID。完成必须具备：

- `SCENE SMOKE COMPLETE`、`run.exit` 中 `exit_code=0`、`run.json.status=complete`，360/90数量正确。
- `graph_capture.actual_forward_checked=true`，每图48+32边；重载最大logit差≤1e-5；完整新架构元数据和 checkpoint SHA256。
- `assignment_probe/probe_manifest.json` 为固定 val 图；`assignment_diagnostics.jsonl` 仅epoch0、1，及 `epoch_000.pt`、`best.pt`、`final.pt`。统计有限、候选外分配为0、S行和与W列和为1。
- 在同次小样本验收中确认仅依赖分配 S 的梯度能到达投影/旧注意力/骨干，不能只看分类loss下降。需要服务器验收辅助调用或独立小检查；scene 日志本身不自动证明这项梯度。
- 记录训练/验证时间、吞吐、峰值allocated/reserved显存和总墙钟；probe时间单列。真实 CUDA 未测前不据CPU时间报价。

## 3. BigEarthNet 96/64 原图短测

保持正式237871 / 122342 / 119825划分不变。复制 train 和 val 清单的确定性小子集到独立短测目录，seed42，96训练/64验证；保留原19列标签和原始patch路径，不读取test图片或将test样本放进探针。记录原CSV与小CSV的哈希、抽样规则、逐类正负支持数。验证这160图的B02/B03/B04文件存在且可解码；某类在子集中没有正例时明确记录，该类AP不能作质量结论。

使用 `train_bigearth.py`，参数为 `--model gavit --grouping soft_prototype --edge_type sparse_hybrid --integration token_feedback --num_regions 16 --seed 42 --epochs 1 --batch_size 32 --lr 0.0003 --run_stage smoke`，并填入已核实的 `--data_dir`、本地 `--pretrained_path`、唯一 `--run_tag` 与独立 `--checkpoint_path`。这些是参数说明，尚非已发布版本的启动命令。

保留旧入口的BCE、AdamW/wd1e-4、阈值0.5、FP32/原TF32设置、workers4，以及train pin_memory=True / val False。入口会自动选择CUDA或CPU，服务器验收必须确认日志实际为CUDA。若主机内存失败，先记录故障；不要静默改配方重试。

完成需有1/1轮、有限loss/val指标、shell保存的退出码0、best权重及meta、last训练状态、`SOFT EPOCH`、epoch0/1探针与best/final快照。另对保存权重做真实CUDA重建、固定输入logit差≤1e-5和捕获边48+32核验；BigEarth入口本身没有scene式自动重载/显存完整报告，必须在服务器验收步骤补齐，不能仅凭best文件存在放行。不要用正式test集做短测。

## 4. 中断与历史兼容

短测失败保留目录并换新输出重新开始。BigEarth新模式的 `--resume` 要求训练状态、架构、seed/轮数/batch/lr、run tag及原始probe身份和epoch一致；已有epoch快照可恢复缺失或过期的best/final别名。若训练状态和诊断epoch不同或快照缺失，明确失败，不截断历史或重新定义epoch0。

继承的BigEarth续训机制未保存完整RNG状态，也没有自动绑定全部split/源文件哈希；不宣称与无中断训练逐bit等价。服务器必须保持并核验冻结的源码、完整split和预训练身份。需要严格完整实验身份或bit级续训时，另行补齐机制后再运行。

旧Swin/固定分组state_dict和默认路径保持；旧scene checkpoint仍需其记录的旧源码来通过原有code identity核验。不得为了载入历史结果取消该校验。旧NWPU `test_gavit.py` 会拒绝新soft架构并指向 `run_scene.py evaluate`。

## 5. 后续研究问题

三模型seed42校准回答“软分组整体是否改善NWPU验证表现，代价多大”。共同manifest、增强、CE、batch32、AdamW3e-4/wd1e-4、Cosine30，按val OA选轮；先审计同协议已有权重能否复用。配置和选轮冻结后才refit/test，不从test反调参数。之后按同样原则推进BigEarthNet验证和导师要求的双数据集多seed消融，保留负面seed。

HGTNet/DFAGCN材料仍待作者回复；本模块工程通过不代表外部比较、GPU短测或论文消融完成。
