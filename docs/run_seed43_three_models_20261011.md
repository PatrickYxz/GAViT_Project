# 补充seed43：NWPU三模型与BigEarthNet软分配

2026-10-12最新：本批NWPU43三模型与BigEarth soft43正式训练/test/资源全部完成并归档，BigEarth总/训练/test退出码0；soft43 mAP72.867571%、macro-F1 66.673892%、micro-F1 76.905971%，best第8轮/val mAP78.99004691745016%。[正式对比及资源](../results/bigearth_soft43_ablation.md)与[43/45两seed表](../results/bigearth_three_model_two_seed_summary.md)已更新。BigEarth整批6小时22分21.73秒/约10.64元，加上NWPU43三模型，两批正式队列合计7小时22分39.06秒/约12.32元（1.67元/h，准备/短测/空闲另计）。本批无需继续回传日志或重跑。下方所有预检/短测/正式启动及读摘要步骤均为历史，不重复执行。

2026-10-11用户明确要求：补NWPU43三模型及BigEarthNet soft43。**NWPU43已正式完成并回传exit0、test指标、选轮和资源；BigEarth soft43预检与当前4090单轮工程检查均已通过，现在执行下面正式训练命令。下方旧预检/短测/NWPU启动步骤留作历史，不重复运行。** [NWPU43结果](../results/nwpu_seed43_test_comparison.md)、[双seed比较](../results/nwpu_two_seed_summary.md)。

## 历史执行步骤（本批已完成）：正式soft43训练30轮→best完整test

当前4090工程短测已回传exit0/complete，96/64、1轮、CUDA重载差2.38e-7、实际48+32图边与探针归一化符合，见[工程报告](../results/bigearth_soft43_smoke_validation.md)。小样本31.63%mAP不进入论文正式表，也不作为改配置依据。

复制[正式单行启动命令](../results/evidence/seed43_bigearth_smoke4090_20261011/bigearth_launch_checked_command.txt)完整一行到服务器，执行一次。它只读核对已存工程产物与重新完整预检，通过后创建`/home/featurize/work/bigearth_soft43_20261011_01`，后台串行训练30轮和best checkpoint的119825张test；冻结模型/配方与已有调度器保持，只补soft43。若输出目录已存在或任何身份不一致，停止并保留产物诊断。

看到PID后用两个终端查看日志和GPU：

```bash
tail -f /home/featurize/work/bigearth_soft43_20261011_01/run.log
```

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

回传`DEVICE`、`Train: 237,871 | Val: 122,342`、`Training GAVIT ... 30 epochs`及首个实际训练batch进度/GPU进程。只有PID不能算训练启动成功。第一次完整epoch的`EPOCH RESOURCE`/`SOFT EPOCH`回传后更新预算；现仍为4090估计6–8h、按1.67元/h10.02–13.36元，准备/空闲另计，不能按14秒短测外推。Ctrl+C仅退出tail；队列继续运行。

最终须`BIGEARTH SOFT43 COMPLETE`、总与训练/test各exit0、完整30轮、val best及精确119825 test mAP/两F1/19类AP、预测、配置和权重SHA保存。结束后只读回传：

```bash
cat /home/featurize/work/bigearth_soft43_20261011_01/run.exit /home/featurize/work/bigearth_soft43_20261011_01/summary.json
```

[新增只读门禁及命令验证](../results/evidence/seed43_bigearth_smoke4090_20261011/command_validation.json)、[9项工件fixture验证](../results/evidence/seed43_bigearth_smoke4090_20261011/artifact_gate_validation.json)。正式尚未收到PID/GPU/首batch，状态仍待启动。下面按当时状态保留准备历史。

## 历史：BigEarth soft43目标GPU工程检查（已完成）

预检确认237871/122342/119825、ed480e6、原43对照权重、软件版本与FP32设置，当前GPU为`GPU-cede40c0-f93d-7c8e-1006-cf6183e53592`、RTX4090。[回传转录与核验](../results/evidence/seed43_bigearth_preflight_20261011/validation.json)。旧Swin43完整30轮/best6、固定43保存21轮/best8及历史版本/硬件限制继续保留。

复制[BigEarth96/64一轮命令](../results/evidence/seed43_bigearth_preflight_20261011/bigearth_smoke_command.txt)完整一行到服务器，前台执行一次。沿用已验证工程子集的Random(42)抽样规则，训练seed43、batch32/workers4、BCE、soft原配方、原精度；单轮cosine周期1仅用于工程检查，正式仍30。命令在创建独立目录前检查GPU空闲、代码/预训练和本次硬件/软件/精度；使用原TIF解码，保存子集CSV/原波段ZIP/哈希/正例数，不改正式划分。输出`/home/featurize/work/gpucheck_bigearth_soft43_4090_20261011_01`，不覆盖旧短测或正式目录。

完成须有`BIGEARTH_SOFT43_SMOKE_COMPLETE`、exit0、96/64、1轮、有限loss、正确19类配置与metadata、严格CUDA重载差≤1e-5、epoch0/1/best/final探针及真实48+32边。工程结果不填论文精度表，部分小样本类别可能无正例。完成后回传：

```bash
cat /home/featurize/work/gpucheck_bigearth_soft43_4090_20261011_01/run.exit /home/featurize/work/gpucheck_bigearth_soft43_4090_20261011_01/summary.json
```

失败保留目录和报错，已有目录先检查，不另起retry绕过。命令Python AST、真实ed480e6参数解析、Bash/zsh、载荷字节/SHA和成功标记顺序检查通过；本轮没有本地CUDA执行。工程回传通过后进入已授权单soft43正式训练→best完整test，不新增NWPU任务。[命令验证](../results/evidence/seed43_bigearth_preflight_20261011/smoke_command_validation.json)。

研究问题：冻结现有配方与划分，观察训练随机种子从已完成的42/45换到43后，三模型排序及差值是否重复。不是根据结果重试，也不追加FFN、空间先验、温度或图边搜索。结果进入双数据集三模型消融与多seed描述统计；有既存匹配checkpoint时优先复用，不重新训练。

| 数据集/seed | Swin | 固定4×4 GAViT | soft_prototype GAViT |
|---|---|---|---|
| NWPU42，正式test OA% | 93.932540，保留 | 94.103175，保留 | 93.869048，保留 |
| NWPU43，正式test OA% | 93.857143，完成 | 93.765873，完成 | 93.626984，完成 |
| BigEarth43，test mAP% | 71.1，复用（打印一位小数） | 71.7，复用（打印一位小数） | 72.867571，正式30轮/test完成 |
| BigEarth45，test mAP% | 71.676930，保留 | 70.258236，保留 | 70.836919，保留 |

旧分数来源：[NWPU42正式test](../results/nwpu_seed42_test_comparison.md)、[BigEarth43逐类证据](../results/evidence/bigearth_seed43_per_class_20261010/server_return.txt)、[BigEarth45三模型](../results/bigearth_soft45_ablation.md)。补完本批后NWPU三模型各有42/43两seed；BigEarth soft有43/45两seed，Swin/fixed仍保留42/43/44/45全部结果。用于三模型配对消融的共同seed是43/45，不能拿Swin/fixed四seed均值与soft两seed均值冒充配对比较。最终三seed要求仍有缺口，不能提前标为满足全部导师要求。

## 冻结配置

两个数据集均调用现有干净发布源码`/home/featurize/work/GAViT_SOFT_20260927`，commit `ed480e60fea03ed786fdba70c144de5e815c34f5`。不使用当前本地包含其他候选的脏工作区，也不在服务器修改正式源码。预训练SHA为`f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`。

NWPU使用原manifest `/home/featurize/work/nwpu_protocol_20260927_02.json`，SHA `ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`。清单划分seed42/4242保持不变，**43只改变训练随机种子**。先三个模型各30轮5040train/1260val，以val OA最大值选轮、同分取首次；再从同一ImageNet权重重新初始化，各自按**本次seed43自己的best epoch**拟合6300train+val，Cosine T_max仍30；最后用各自refit `last.pth`测试25200张。不能套用seed42的29/29/28轮或从42权重续训。batch32/workers2、RGB224、CE、AdamW3e-4/wd1e-4、FP32无AMP、两项TF32=false、8CPU线程。三个calibrate→三个refit→三个test，共九阶段。

BigEarth只补soft43：237871train/122342val/119825test、30轮、batch32/workers4、RGB224/ImageNet normalization与原水平/垂直翻转、BCEWithLogitsLoss、AdamW3e-4/wd1e-4、Cosine T_max30、阈值0.5，FP32无AMP、matmul TF32=false/cuDNN TF32=true、8线程。按val mAP选best，评估完整test；不合并val、不续训固定模型。固定CSV/pretrained SHA沿用已完成soft45门禁。K16、sparse_hybrid 48+32有向边、GAT256/4heads/2layers、token_feedback；soft dim128/radius1/temperature0.1/iteration1，`local_prototype_cosine_v1`全部保持。

**BigEarth43比较限制：** 旧Swin43完成30轮/best6，固定43停在21轮/best8；旧训练4090、固定测试3060、commit bb9f1cf。新soft43使用30轮及ed480e6，当前使用用户指定的RTX 4090，软件/精度仍与已验证soft45参照核对；显卡型号与物理身份变化单独记录。预检核对旧checkpoint/metadata/实际保存轮数/现有测试日志，但不能消除这些历史混杂。seed相同不代表三模型完全同条件，多seed稳定收益或纯分组因果结论必须保持边界。完整30轮的seed45对照继续保留；本次不自动重跑旧43。

## 先执行只读预检

1. 复制[NWPU43只读预检命令](../results/evidence/seed43_gpu4090_20261011/nwpu_preflight_command.txt)的完整一行到Featurize终端。检查源码、清单、预训练、31500路径、已有43产物及GPU空闲；出现`NWPU43_PREFLIGHT_PASS`后回传最后JSON。发现已有产物先审计复用；不能删除或另起名称绕过重复检查。
2. 若`NWPU DATA_MISSING`，仅恢复原ZIP，不重新prepare。下面命令先验SHA再`unzip -n`保留已有文件；恢复后重新只读预检。全量像素/解码身份检查仍由每个原始入口执行。

```bash
printf '%s\n' 'beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53  /home/featurize/work/NWPU-RESISC45_a826b44.zip' | sha256sum -c - && unzip -n -q /home/featurize/work/NWPU-RESISC45_a826b44.zip -d /home/featurize/data
```

3. GPU空闲时执行[BigEarth soft43只读预检](../results/evidence/seed43_gpu4090_20261011/bigearth_preflight_command.txt)。它核对原43两模型、已验证soft45环境、冻结CSV与19维标签、所有RGB波段，并搜索已有soft43。若数据缺失，按[Featurize手册](featurize_runbook.md)恢复实例本地`/home/featurize/data/BigEarthNet-S2`。不重生成split、不修改正式代码。最后须为`BIGEARTH_SOFT43_PREFLIGHT_PASS`；将JSON或最后错误回传。预检只读，不训练、解压或写checkpoint。

当前两批门禁均要求单张RTX 4090，记录实际UUID/驱动/显存；每批各阶段保持初始GPU身份一致。BigEarth允许历史soft45的2080 Ti换到4090，仍严格核对Python、CUDA、cuDNN、包版本及精度。软件不匹配时先诊断；不要删除断言或更换数据划分。此次报错来自旧显卡限制，与解压无关，无需因此重新解压。GPU身份通过不等于目标CUDA前向和小规模训练已验收。BigEarth依赖已存在且hash固定的`bigearth_seed45_20260922_01/run_seed45.py`及soft45环境记录，缺失时先找持久产物，不重跑45。

## 4090目标设备工程短测

**工程完成回传已收到：** 当前4090完成31500图校验、360/90一轮、实际48+32图边和`SCENE SMOKE COMPLETE`。46.67%仅为工程小样本OA，train1.70秒仅为纯train段。无需再运行短测；确切artifact/exit/重建数值由下面新的正式启动命令只读核验，通过后直接进已授权九阶段。[回传与交付记录](../results/evidence/seed43_nwpu_smoke4090_20261011/validation.json)。下列短测步骤保留为历史。

最新只读回传仍为PASS，但物理GPU UUID变为`GPU-cede40c0-f93d-7c8e-1006-cf6183e53592`；型号/驱动/显存与前次相同，见[第二次身份记录](../results/evidence/seed43_nwpu_preflight_20261011/validation_gpu2.json)。当前没有工程短测或正式启动回传。若工程目录已存在，先读日志和exit核对身份，不覆盖或另起retry绕过。

NWPU只读预检已回传`NWPU43_PREFLIGHT_PASS`，当前helper SHA为`bb50d71a98aaa44b14c7f604c2236218c8b663a90f1e89254d6fd524c0340e5e`，与交付代码相符；源commit、manifest/pretrained SHA、31500路径、空闲单4090和已有正式43产物检查通过。记录见[预检回传核对](../results/evidence/seed43_nwpu_preflight_20261011/validation.json)。这些检查不包含图片实际解码和CUDA前后向。

按`AGENTS.md`训练前检查，先在新4090实例执行[一轮软分配短测命令](../results/evidence/seed43_nwpu_preflight_20261011/nwpu_smoke_command.txt)：沿用ed480e6真实入口和ImageNet、seed43、每类8train/2val（共360/90）、batch32/workers2/8线程/原FP32配方，前台单轮，包含forward/backward、validation、保存和CUDA重建。独立工程目录`/home/featurize/work/gpucheck_nwpu_soft43_4090_20261011_01`保留，与正式输出分开；小样本分数不进入正式效果表，不作为选配置依据。

完成需`SCENE SMOKE COMPLETE`、`run.exit=exit_code=0`、`run.json.status=complete`、counts360/90、final_epoch1、真实设备RTX 4090、实际48+32图边、正确源版本与预训练、有限loss及重建差≤1e-5。回传：

```bash
cat /home/featurize/work/gpucheck_nwpu_soft43_4090_20261011_01/run.exit /home/featurize/work/gpucheck_nwpu_soft43_4090_20261011_01/run.json
```

短测失败保留日志和目录，不重复启动同一目录。CLI及Bash/zsh语法已核对；本地使用仅解析参数的fixture，未运行真实CUDA。通过后接下方正式九阶段，不改变已有发布模型或主指标。

## 预检通过后的执行顺序

**最新九阶段总完成回传已收到：** 所有阶段均STAGE_COMPLETE并打印NWPU43_THREE_MODELS_COMPLETE，九阶段wall合计3615.20秒，按用户1.67元/h折算约1.68元（准备/空闲另计）。不重复启动、短测或测试。执行[紧凑正式摘要命令](../results/evidence/seed43_nwpu_queue_complete_20261011/read_summary_command.txt)回传实际OA/F1/选轮、终态exit及资源/身份；随后可执行上方BigEarth soft43只读预检。[本批完成记录](../results/evidence/seed43_nwpu_queue_complete_20261011/summary.json)。下列启动与阶段查看步骤保留为历史。

**已执行启动，勿重复运行启动命令：** 当前Swin校准已打印30/30及完成标记，最大val OA93.095238%、首次epoch23。队列验收与后续模型需读总日志确认。[已回传进度](../results/nwpu_seed43_calibration_progress.md)。`tail -f`只跟踪指定文件，看到Swin完成后Ctrl+C退出查看，再执行`tail -n 40 /home/featurize/work/nwpu_three43_20261011_01/run.log`核对实际当前阶段；固定模型日志为`fixed_calibrate.process.log`，不能因Swin日志停止更新而判定队列停止。以下启动命令保留为记录。

复制[带工程验收的NWPU九阶段后台命令](../results/evidence/seed43_nwpu_smoke4090_20261011/nwpu_launch_checked_command.txt)的完整一行，只执行一次。它先核对工程exit/metadata/权重/重建/history和当前软件及最新GPU UUID，确认通过才创建输出并启动；核对失败不创建正式目录。脚本写入新目录`/home/featurize/work/nwpu_three43_20261011_01`，每阶段进程exit0、run.exit0与身份/数量/权重检查通过才进入下一项；源码/输入/GPU在每个阶段重新检查。独立process日志保留入口预检阶段的错误，原入口仍保存自己的run.log/run.json/history/checkpoint/metrics。

```bash
tail -f /home/featurize/work/nwpu_three43_20261011_01/swin_calibrate.process.log
```

另一个终端看GPU占用和队列阶段：

```bash
nvidia-smi
tail -n 30 /home/featurize/work/nwpu_three43_20261011_01/run.log
```

首项须看到MODEL swin、PHASE calibrate、5040/1260、实际`TRAIN FIRST BATCH OK`及epoch进度；PID不算启动成功。随后日志名换为`fixed_calibrate`、`soft_calibrate`、对应`*_refit`与`*_test`。refit轮数以此次selected_epochs为准。Ctrl+C只退出tail。

NWPU完成需`NWPU43_THREE_MODELS_COMPLETE`、总run.exit0及九个原入口run.exit0。回传完整摘要：

```bash
cat /home/featurize/work/nwpu_three43_20261011_01/run.exit /home/featurize/work/nwpu_three43_20261011_01/summary.json
```

随后在GPU空闲且NWPU完成后，复制[BigEarth soft43训练→测试后台命令](../results/evidence/seed43_gpu4090_20261011/bigearth_launch_command.txt)。它先在独立子进程重核只读预检，确认CUDA上下文退出后才创建新目录`/home/featurize/work/bigearth_soft43_20261011_01`并启动；队列再独立核对同一身份。不会排队重训Swin43/固定43或其他seed。

```bash
tail -f /home/featurize/work/bigearth_soft43_20261011_01/run.log
```

首项须看到实际DEVICE、237871/122342样本及`Epoch [1/30] Train`持续前进，并有GPU进程/显存；第一轮`EPOCH RESOURCE`用于更新预算。完成须`BIGEARTH SOFT43 COMPLETE`、总/预检/训练/测试exit0、30轮历史及完整119825test。

```bash
cat /home/featurize/work/bigearth_soft43_20261011_01/run.exit /home/featurize/work/bigearth_soft43_20261011_01/summary.json
```

任何失败保留整个目录/日志，先诊断并复用已完成阶段；不自动重试或删除重跑。SIGKILL/归还实例可能无法保存exit标记，缺标记不能当完成。两批均独占输出，重复启动在写新文件前拒绝。

## 输出、时间与验证边界

NWPU保存九阶段完整日志、命令、exit、训练曲线、best/last配置、calibration来源SHA、各25200条预测/45类accuracy及macro-F1/混淆矩阵/资源。BigEarth保存完整30轮精确history、best及last训练状态、固定8val探针、119825×19标签/概率、三项指标/19类AP、权重SHA、配置及资源。按验证选best，不按test改配置。

历史2080 Ti NWPU42九阶段约2小时18分，soft45约14小时44分；它们不是当前4090的实测耗时。沿用已沟通的4090计划估计：NWPU三模型约1–2h，BigEarth soft43约6–8h，两项串行7–10h；按用户实际1.67元/h约11.69–16.70元，12h预算20.04元，含数据恢复/预检/空闲可预留25–30元。实际首轮资源记录用于更新估计，不保证按此结束。阶段有效吞吐包含验证/IO/探针，不能当公平纯前向效率。

初版21项验收保留；此次加入4090回归，本地共23项通过，exit0（包括错误GPU、软件/精度漂移及旧固定43保存21轮边界）；Python/Bash/zsh、打包SHA/载荷往返、两批实际后台fixture启动/预检拒绝/重复运行不改产物检查记录见[4090交付验收](../results/evidence/seed43_gpu4090_20261011/validation.json)。CUDA/GPU/训练数据使用fixture的检查只证明调度门禁；未在本地运行真实模型训练，仅NWPU服务器只读预检已有用户回传PASS，目标CUDA短测已有完成回传，确切artifact待启动命令核验；NWPU完整九阶段验收已回传完成，正式数值待摘要；BigEarth预检仍待回传。正式模型代码未修改、未Git提交/推送。训练返回后立即更新日记与专门结果表，当前不给未运行模型填指标。
