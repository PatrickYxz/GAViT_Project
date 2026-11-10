# AID HGTNet：3060 batch16容量门禁

沿用用户已授权的AID推进流程。实际3060已完成240/60、batch2、一轮训练/验证/保存/重载，status exit0、完成产物与来源检查均true；验证OA76.6667%、macro-F1 72.3492%、重载0、总48.7654秒、allocated1.4314GiB/reserved1.578125GiB。上述是工程smoke，不能据此填写正式AID性能或batch16资源。

## 决策与范围

采用现有Python/PyTorch工具链新增隔离的小包，复用已在目标3060验证的身份、runtime与数据安全函数。保留旧AID/NWPU源、配置和包字节。原样运行NWPU探针不可行（45类、3090、NWPU输入）；直接完整训练会把未测batch16容量问题带入付费长任务。下一个GPU动作仅为16张原train图、四次一次性更新，结果用于正式包资源准备。

正式AID训练的待实现配方延续已披露NWPU独立实现：原30类manifest4000train/1000val/5000pool/5000test，最多100轮calibration，按验证correct严格增加选最佳/相同时选最早，从相同初态fresh refit pool到选中轮，再一次最终test；batch16、Adam lr1e-5/WD5e-5、FP32、同一Swin-B候选。对应25000+313E次更新，E为选中1–100轮。这是条件工作量，当前不生成完整GPU时长、费用或正式分数。

## 输入与数值引擎

- 原manifest SHA `30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22`；30类配置SHA `f313abb893b29d674ace281d8d2905670ee65db131f4240640079123e783589b`；Swin-B候选351654207B/SHA `68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893`。
- 用既有aid_data验证10000行元数据，从冻结smoke train中按label0–15各取路径最小一张，16张真实原train图。仅对16图读字节/SHA及RGB600→bilinear224→ImageNet归一化；无val/test图片读取，不重划分、不去重、不恢复ZIP、不载入已训练smoke权重。
- 新`aid_batch_probe_engine.run_batch_probe(model,images,labels,*,device,batch_size=16,warmup_steps=1,measured_steps=3,on_progress=None)`固定30类、全模型可训练FP32；Adam lr1e-5/WD5e-5/betas0.9,0.999/eps1e-8，无AMP/TF32/调度/梯度裁剪。沿用四步引擎，首步五组键精确匹配，梯度无缺失/非有限值、组L2正、代表参数实际更新。失败保留已实际发生更新。
- 同一16图重复四次，unique16/processed64/updates4。1warmup不进中位数，3measured步均CUDA同步；计算时间排除读图、增强、transfer、初始化、callback、首步诊断。保留原引擎peak reset位置与明确scope，不称全流程模型初始化峰值。没有epoch、验证、test、accuracy、checkpoint或正式训练。

## 入口、状态及交付

- 新`aid_batch_probe.py`：stdlib inspect/status，run懒导入ML；复用aid_entry严格3060选择/runtime函数，但自建来源清单与完成验收，不能调用240/60 smoke的inspect/status/source identity。
- 默认不硬编码UUID；唯一空闲RTX3060 12288MiB自动绑定，Torch导入前设置CUDA_VISIBLE_DEVICES。实际Python3.11/Torch2.2.2+cu121/vision0.17.2+cu121/NumPy1.26.4/Pillow10.3.0/CUDA12.1/cuDNN8902、8线程、FP32/seed42、cudnn deterministic/benchmarkfalse/双TF32false，CUDA容量与NVML差≤1GiB。
- 新schema `hgtnet-aid-batch16-capacity-probe-v1`、input_scope real_aid、counts16/0/0、target_gpu3060；标准库source/config/manifest/weights/16图在起止核验，独占输出锁、拒绝覆盖；首步、每个实际更新进度、四步列表、runtime、selection、provenance、summary、exit均落盘。最终来源/输入复核与一致产物通过后打印`HGTNET_AID_BATCH16_PROBE_COMPLETE`。
- status须核对完整real_aid身份、counts/16/64/4/1+3、有限正计时与中位数、实际3060运行时、五组451梯度张量/代表更新、selection/first_step/steps/optimizer_progress/progress/summary一致、末尾检查和exit0，拒绝skeletal或篡改结果；无ML导入/训练/读test。保留失败的实际步数，不用completed字段代替产物验收。
- 新builder source-only安全安装、显式inspect/run/status、不可覆盖。18个source/config文件=旧AID17减aid_engine和旧builder，加新entry/engine/builder；加provenance以及bootstrap/manifest为21个ZIP成员。自建来源清单恰与交付一致，旧aid_entry的未调用smoke清单不影响本包。
- 包名`hgtnet_aid_batch16_probe_3060_20261105.pyz`；默认持久化输出`/home/featurize/work/hgtnet_aid_batch16_probe_3060_20261105/runs/batch16_seed42_01`。无需新下载权重/数据/依赖。提供单行run并自动status，用户只需上传小包和贴回结果。

## 验证与边界

先有失败的行为测试再实现：30类四步/拒45类/缺组/非有限/首步已更新后诊断失败；只读入口无ML、严格身份/篡改/锁/进度；真实bundle安装/重复/篡改/越界路径和显式动作。CPU小模型仅证明数值与失败记录行为；已完成目标3060 batch2不外推batch16。交付前独立审查、核对包/source闭包及旧源/包/结果表保护快照，更新日记。无需租卡、执行远程命令或Git提交推送。
