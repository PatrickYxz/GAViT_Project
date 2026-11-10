# DFAGCN seed42正式实验：RTX3080 10GiB

**最新操作改用[fix1资源修复包](dfagcn_nwpu_formal_3080_fix1_README_20261030.md)。** 此旧包的metadata后缀/旧UUID和单卷容量门禁会拦截当前实例；下面保留旧批次交付记录，不覆盖旧包。

3080一次性检查已实际通过，VGG与图batch100正常。这个新包用于完整NWPU独立实现实验，保持seed42和完整配方；此前的4090小样本和3080探针结果继续保留。

## 上传和准备

只需把[正式小包](dfagcn_nwpu_formal_3080_20261030.pyz)上传到`/home/featurize/work/`，保留文件名。已有VGG权重、manifest及持久化NWPU ZIP复用，无需重新上传大权重，也不需要修改旧GAViT服务器目录。

文件150901字节，SHA256为`66cbb46bc8bc0fb8a79e7e95794534e6efb430ee1630be2258c91df16c98861b`。开发基线commit为`058f1d8b79d0be538ec358eb742d2cd3992921d4`，DFAGCN增量尚未提交，交付版本以包SHA和保存的逐文件源码SHA识别。

先编译新包内固定版本的LIBLINEAR。它只编译，不训练：

```sh
python /home/featurize/work/dfagcn_nwpu_formal_3080_20261030.pyz build-linear
```

再检查当前资源：

```sh
python /home/featurize/work/dfagcn_nwpu_formal_3080_20261030.pyz inspect
```

`ready: true`才执行输出中`nohup_command`的完整后台命令。该命令已经绑定当前物理UUID、完整SHA、配方和独立新输出目录，不需要手改参数。若存在blockers，回传检查输出即可，不要绕过门禁或删除旧目录。已有后台日志时不会生成覆盖命令，保留旧日志并回传检查结果。

## 正式流程

1. 在这张3080上重新提取完整6300张pool的冻结VGG特征，派生5040张train和1260张val缓存。
2. 最多100轮校准，按验证CE选epoch；九个C只按验证OA选取，平局选较小C。
3. 重置同seed模型，在6300张train+val上按选定轮数重新拟合，保存和重载模型校验。
4. 重新提取25200张test的冻结特征，进行最终测试。保存OA、macro-F1、逐类准确率、混淆矩阵和逐图预测。

图batch100、VGG提取batch1，FP32且AMP/TF32关闭。正式缓存不混用旧450张4090缓存；原图不足时从完整SHA匹配的work ZIP按阶段恢复，只写新运行目录。pool阶段不恢复test，test在fresh refit完成后才读取。

按当前布局，全部float32缓存约9.41GiB；LIBLINEAR九个C及测试会保存文本中间文件，入口保守检查40GiB可用磁盘。容量数值来自格式计算，不是实际运行测量。租赁费率未知，单批0.57秒不能换算整个实验费用，实际每轮时间见日志。

## 查看启动和进度

新输出目录是`/home/featurize/work/dfagcn_nwpu_formal_3080_20261030/runs/nwpu_seed42_01`。后台PID本身不算启动成功，先看真实设备和首个VGG/训练batch，再看GPU进程。

```sh
python /home/featurize/work/dfagcn_nwpu_formal_3080_20261030.pyz status
```

```sh
tail -n 35 /home/featurize/work/dfagcn_nwpu_formal_3080_20261030/runs/nwpu_seed42_01/run.log
```

```sh
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv,noheader
```

把status输出和首批日志贴回即可判断启动是否正常。status为短摘要，不会输出大矩阵；完整指标在运行目录保存。

## 中断和验收

已有output一律拒绝覆盖；失败保留cache、history、failure/traceback和退出记录。本包不承诺epoch断点恢复。出错时回传status和日志，根据实际失败阶段处理，避免重新提取可用缓存或无理由重跑。

正式完成需要最终COMPLETE、exit0、配置、最佳/最终checkpoint和完整测试指标都已保存。当前本地只交付运行包，没有远端启动或正式准确率结果；实际启动和完成以服务器后续回传为准。

本实验是披露配方的DFAGCN独立实现单seed结果，不是作者官方数值复现，不证明稳定优势。与原文及既有Swin/GAViT结果比较时保留数据协议、预训练、实现和硬件差异说明；跨卡耗时分别记录。

完整约定：[正式入口设计](../docs/external_baselines_20261027/formal_3080_runner_20261030.md)。

本地验收：139项DFAGCN相关测试通过，实际pyz的status可用且inspect正确拒绝本机缺少的服务器资源；生成的nohup命令已用无训练副作用的进程实际验证执行及日志防覆盖。Mac OpenMP共享内存中止/两项旧pipeline边界错误保留，加入本地`KMP_INIT_AT_FORK=FALSE`后139项完整通过；该设置不写入服务器配方。独立终审代理因用量限制未完成，最终由主代理复核，不能称本轮独立终审通过。[完整验收证据](../results/evidence/dfagcn_formal_3080_prepare_20261030/validation.json)。
