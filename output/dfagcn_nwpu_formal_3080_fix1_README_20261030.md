# 正式包资源修复版

这次ready:false有三个根因：发行包metadata省略CUDA后缀、实例物理UUID已变化、work空间不足以容纳旧包的全部缓存。manifest、权重和已编译LIBLINEAR均已通过，不需要重传大文件或重新安装Torch。

## 操作

只上传[新的fix1小包](dfagcn_nwpu_formal_3080_20261030_fix1.pyz)到`/home/featurize/work/`，保留文件名。旧包和目录继续保留。

文件151769字节，SHA256为`79398dc4b6d703be65cd32a968d49daa6640fe6c4bbab6a33e930b312954a6c4`。

执行一行：

```sh
/environment/miniconda3/bin/python /home/featurize/work/dfagcn_nwpu_formal_3080_20261030_fix1.pyz inspect
```

它会复用旧正式包中已成功编译的LIBLINEAR，以及现有VGG、manifest、ZIP。若找不到旧binary才需要对这个新包执行build-linear。`ready:true`后，复制输出中完整的`nohup_command`执行，不用手改UUID或参数。若仍有blockers，回传输出，不删除旧目录或绕过检查。

## 本次改动

- metadata接受Torch2.2.2和vision0.17.2省略后缀的形式；run实际导入后仍要求CUDA12.1、cuDNN8902、正确设备、软件及FP32配置。不能凭metadata宣称CUDA已验证，不接受cu118或其他版本。
- inspect只有恰好一张空闲10GiB RTX3080时才自动选择，生成run命令时写入实际UUID。多卡必须显式选择；run始终要求显式UUID并核对本次runtime。当前回传卡为GPU-0d9901ca-d2e0-d398-b9b4-2550e6db2fed。
- 新生成的约9.41GiB缓存和恢复图片放在实例本地`/tmp/dfagcn_nwpu_formal_3080_20261030_fix1_nwpu_seed42_01`，该盘df显示剩约349GiB。无需预先创建目录，run独占创建；已有目录会拒绝覆盖。
- checkpoint、训练history、日志、LIBLINEAR模型/文本、完整指标/预测、源码配方及缓存index保存在work新目录`/home/featurize/work/dfagcn_nwpu_formal_3080_20261030_fix1/runs/nwpu_seed42_01`。
- scratch仍要求40GiB；work独立要求全部SVM文本保守上界15434779500B加1GiB产物/余量，即16508521324B（约15.375GiB）。这次回传work17029922816B满足该分卷预算；inspect和run仍会重新检查当时空间。同卷则合并预算，不把同一份剩余空间重复使用。
- 模型、seed42、100轮上限、batch100/VGG batch1、九C、选轮及fresh6300refit→25200test协议和核心算法文件都保持。没有重复探针或删除历史数据。

## 看进度

```sh
/environment/miniconda3/bin/python /home/featurize/work/dfagcn_nwpu_formal_3080_20261030_fix1.pyz status
```

```sh
tail -n 35 /home/featurize/work/dfagcn_nwpu_formal_3080_20261030_fix1/runs/nwpu_seed42_01/run.log
```

启动以实际设备及首个VGG/训练batch日志、GPU进程为准。正式完成仍需COMPLETE、exit0及完整指标/模型落盘，不只看PID。

实例归还后/tmp缓存会丢失，可按保存的index/原图/权重重新提取；work中的模型、日志与正式结果仍保留。新包没有epoch恢复功能；失败保留现场，再按实际阶段处理。

修复依据：[资源约定](../docs/external_baselines_20261027/formal_resource_fix_20261030.md)。本地/代理测试不代表新实例正式CUDA训练已经开始；以服务器后续回传为准。

验收：148项相关测试通过，新增8项资源回归有有效失败→修复证据，整个进程禁止ML导入仍8/8通过。独立审查无待修项；最终实际pyz已验证status和预期本地阻断、包内源码及21项旧源/配方/包SHA不变。[完整证据](../results/evidence/dfagcn_formal_resource_fix_20261030/validation.json)。
