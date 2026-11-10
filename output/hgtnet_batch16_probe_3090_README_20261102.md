# HGTNet3090 batch16有界容量检查

将[hgtnet_batch16_probe_3090_20261102.pyz](hgtnet_batch16_probe_3090_20261102.pyz)上传到Featurize的/home/featurize/work/。只上传此61006字节小包，沿用现有Swin-B、manifest、ZIP和已恢复图片。

先运行这一行：

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_batch16_probe_3090_20261102.pyz inspect
```

ready=true时，复制输出中的run_command整行执行。它绑定inspect查到的当前3090 UUID，默认使用新的安装目录hgtnet_batch16_probe_3090_20261102、runs/batch16_seed42_01与/tmp/hgtnet_batch16_probe_3090_20261102_rtx3090_seed42_01。不覆盖已完成smoke或旧失败结果；换卡后同样从当前inspect命令获取UUID。

完成HGTNET_BATCH16_PROBE_COMPLETE后执行并回传：

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_batch16_probe_3090_20261102.pyz status
```

## 这次具体做什么

- 全部74619858参数可训练、FP32/seed42、无AMP/TF32，架构/config/Swin-B初值沿用；不读取已训练smoke checkpoint。
- 从冻结smoke train360中按路径取class0–15各第一张，共16张，只解码train；验证/test均0。优先逐图SHA核验并复用旧smoke /tmp缓存，否则验证完整ZIP SHA后仅恢复16图到新scratch。
- batch16；Adam lr1e-5、weight_decay5e-5、CE；betas0.9/0.999、eps1e-8、无scheduler、无实际梯度裁剪是显式工程选择。RGB resize224 bilinear/ImageNet normalize，无增强。该预处理不是正式crop/flip配方。
- 同16图复用4次：1次预热+3次计时，共4次临时更新、64次样本呈现。无训练epoch、准确率、val/test、模型保存或正式训练。
- step_seconds/median_step_seconds/compute_images_per_second仅计已解码并放GPU后的forward/backward/Adam及有限梯度检查，CUDA同步；不含IO/预处理/拷贝/初始化/日志/首步诊断，不是正式epoch吞吐。wall_seconds另含上述整个准备和检查流程。首步预热含Adam状态分配与梯度诊断，排除于3次计时统计。
- CUDA allocated/reserved峰值覆盖模型/输入/Adam状态/梯度/激活/诊断，只有一份模型，不做checkpoint回读。保守磁盘门沿用work2GiB、scratch1GiB，非实际磁盘开销。OOM或失败保留日志与实际已成功更新次数，不自动改小batch、重跑或删除目录。

## 本地交付核验

333项HGT/DF相关测试通过（无失败/错误/跳过）；独立28项审查通过。实际生产pyz的stdlib-only inspect/status通过，未创建runs；本机没有目标GPU/数据，故本地ready=false是预期结果，不能当作3090已通过。104项交付/历史保护核验通过；16份旧源码/config与旧包保持一致，新增两个probe模块及受限打包入口。

包SHA256：9563c9b14bb182bb58104480bedee963efda6c44e48cd63fce5f8522a9426799

bundle ID：75e7fd09dc0fd9404532f99e319cfe605ecb301c453ab849ce1d2eac25cd72db

候选权重完整SHA仍为68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893。代码Git基线058f1d8b79d0be538ec358eb742d2cd3992921d4，当前未提交增量由包内逐文件SHA绑定。[完整本地证据](../results/evidence/hgtnet_batch16_prepare_20261102/README.md)。真实3090的batch16容量/速度待服务器回传，当前没有正式训练命令或新benchmark分数。
