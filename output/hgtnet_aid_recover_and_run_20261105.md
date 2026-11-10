# AID当前缺项恢复与一轮验证

2026-11-05详细inspect明确两项原因：当前GPU为`GPU-df386e6c-dbb3-0952-8190-3265a7bd8ae8`（RTX3060 12288MiB、空闲11909MiB、无计算进程），而命令固定的是昨天的`GPU-7acce72a-93cd-e56f-35d8-5d33ecf6655e`；默认原图路径中的data目录不存在。原清单/300条选择及初始权重已通过，软件metadata和work容量也未阻塞。

不需要修改或重传现有运行包。执行这一整行：旧ZIP当前完整SHA匹配才会解压，解压成功才run；run再次验证300图、原始清单、初值、空闲显卡与实际CUDA/软件，再做240/60、batch2的一轮smoke。

```bash
printf '%s\n' 'cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7  /home/featurize/work/AID_visionflan_20260920.zip' | sha256sum -c - && unzip -n -q /home/featurize/work/AID_visionflan_20260920.zip -d /home/featurize/data && /environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_smoke_3060_20261104.pyz run
```

这里省略手填UUID，使用程序已有的唯一空闲RTX3060检测；仍要求3060/12288MiB、唯一且无计算占用，并把实际UUID写入runtime/binding后设置CUDA_VISIBLE_DEVICES，再导入Torch。没有关闭身份或硬件检查。若有多个符合条件GPU会停止，不随意选卡。

ZIP缺失、SHA不符或解压失败会停止，不下载、不重新划分或覆盖原清单；unzip -n保留现有文件。解压阶段暂时没有输出正常。恢复原图会解包包含test在内的原数据；本次模型只使用240train/60val/test0。

看到`HGTNET_AID_SMOKE_COMPLETE`后读取状态：

```bash
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_smoke_3060_20261104.pyz status
```

若又报错，保留错误输出；不直接重试已创建的同一输出目录。[当前诊断记录](../results/hgtnet_aid_smoke_preflight_20261105.json)保存本次已知信息，实际恢复和GPU运行结果仍待返回。
