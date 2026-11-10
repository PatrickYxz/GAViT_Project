# AID原图恢复：沿用已经通过的划分与权重

2026-11-04目标机回传：原AID manifest SHA和结构通过；Swin-B候选权重完整SHA通过；`/home/featurize/data/AID/AID`不存在，240train+60val全部missing，尚未读取图片或运行训练。当前设备为RTX3060、12288MiB，UUID `GPU-7acce72a-93cd-e56f-35d8-5d33ecf6655e`；这只是nvidia-smi清单，实际Torch/CUDA与模型容量还未验证。

已从历史记录找回持久化原ZIP：`/home/featurize/work/AID_visionflan_20260920.zip`，2628474154字节，SHA256 `cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7`。2026-09-20用户曾回传完整SHA并核对一致，但本次实例上的ZIP是否仍在、字节是否相同尚未知。

复制这一整行到Featurize终端。校验成功才会解压，解压成功才会运行现有检查包；任一步失败会停止。`unzip -n`保留已有文件。命令不下载、不重新划分数据、不运行训练：

```bash
printf '%s\n' 'cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7  /home/featurize/work/AID_visionflan_20260920.zip' | sha256sum -c - && unzip -n -q /home/featurize/work/AID_visionflan_20260920.zip -d /home/featurize/data && /environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_inputs_20261104.pyz inspect
```

原ZIP有两层AID，解压后目录为`/home/featurize/data/AID/AID/类别`，正好对应现有manifest相对路径。`-q`解压时短暂无输出是预期行为。恢复阶段解包原数据，包括test文件；没有模型评估或读取标签调参，随后inspect仍只哈希预定300张train/val图片。

若ZIP不存在、SHA不符或解压失败，保留错误输出，不执行旧文档里的prepare命令、不覆盖原manifest。只有旧ZIP也不可用时才另行处理原数据源恢复，不盲目重复下载。

本次work余9767485440字节，实例本地祖先目录余349589377024字节。图片放实例本地data；不往work解压，不删除旧checkpoint/日志腾空间。

目标：同一份inspect输出显示300张全部verified、missing/invalid为0；不据此称GPU smoke或正式训练已完成。随后为实际3060配置独立AID小样本验证。

来源：[本次回传摘要](../results/hgtnet_aid_inputs_20261104.json)、[完整原样输出与核对](../results/evidence/hgtnet_aid_inputs_return_20261104/validation.json)、[历史ZIP下载与布局](../.worktrees/sparse-hybrid-4n-top2/docs/download_aid.md)。完整SHA的历史用户确认见research_diary.md的“2026-09-20 — AID压缩包哈希确认”条目。
