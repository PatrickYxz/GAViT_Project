# AID恢复：原图重建、新3060证明与第75轮快照续训

已收到当前namespace内ps9450无行、锁PID9450/目标输出一致，以及用户明确确认“旧实例已关闭，只剩当前实例”。服务器inspect --resume已通过第75轮refit/generation175恢复文件的归属与完整SHA检查；897033192B，SHA `bb7c8bb4f2115bd1b384f9a86f47688cf96fada39b3e3ad3ad27ad347be5cefc`。这尚不替代resume对模型Tensor/Adam/选择状态的严格载入验证。

当前阻塞为本地data不存在，以及GPU从df386e6c变为`GPU-91a8cd0f-e755-4d15-863a-8a5bdd741ed4`，旧容量证明不匹配。原manifest、22项源/配置和初始化仍一致；正式checkpoint和历史输出保留。以下不改变正式程序或binding，不删除checkpoint。

先恢复原ZIP并对新卡只做4次一次性更新。原图解压到实例本地data；新探针输出为独立目录，保留原结果：

```sh
printf '%s\n' 'cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7  /home/featurize/work/AID_visionflan_20260920.zip' | sha256sum -c - && unzip -n -q /home/featurize/work/AID_visionflan_20260920.zip -d /home/featurize/data && /environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_batch16_probe_3060_20261105.pyz run --output /home/featurize/work/hgtnet_aid_batch16_probe_3060_20261105/runs/batch16_seed42_gpu91a8cd0f_01
```

只有前一行成功且出现HGTNET_AID_BATCH16_PROBE_COMPLETE后，执行下一行：把已确认旧实例关闭/PID9450不存在的旧锁改名保存，然后后台resume。新容量路径不进入数值binding，原正式包仍先严格检查5000pool、同当前GPU证明、磁盘/主机RAM、来源与快照，再继续训练。

```sh
mv -n /home/featurize/work/hgtnet_aid_formal_3060_20261105_v1/runs/aid_seed42_01.lock /home/featurize/work/hgtnet_aid_formal_3060_20261105_v1/runs/aid_seed42_01.lock.closed_pid9450_20261106 && (nohup /environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_formal_3060_20261105_v1.pyz resume --capacity-output /home/featurize/work/hgtnet_aid_batch16_probe_3060_20261105/runs/batch16_seed42_gpu91a8cd0f_01 >> /home/featurize/work/hgtnet_aid_formal_3060_20261105_v1/runs/aid_seed42_01/resume_launch_20261106.log 2>&1 < /dev/null &)
```

该命令返回终端提示符只表示后台启动请求已发出。载入完整模型/Adam快照可能需要一段时间；需要下面日志出现resumed及新first_optimizer_step，并结合GPU进程，才能确认续训首批成功。实例须保持开启。

首条CUDA输出前还会静默校验5000张pool图、初值、两个大快照的文件SHA及容量证明；刚发出命令时日志为空、GPU无占用，可能仍处CPU预检。等待一会儿后用下面命令确认后台进程，不重复启动：

```sh
pgrep -af hgtnet_aid_formal_3060_20261105_v1.pyz
```

```sh
tail -n 15 /home/featurize/work/hgtnet_aid_formal_3060_20261105_v1/runs/aid_seed42_01/resume_launch_20261106.log
```

```sh
nvidia-smi
```

把这两项输出贴回。若前序失败、提示已有探针输出/锁、或resume报告blockers，保留输出，不反复执行run或覆盖目录。可用只读预检查看具体原因：

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_formal_3060_20261105_v1.pyz inspect --resume --capacity-output /home/featurize/work/hgtnet_aid_batch16_probe_3060_20261105/runs/batch16_seed42_gpu91a8cd0f_01
```

当前work可用6478102528B，比现有6GiB门槛仅多34MiB；恢复原图使用data，探针只写小型JSON/日志。若容量门禁再次不通过，先查实际占用，不删除恢复点或降低门禁。第75轮checkpoint若严格载入成功，将从第76轮重做当前未提交部分，继续到选中96轮，再独立重载和最终test。旧180次部分更新保留为历史证据，不混入接受轨迹；最终结果仍待完成验收。
