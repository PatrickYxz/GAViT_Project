> 当前状态（2026-11-06）：用户已归还此实例，旧SSH配置已停用并归档。后续分析使用已下载本地记录；若需访问持久化work，先获取新实例连接信息。下文为历史连接说明。

# 当前Featurize实例的直接SSH连接

2026-11-06已实测公钥登录、GPU查询、正式包status，以及同一训练PID的进程/GPU/日志读取，均成功退出0。当前GPU为RTX3060，UUID `GPU-91a8cd0f-e755-4d15-863a-8a5bdd741ed4`；现有训练保持原运行轨迹。本次没有使用登录密码，也没有保存密码或私钥内容。

## 连接方式

普通连接经过198.18.0.11/utun0时反复超时或关闭；项目配置使用nc将这一条连接绑定物理网卡en0，访问当时两家公网DNS一致返回的183.223.25.19:15437。已保留原host-key pin、严格主机校验和专用agent密钥；正式连接使用SSH默认密钥交换算法并已成功。全局代理和用户的~/.ssh配置均未改动。

配置：`/Users/patrick/Projects/GAViT_Project/output/featurize_gavit_ssh_20261106.conf`

手动进入实例：

```bash
ssh -F /Users/patrick/Projects/GAViT_Project/output/featurize_gavit_ssh_20261106.conf featurize-gavit
```

只读查询当前正式任务：

```bash
ssh -F /Users/patrick/Projects/GAViT_Project/output/featurize_gavit_ssh_20261106.conf featurize-gavit /environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_formal_3060_20261105_v1.pyz status
```

后续聊天中要求查看进度、读取日志或回收结果时，可直接使用上述配置。当前任务运行中，不因为连接恢复就再次run/resume或test。若以后需要周期通知，另设定时检查。

## 适用范围

- 本机密钥由SSH agent提供；Mac重启或agent清空后，可在本机运行 `ssh-add /Users/patrick/.ssh/featurize_gavit_ed25519` 解锁。口令仅在本机输入。
- 该配置绑定当前实例端口、公网IP、host-key pin及物理接口en0。换实例或网络接口变化时需复核；DNS变化时重新核实公网地址，并保持主机指纹校验。
- 已连接不等于实验完成。2026-11-06T02:22:23Z直接读取显示refit88/96，PID3136仍在运行；最终重载、test及完整完成证据待任务完成后读取。

证据：`results/evidence/featurize_ssh_connect_20261106/`；其中direct_status.json/direct_inspection.json为直接SSH命令输出，connection_check.json记录验证和失败对照，未保存任何登录秘密。
