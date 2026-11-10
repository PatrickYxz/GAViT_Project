# NWPU FFN seed42 GPU 短测回传验收 — 2026-10-04

来源：用户复制服务器的两行精简摘要，已按字段归档为[server_return.json](evidence/nwpu_ffn_smoke_20261004/server_return.json)，[本地一致性检查](evidence/nwpu_ffn_smoke_20261004/validation.json)通过。此前被截断的长输出保留在同目录。以下是回传证据核对，不是本地独立GPU复跑或checkpoint文件重哈希。

| 项目 | Swin+FFN | soft GAViT+FFN |
|---|---:|---:|
| 退出 / 状态 | 0 / complete | 0 / complete |
| 阶段 / seed | smoke / 42 | smoke / 42 |
| train / val / epoch | 360 / 90 / 1 | 360 / 90 / 1 |
| 总耗时（秒） | 108.947310 | 116.192555 |
| 峰值allocated显存（GiB，回传已取整） | 3.565 | 3.667 |
| 保存后重载最大logit差 | 0 | 2.3841858e-7 |
| FFN新增参数 | 297408 | 297408 |
| 初始残差最大绝对值 | 0 | 0 |
| 第一步 fc1 / norm / fc2 梯度绝对和 | 0 / 0 / 427.378943 | 0 / 0 / 286.331063 |
| 第二步 fc1 / norm / fc2 梯度绝对和 | 13.612891 / 0.184172 / 496.065482 | 13.282884 / 0.157888 / 334.527931 |
| 实际图前向 | 不适用 | 48空间+32特征=80边，通过 |

两组报告同为RTX2080Ti、干净058f1d8，对应已发布058f1d8b79d0be538ec358eb742d2cd3992921d4。初始FFN SHA相同：566b722f15ccd6b8bc927d96d73a426f582bc320f5216eedcd1b2b13f45944ca；各自selected/final相同且与初始不同，具体哈希见归档。初次捕获形状分别32×49×768和8×49×768；soft的8来自固定验证探针触发首次前向，不代表训练batch从32改为8（对应入口先运行probe.record(0)再训练）。第一步fc1/norm为0符合fc2零初始化，第二步三组梯度均非零。

已证明真实GPU下该短测运行、反传、更新、保存重载链路的回传指标一致；尚不能说明正式效果改善、稳定性或因果根因。短摘要没有完整history训练吞吐、reserved显存、硬件UUID/软件包或checkpoint文件SHA；不从末尾未署名36.67%片段推算某模型效果，不将FFN张量SHA当checkpoint文件SHA。后续校准启动命令将再次读取实际best文件核对hash，保留完整输入/发布身份检查。

下一步仅[双模型30轮校准](../docs/run_nwpu_ffn_calibration_20261004.md)，当前待服务器启动，正式val/test指标均待测试。新增容量能否改善soft相对同样加FFN的Swin，将由冻结四格比较回答。旧同卡两模型校准共2877.894717秒只是预算参考；本次smoke总耗时225.139865秒含预检/保存，不直接外推为30轮时间。
