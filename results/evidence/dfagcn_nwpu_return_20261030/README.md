# 真实NWPU工程检查回传核验

当前状态：SERVER_REPORTED_REAL_NWPU_SMOKE_PASS_DELIVERY_IDENTITIES_MATCH。

[最终核验](validation.json)确认用户回传exit0、completed=true、360train/90val/450pool/test0、1轮、C1、RTX4090、seed42/精度与重载误差0；pipeline源码/配方/完整VGG权重/固定manifest身份与交付包一致。另一路独立代理重算交付哈希并确认工程门禁通过。

- [规范化短摘要](compact_user_return.json)：从用户聊天转录为有效JSON，省略终端提示符和不完整外括号，没有补造字段。
- [先前粘贴原文](partial_user_return.txt)及[片段复算](partial_audit.json)：45类各2张，共90；58正确、OA64.44444444444444%、macro-F1 62.338624338624335%。partial_audit中的待确认状态保留为历史，当前结论以validation.json为准。
- [短摘要读取命令](read_compact_summary_command.txt)与[命令检查](command_check.json)：只读/过滤字段，省略混淆矩阵，不重新训练。
- [可复制结果汇总](../../dfagcn_nwpu_smoke_20261030.md)。

总26.2637秒，PyTorch峰值allocated0.5673GiB/reserved0.6777GiB。提取与缓存、校准、refit分别计时，不代替完整规模成本或账单。

来源为用户服务器终端回传，尚未直接连接服务器或逐字节下载checkpoint；对这次工程门禁无需增加大文件取回。没有正式test结果、MatConvNet等价或总体收益结论，没有追加训练或改写正式性能表。旧19个指标文件及12个旧入口源码保持。
