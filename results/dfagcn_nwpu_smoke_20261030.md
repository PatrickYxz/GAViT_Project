# DFAGCN真实NWPU工程检查回传通过

日期：2026-10-30；实现身份dfagcn_independent_v1，seed42。

## 结果

用户回传exit_code0、completed/training_started均true。train360、val90、pool450、test0；selected_epoch1、C1、保存重载最大绝对误差0。从交付pyz内部独立重算pipeline/recipe/weights身份，并核对固定manifest，20项检查全部通过。另一路只读代理核验一致。

本次真实数据/CUDA工程门禁已通过，不需要重跑或为此取回大checkpoint。来源是用户终端回传，不是直接连接服务器后的原始文件审计。

val正确58/90，OA64.44444444444444%、macro-F1 62.338624338624335%；45类各2张，逐类指标复算一致。本次epoch上限1、C网格仅[1]，不代表完整选参或正式test性能，不写入正式方法对比表。7×7、GCN256×2、MLP1024→512等仍是已披露独立选择，不证明原MatConvNet/Table I等价。

## 资源

- RTX4090，UUID GPU-d78aad24-a3ce-faa2-a189-674ccaa4d6f1，总显存24564MiB。
- Python3.11.8、Torch2.2.2+cu121、vision0.17.2+cu121、CUDA12.1/cuDNN8902、NumPy1.26.4/Pillow10.3.0，CPU8线程。
- seed42，FP32、AMP/两项TF32关闭，cuDNN benchmark关闭/deterministic开启。
- 总26.263705574秒；权重载入0.955864534秒；pool450提取+缓存6.892843317秒，约65.29图/秒；calibration3.455476798秒，subset refit4.244405087秒。
- PyTorch峰值allocated609131008字节/0.5673GiB，reserved727711744字节/0.6777GiB。这不是整进程或驱动显存测量。

以上只作本次小样本资源记录，不代替完整规模训练/端到端推理基准或账单。

## 证据与后续

[完整核验](evidence/dfagcn_nwpu_return_20261030/validation.json)、[规范化用户摘要](evidence/dfagcn_nwpu_return_20261030/compact_user_return.json)、[逐类指标](evidence/dfagcn_nwpu_return_20261030/partial_audit.json)均保留，旧19个指标和12个入口源码不变。

下一阶段转固定独立配方的NWPU seed42正式实验准备：5040/1260校准，按验证选择后重新初始化并在6300训练池refit，最终25200张test。正式命令、预算、输出身份和资源记录须在启动前确定；本轮没有启动正式训练。
