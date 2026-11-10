# AID HGTNet smoke：3060完成回传，最终status待读取

更新：用户已补回[status原文](status_reported_output.txt)/[JSON](status_reported.json)，[10项核对](status_validation.json)通过。state complete、exit0、完成产物/来源检查true；val Accuracy76.66666666666667%、macro-F1 72.34920634920634%，重载0，allocated1536937984B（1.431385GiB）/reserved1694498816B（1.578125GiB），wall48.76537990199995秒；checkpoint SHA `9d2040b95eb27756f8f82f26c3b49e05fd787849776e45fa7df9088a8a076910`。原pending记录快照为[record_before_status.json](record_before_status.json)。这些是服务器程序用户回传，不是下载原预测或checkpoint后的本地独立数值复验。下一阶段准备隔离的batch16资源门禁，以下保留首次COMPLETE回传时的边界。

2026-11-05。来源为用户直接粘贴的终端输出，未直连服务器或下载服务器产物。

- [原样转录](reported_output.txt)：2942字节，SHA256 `2bdfad60029e3653f0f7b90c4482cb96f0907fcc47b5e0ad138e5c478bdb3bc3`；哈希标识本地转录文件，不是服务器run.log的哈希。
- [事件解析](parsed_report.json)、[17项核对](validation.json)及[当前结构化记录](../../hgtnet_aid_smoke_3060_20261105.json)已保存。仅标准库核对回传字段、复用当前runtime/容量检查函数，并验证17个源码/配置与已交付包逐字节一致；没有重新训练或运行GPU检查。
- 旧ZIP完整SHA检查回传OK，现有包无手填旧UUID，绑定实际RTX3060 `GPU-df386e6c-dbb3-0952-8190-3265a7bd8ae8`。实际Python3.11.8、Torch2.2.2+cu121、vision0.17.2+cu121、CUDA12.1/cuDNN8902及FP32/AMP和双TF32关闭均通过当前严格检查函数。
- 首步loss有限，五组共451个梯度张量无缺失/非有限值，各组梯度L2为正，各组代表参数实际更新。不能由组L2或代表参数检查推出每个张量均非零或每个参数均已更新。随后30/60/90/120步对应60/120/180/240训练样本，最后回传COMPLETE及预期summary路径。
- 冻结工程范围为30类、seed42、1epoch、batch2、240train/60val/test0。源码正常完成路径要求60条有序验证、独立checkpoint重载误差≤1e-6、输入/源码/300张选图及checkpoint末尾复核、一致的完成产物和run.exit=0，之后才打印完成标记。这是源码条件背书，未替代服务器产物字节验收。
- CUDA总容量12487622656字节、NVML12288MiB，接口容量差378.875MiB；这些都不是峰值显存。精确验证Accuracy/macro-F1、checkpoint SHA、重载误差、峰值allocated/reserved和总耗时仍为空，等待现有status回传。
- [此前失败状态快照](preflight_before_success.json)保留；原失败记录只新增后续已解决链接，未把失败尝试改成已训练。包仍为63235字节、SHA256 `a70b6fbf038aa82daac9398b1d75c057d37b4fe3bf18167604b95d6f6a9d3961`；研究成绩表不写入smoke指标。

下一条命令只读取现有结果：

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_smoke_3060_20261104.pyz status
```

收到完整JSON后核对`state=complete`、`exit_code=0`、`completion_evidence_valid=true`、`source_identity_matches=true`及具体指标/资源，再准备独立正式AID配置与预算。
