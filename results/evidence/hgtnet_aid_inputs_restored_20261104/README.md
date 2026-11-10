# AID恢复与300图验证回传

2026-11-04，用户回传校验→解压→检查的完整终端输出。`reported_output.txt`保留原始20506B，SHA256 `deea789f93e2207b9b4dfb7006e22f1a556b85183ffa8853392f6b208643fc9d`；未执行附件中的命令。`inventory_reported.json`只规范化JSON，`audit_return.py`及`validation.json`完成17项复核。

旧ZIP显示`AID_visionflan_20260920.zip: OK`，原manifest与初始候选SHA继续通过；240train/60val共300图checked=verified=300，missing=invalid=0，test图片读取0。所选路径与之前缺图盘点完全相同，原Industrial重复组保留。

实际设备仍为RTX3060、12288MiB、UUID `GPU-7acce72a-93cd-e56f-35d8-5d33ecf6655e`。distribution metadata与nvidia-smi清单不替代实际Torch/CUDA或模型验证。当前输入通过属于服务器程序的用户回传；本地未回收原数据字节或直连服务器。

[当前状态JSON](../../hgtnet_aid_inputs_20261104.json)已更新为SERVER_REPORTED_AID_SMOKE_INPUTS_VERIFIED_GPU_SMOKE_PENDING，旧缺图状态备份在`before_inputs_status.json`，原缺图证据目录保留。下一步为新AID240/60一轮batch2小样本程序，本次回传尚无训练或test性能结果。
