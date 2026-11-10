# HGTNet AID seed42 正式结果（独立实现）

状态：已完成，服务器完成检查、本地重算与独立41项审计全部通过。
完成观察时间：2026-11-06 11:00:49（北京时间）。

## 可直接复制的结果

模型：HGTNet独立实现。
数据：AID，30类，固定seed42划分。
校准训练4000张；验证1000张；refit池5000张；最终test5000张。
测试Accuracy：96.92%（4846/5000正确，154个错误）。
测试macro-F1：96.77286814931982%。
最佳验证Accuracy：97.20%；最佳验证macro-F1：97.0745569519779%。
验证选择第96轮；calibration完整运行100轮；fresh refit运行96轮。
batch size：16；可训练参数：74612163；接受轨迹总更新：55048步。
checkpoint独立重载最大误差：0.0。
结果定位：固定配方下的一次seed42独立实现结果，不能据此声称官方复现、多seed稳定提升或统一backbone公平比较。

## 恢复与资源范围

原实例在refit第76轮中断；新RTX3060从第75轮完整快照/48475步恢复，保留原历史，重算180次未提交更新，连续完成至第96轮。校准和refit初始状态SHA相同；最终test事务complete，test_reused=false，没有重新运行独立评估。

恢复后的attempt2墙钟：4986.588285893秒，约83.11分钟。
attempt2峰值allocated：4731174400字节，约4.406GiB。
attempt2峰值reserved：5175771136字节，约4.820GiB。
以上耗时和峰值仅覆盖恢复后的本次调用，不能当作整个实验总耗时、全程峰值或账单。完整总耗时/费用未获得。

当前GPU：RTX3060，GPU-91a8cd0f-e755-4d15-863a-8a5bdd741ed4。
代码：Git base 058f1d8b79d0be538ec358eb742d2cd3992921d4，uncommitted_source_snapshot；22项源文件/config哈希与下载身份记录一致。原冻结划分、预处理和优化配方保留；完整recipe见结果JSON及原始summary/binding。

## 保存位置

本地结果：/Users/patrick/Projects/GAViT_Project/results/hgtnet_aid_seed42_20261106.json
独立AID结果表：/Users/patrick/Projects/GAViT_Project/results/external_baselines_aid_seed42_20261106.csv
本地原始记录：/Users/patrick/Projects/GAViT_Project/results/evidence/hgtnet_aid_formal_complete_20261106/raw/
本地记录包：/Users/patrick/Projects/GAViT_Project/results/evidence/hgtnet_aid_formal_complete_20261106/completed_run_records.tar

46份服务器原始记录及逐文件SHA已核对，包括5000条有序test预测、30类逐类指标/混淆矩阵、100+96轮history、原manifest、test事务、模型重载、三份checkpoint收据、两次attempt日志和完成标记。服务器strong status已核对三个模型二进制SHA；大型.pt保留在持久化work，本地未下载或加载模型二进制。

服务器目录：/home/featurize/work/hgtnet_aid_formal_3060_20261105_v1/runs/aid_seed42_01
最终模型：该目录下final_refit.pt。
最终模型SHA256：4de7bdfbbc6afdf91ee3dc59b0c084b36d0161a07d1457b0c9f56cd6e0504aa2
测试文件SHA256：6d35db2feaef0864c0f3181002164e8e999c84685f01064af3cd83c6306e7a0b

完成证据：status的completed/state/exit_code/source_identity_matches/completion_evidence_valid全部通过；run.exit=0，completion.marker及日志footer均为HGTNET_AID_FORMAL_COMPLETE，原PID3136和GPU计算进程已退出。本地监控已结束，后备heartbeat hgtnet-aid已设为PAUSED。
