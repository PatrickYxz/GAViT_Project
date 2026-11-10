# DFAGCN正式seed42回传验收

实验批次2026-10-30，状态SERVER_REPORTED_FORMAL_COMPLETE_SUMMARY_IDENTITIES_MATCH。

- [原始终端回传](user_summary_return.txt)、[退出记录](run.exit.json)、[完整规范化摘要](reported_summary.json)。
- [本轮验证摘要](validation.json)、[此前只有完成标记的验证](completion_marker_validation.json)。
- [独立29项核验](review/result_audit.json)、[说明](review/result_audit.md)。
- [原完成标记语义审计](review/marker_audit.md)、[对比范围审查](review/comparison_scope.md)。比较审查最初用本任务转述的摘要数字，主代理随后将实际附件归档并逐项核对精确数值。

正式OA86.53174603174604%、macro-F186.45890874660422%；OA对应21806/25200正确。选定51轮、C10；full6300 refit及25200test完成，重载误差0。wall2035.63461228秒；PyTorch阶段allocated峰值588.41MiB、reserved峰值666MiB。完整资源/参数/比较解释见[结果报告](../../dfagcn_nwpu_seed42_20261030.md)。

主代理复算runtime.json字节SHA d56d05d63aff4513f4fe733971736e430ecc40532f5ec9e45c032a72aafb1054及source_identity.json字节SHA 4b6ce21462fc18755ef568e1553e7cdaffac9208b66a6d51793a23bf1d3d351b，与六阶段报告一致；pipeline/recipe/VGG身份与本地交付源码和文件可核部分一致。

数据来自用户粘贴服务器输出，非助手直连或完整远端下载。完整逐图/逐类/混淆矩阵、history和checkpoint原件未回本地；macro-F1尚未独立复算，OA整数只是算术一致性核对。原manifest与远端Linux binary/模型字节不冒称本地独立验收。没有新训练、后验调参、Git提交推送或外发。
