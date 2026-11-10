# AID HGTNet正式包本地交付证据

状态：LOCAL_AID_FORMAL_BUNDLE_VALIDATED_SERVER_PREFLIGHT_AND_START_PENDING。真实3060 batch16容量已有独立工程回传，正式100轮选择/fresh refit/test仍未在服务器执行。

- [设计](../../../docs/superpowers/specs/2026-11-05-hgtnet-aid-formal-design.md)、[计划](../../../docs/superpowers/plans/2026-11-05-hgtnet-aid-formal.md)、[执行账本](progress.md)。原30类/4000train/1000val/5000pool/5000test与初始权重/config固定。
- 三个原实现子任务因账户额度中断，主代理保留其部分代码/失败日志并接手完成protocol/data、入口/强status、封装与集成；未把中断当完成。后续恢复可用的入口实现者完成两个审查问题的窄修复。
- [43项CPU协议/数据/数值/恢复测试](engine/root_integration_attempt.log)通过；包括生成RGB600字节夹具、30类小模型calibration/refit/独立重载、完整epoch恢复一致性、test事务完整复用/started无有效packet拒绝/packet先提交时不重测、非有限/45类/错误恢复状态拒绝。这些不是完整HGTNet或原AID/GPU正式证据。
- 新entry/status最初10项通过；[独立审查](review/independent_review.md)发现并关闭两个P2：被回滚的失败首步凭据影响最终验收，以及进行中status错误formal_started=false。[RED/GREEN与窄差异](entry_delivery/fix_round1/report.json)保留，14项入口/状态/恢复回归通过；完整旧失败字节被归档、已提交证明保留，不伪造重写历史。纯receipt成功夹具不反序列化Tensor或声称GPU执行。
- [7项实际交付与Git来源元数据测试](entry_delivery/provenance_green.log)通过。主代理最后以同一CPU解释器运行本次全部新formal范围suite：[64项、exit0](scoped_final_tests.log)，含上述协议/数据/engine/checkpoint/entry/status/resume/delivery。不是全项目或实际GPUsuite。旧AID15+交付6+batch16入口8项基线另行通过。
- [最终构建](integration/final_build.json)/[最终验收](final_validation.json)：85287B、SHA `ef2bc84e0df446b7d2bdb5ac2097b21456e66a97c381d8cf5d16e87a3bac5c4b`；22source、23payload、25ZIP成员。候选与最终字节相同。实际最终命名包在禁止ML实际导入的进程中[inspect](integration/final_inspect.json)/[status](integration/final_status.json)exit0，本地缺目标资产/GPU所以readyfalse/not_started。Python3.11语法、源码载荷逐字节和[43旧文件](before_identity.json)保护哈希核对一致。
- Git基础提交明确为旧058f1d8...，manifest记录source_state=uncommitted_source_snapshot；新源由文件哈希/包ID冻结，不冒充已经Git提交。运行端只读来源metadata，不执行Git。无本地/远端正式训练、安装依赖、租卡或Git提交/暂存/推送。

已知操作边界：SIGKILL或机器中断可留下独占锁；[用户说明](../../../output/hgtnet_aid_formal_3060_README_20261105.md)给出只读inspect/锁归属/GPU进程核对后才处理死锁的流程。恢复只从完整epoch，不从部分epoch或没有snapshot时暗中重启；test已开始却无完整packet时拒绝自动重测。

下一步用户上传[正式小包](../../../output/hgtnet_aid_formal_3060_20261105_v1.pyz)，执行说明中的inspect&&run。必须见到实际设备和首步451张量/五组更新通过才确认启动；正式指标与端到端时长仍待服务器产物。
