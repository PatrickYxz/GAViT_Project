# NWPU seed43 校准进度（2026-10-11）

**2026-10-11最终结果已回传：** calibrate/refit/test全部完成，总exit0；实际选轮Swin23/fixed26/soft29，25200test OA93.857143/93.765873/93.626984%，macro-F1 93.834530/93.743513/93.587821%。固定与soft均低于Swin，见[正式报告](nwpu_seed43_test_comparison.md)与[42/43双seed表](nwpu_two_seed_summary.md)。BigEarth soft43已预检PASS，下一步工程短测。下方单阶段及等待摘要文字仅为历史，不是当前状态。

**后续九阶段已回传完成：** 三模型calibrate/refit/test均通过调度验收并打印总完成标记，正式数值/选轮仍待紧凑摘要；见[最新回传](evidence/seed43_nwpu_queue_complete_20261011/summary.json)。下方是先前单阶段记录，不是当前未完成状态。

正式入口已有真实RTX 4090首batch及Swin30轮训练回传，Swin校准已打印完成。九阶段队列当前所处阶段、阶段验收exit和后续模型尚未回传；不据单阶段日志判定整个队列完成。

| 模型 | 已回传阶段 | 最大val OA% | 最大值首次epoch | macro-F1 / checkpoint / 后续 |
|---|---|---:|---:|---|
| Swin | calibrate30/30，完成标记 | 93.0952380952381 | 23（由完整打印曲线及首个同分规则确定，待metadata核对） | 待完整report；refit/test未回传 |
| 固定4×4 GAViT | 未回传 | 待测试 | 待确认 | 队列总日志待核验 |
| soft_prototype GAViT | 未回传 | 待测试 | 待确认 | 队列总日志待核验 |

Swin最高值在23/26/28/29/30轮打印持平，final epoch30的val OA同为93.0952380952381%，最终train loss0.001668。30轮纯train时间打印合计335.83秒，均值11.1943秒/轮；验证、检查、存盘与IO不在这个合计内，不能当阶段总墙钟或九阶段预算。

本批依旧使用先5040/1260校准、各自最高val OA首轮选轮、6300pool重新初始化refit、25200test的固定协议。当前93.0952%只对应校准val，不是正式test，不据它判断seed43三模型胜负或与seed42最终test直接对比。[原始回传](evidence/seed43_nwpu_swin_calibrate_20261011/server_return.txt)、[完整打印曲线](evidence/seed43_nwpu_swin_calibrate_20261011/history_from_printed_log.csv)、[解析核对及边界](evidence/seed43_nwpu_swin_calibrate_20261011/summary.json)。

下一步只读队列run.log核对STAGE_COMPLETE swin_calibrate及START fixed_calibrate；当前tail固定在已结束的swin_calibrate.process.log，不会自动切换。Ctrl+C只退出tail，后台队列保持。无需重复启动、追加seed或停止现有模型训练。BigEarth soft43仍等本批结束后单独预检。
