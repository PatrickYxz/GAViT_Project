# 正式完成标记核验

结论：可按用户回传记录“服务器报告正式全量流程完成，结果字段待回传”；无需重跑。此处未直接读取远端文件。

交付fix1包SHA256为 `79398dc4b6d703be65cd32a968d49daa6640fe6c4bbab6a33e930b312954a6c4`。包内runner、pipeline和配方均与当前源码逐字节一致，全部Python文件中该完成标记仅出现在正式runner一处。

`run_dfagcn_nwpu_formal.py:413–420` 只有在完整pipeline返回且源码/manifest/recipe/权重/二进制复核不变后，先写 `summary.json`（completed=true、formal_training_started=true）、`run.exit`（exit_code=0）和complete进度，再打印用户回传的 `DFAGCN_FORMAL_COMPLETE .../summary.json`。

该正常成功分支要求：固定5040/1260/6300/25200清单；新6300 pool缓存及train/val切片；正式train-only calibration完成九个C；fresh6300 train+val refit和重载检查通过后才提取25200 test；完整test评估返回并写test.json；best/final/model、selection、refit、线性model等必需产物存在；四份cache索引已保留在work。实际runtime门禁也在此前执行。

因此，这一行具有完整流程完成语义，可以接受为用户日志提供的服务器完成证据；尚未取得OA、macro-F1、选中epoch/C、实际GPU/runtime、耗时、峰值显存、参数及源码/缓存身份的具体值，不填写正式分数，不宣称三seed稳定性或原论文数值等价。下一步只读取已有 `run.exit` 与短 `summary.json` 即可；逐类与矩阵已由流程写入test.json，当前无需输出整份矩阵。

机器证据见 `marker_audit.json`。本轮仅做本地只读核验，无新测试、GPU/训练或远程操作。
