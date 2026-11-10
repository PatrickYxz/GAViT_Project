# HGTNet3090真实NWPU小样本工程检查

状态：**服务器回传完成，摘要身份匹配，工程门禁通过**。来源为用户终端`status`输出，本地未直连服务器读取原件；[回传记录与核验](evidence/hgtnet_3090_smoke_return_20261102/README.md)已保存。

- train360、val90、test0；seed42、batch2、一轮180个optimizer steps，best/final均epoch1。
- state=complete、completed=true、exit_code=0、training_started=true；formal_training_started=false。
- 验证OA **36.666666666666664%**，算术对应33/90正确、57错误；macro-F1 **29.33525733525733%**为服务器报告，未从逐图预测独立复算。
- wall **58.257502633秒**。该时间从stdlib预检后开始，包含实际imports、图片恢复/核验、初始化、训练、验证、保存/重载与末尾检查，不是纯训练epoch耗时或租卡账单。
- PyTorch峰值allocated **1538072576B =1466.8203125MiB =1.432441711GiB**；reserved **1688207360B =1610MiB =1.572265625GiB**。作用域为整个smoke CUDA allocator（含更新诊断和checkpoint回读），不是全GPU/进程显存，也不是batch16训练峰值。
- checkpoint回读最大误差 **0.0**。返回SHA `de78efa372cf79862859efaf13b34ff4e82c25fceb58af9a0caf84c43e7232e5`仅核验格式并保存引用；checkpoint原始字节未回收。
- 实际卡UUID `GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1`与先前用户3090清单一致。完整runtime字段未包含在短status中，软件/精度遵守交付runner的正常通过条件，不编造具体返回版本。

身份：fix1包SHA90c60dab...7c42、config canonical SHA af6ef410...ac25、manifest SHA ce0cfef2...840e、候选权重SHA68c6b09e...1a893均与交付固定值相符。注意canonical配置SHA与JSON原文件字节SHA不同。本地56项摘要/包检查及独立70项核验通过。

该结果验证独立HGTNet在真实输入上的CUDA训练、验证及持久化路径可运行，不是正式方法对比、总体性能估计或原文数值复现。仅一轮/360张训练且smoke学习率/增强/batch不同，不能把该OA放入正式比较表，也不能由分数好坏决定重跑seed或修改原定选择规则。

## 已回传history的分段耗时

[history规范化原值](evidence/hgtnet_3090_smoke_return_20261102/history_reported.json)与[8项算术核验](evidence/hgtnet_3090_smoke_return_20261102/history_validation.json)已保存。训练段40.84089300300002秒、360图/180步，对应8.81469462417376图/秒；验证段1.9847141220002413秒。二者合计42.82560712500026秒，其他smoke阶段占15.431895507999798秒。训练段含数据读取、首步诊断和更新，不能等同纯GPU计算时间。

train_loss3.7911980125639175、train_accuracy12.222222222222221%，训练在线累计预测算术对应44/360；val_loss2.315320509051283。训练accuracy在参数持续更新期间累计，validation在轮末固定模型上对不同样本计算，不能直接用12.22%与36.67%的差值判断异常或归因。

仅作为相同batch2/无增强且吞吐不变的条件算术：train5040约571.77秒，refit6300约714.72秒，均只含按已观察训练率缩放的训练段，不包括后续阶段、不同增强或IO变化。这不是batch16正式预算，也未获得租卡单价/实际费用。

下一步依据原文确认参数另准备正式seed42协议：100epoch校准上限、batch16、Adam lr1e-5/weight_decay5e-5、CE。[batch16独立容量检查包](../output/hgtnet_batch16_probe_3090_README_20261102.md)已完成本地验证，固定16张train图、1次预热+3次计时更新，不跑val/test或保存权重。真实batch16峰值/吞吐仍待当前3090执行；正式训练代码/命令尚未交付或执行。
