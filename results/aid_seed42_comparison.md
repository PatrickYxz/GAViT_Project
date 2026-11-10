# AID seed42正式实验

09-22后续进展：seed43/44均已完成，最新结论见[三seed主表](aid_multiseed_summary.md)；下一步只读已有曲线/混淆矩阵，之后复用权重补效率。下文保留seed42当时记录。

2026-09-21更新：两模型校准、refit及5000test均已完成且exit0；配对计数和30类分析已验收。最终test OA差+0.60个百分点、macro-F1差+0.56024201486908个百分点，纠正93/新增错误63；Commercial/Park净增23/7张，其余28类增减合计0。单seed稳定性尚未建立，下一步[seed43重复实验](../docs/run_aid_seed43.md)。详见[逐类报告](aid_seed42_per_class_analysis.md)。

| 模型 | 校准阶段（4000 train / 1000 val） | 重训阶段（5000 pool） | 最终test OA / macro-F1（5000 test） |
| --- | --- | --- | --- |
| Swin | complete / exit0；30轮；best epoch21、val OA96.9%；末轮val OA96.3% | complete / exit0；5000张、21轮；T_max仍30；last.pth重载差0 | 96.34% / 96.22088514124282%；complete / exit0 |
| GAViT | complete / exit0；30轮；best epoch20、val OA97.7%；末轮val OA97.1% | complete / exit0；5000张、20轮；T_max仍30；last.pth重载差4.768e-7 | 96.94% / 96.7811271561119%；complete / exit0 |

| 最终test指标 | Swin | GAViT | GAViT − Swin |
| --- | ---: | ---: | ---: |
| OA（%） | 96.34 | 96.94 | +0.60 pp |
| macro-F1（%） | 96.22088514124282 | 96.7811271561119 | +0.56024201486908 pp |
| 正确图片数（由OA×5000推得） | 4817 | 4847 | +30 |
| 错误图片数（由5000−正确数推得） | 183 | 153 | −30 |

计算保存于[汇总JSON](aid_seed42_test_summary.json)。配对结果已确认4754张两者都对、93张Swin错/GAViT对、63张Swin对/GAViT错、90张两者都错；30类支持数/加权准确率/宏平均F1均重现总指标。完整混淆矩阵与逐样本原始CSV仍在服务器。不能用这一次结果宣称稳定提升或统计显著，不能将较短的GAViT refit总时间解释为更快推理。

## Swin校准实测

- RTX2080Ti；Git deb2b45 / dirty=false；seed42；batch32、workers2；全量train4000/val1000。架构、配方、硬件、manifest、预训练及17个源码哈希与已通过的Swin短测一致。
- best epoch21 / val OA96.9%；final epoch30 / val OA96.3%、loss0.000756（来自前次末尾日志）。status complete、exit_code0、保存重载logit最大差0。macro-F1、逐类指标和混淆矩阵未回传，不补填。
- 全程1281.2088429679998秒（约21分21秒）；可见epoch21—30训练耗时24.55—31.20秒/轮，只含训练循环。全量history尚未回传，不把全程平均或末十轮耗时当完整30轮训练/验证统计。
- CUDA峰值allocated 3820576256字节（3.5582GiB）/ reserved 4112515072字节（3.8301GiB），范围含保存重载；主进程RSS峰值1549766656字节（1.4433GiB）；参数27543960均可训练。费用待提供。
- manifest SHA256 `30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22`；预训练SHA256 `f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`。
- 最佳checkpoint `/home/featurize/work/aid_swin_calibrate42_20260920_01/best.pth`，SHA256 `3542b3b5cd40fbf1ea44881f284ab0d997322e9e2ed01687341fc2300bb8e939`。日志、最后checkpoint、metadata及其他完整指标按入口保存在同一运行目录，未直接下载核查全部文件。
- [完整报告规范化副本](evidence/aid_swin_calibrate42_20260920_01/run.json)、[退出码](evidence/aid_swin_calibrate42_20260920_01/run.exit)、[末十轮日志](evidence/aid_swin_calibrate42_20260920_01/reported_tail.log)已归档。副本恢复了聊天转义下划线及__init__.py拼写，并记录[来源说明](evidence/aid_swin_calibrate42_20260920_01/provenance.json)；不声称与服务器run.json逐字节一致，不用于替代服务器refit身份校验。

Swin/GAViT均已从相同预训练重新初始化，在5000训练池分别完成21/20轮，scheduler T_max仍30，后续使用各自末轮权重测试。两模型轮数均按事先确定的自身val OA最高规则选定，不依据test改变。

## GAViT校准实测

- RTX2080Ti；Git deb2b45 / dirty=false；seed42；train4000/val1000、batch32/workers2。与Swin的manifest、预训练、训练配置、硬件环境、代码身份和17个源码哈希一致；架构与已通过的GAViT smoke一致。
- best epoch20 / val OA97.7%；final epoch30 / val OA97.1%、loss0.000893（来自此前末段日志）。status complete、exit_code0；真实前向48空间边+32特征边=80，actual_forward_checked=true；checkpoint重载logit最大差4.76837158203125e-07，低于1e-5。
- 全程1366.7442648819997秒（约22分47秒）；可见epoch27—30训练耗时25.03—25.42秒/轮。CUDA峰值allocated3924388864字节（3.6549GiB）/ reserved4162846720字节（3.8770GiB），含保存重建；主进程RSS1787305984字节（1.6646GiB）；参数31375385均可训练。全程含数据扫描等，不能据两次wall time差直接报告推理速度差。完整history、macro-F1/per-class/confusion及费用未回传。
- 最佳checkpoint `/home/featurize/work/aid_gavit_calibrate42_20260920_01/best.pth`，SHA256 `6a1a99e337b7f810b60bce86308ec23ca52c7a327ec7a8e9a9c4e39ebbfff7eb`。服务器其他产物未直接下载核查。
- [完整报告规范化副本](evidence/aid_gavit_calibrate42_20260920_01/run.json)、[退出码](evidence/aid_gavit_calibrate42_20260920_01/run.exit)、[来源说明](evidence/aid_gavit_calibrate42_20260920_01/provenance.json)及[末段日志](evidence/aid_gavit_calibrate42_20260920_01/reported_tail.log)已归档。规范化副本不是服务器run.json逐字节副本，不替换服务器refit来源文件。
- 使用发布代码的validate_calibration进行本地只读配置核对：两模型refit配置与各自报告一致，分别返回21/20轮。该检查未加载服务器checkpoint或原图，完整来源/数据校验仍由服务器refit入口完成；未启动本地或远端训练。

当前证据支持按冻结协议进入最终测试。+0.8个百分点不作为最终test增益或多seed稳定性结论；主测试比较表保持待测试。

## Swin最终拟合实测

- 用户回传EPOCH 16—21/21、SCENE REFIT COMPLETE及完整run.json/run.exit；status complete、exit_code0、train5000/val0、final_epoch21。末轮loss0.003510，val_OA=None、best_epoch/best_val均null符合固定预算refit设计，不在训练池重新选最佳轮。
- 与Swin校准逐项核对architecture、code、hardware、manifest、预训练、seed、全部training、参数量与数据集一致；校准来源best checkpoint SHA匹配。17个源码哈希与当前干净发布deb2b45一致。refit_source.run_json_sha256为服务器报告的`4ca9a57bacb5436c1912823f95250fb980e68123cfe623b1742df9fda0964148`，不与经过重新格式化的本地校准JSON进行字节哈希比较。
- RTX2080Ti，全程892.877999542秒（约14分53秒）；可见epoch16—21训练耗时30.61—31.26秒/轮，完整history未回传。CUDA峰值allocated3820575744字节（3.5582GiB）/reserved4112515072字节（3.8301GiB），含保存重建；主进程RSS1504518144字节（1.4012GiB）。费用待提供。
- 最终权重`/home/featurize/work/aid_swin_refit42_20260920_01/last.pth`，SHA256 `fc04ddd58e233bc7291b6a5e54a2c3e9efc45c0584a709aa28a0eba496178018`；重载logit最大差0。该权重尚未进行最终测试，没有test OA或macro-F1。
- [原始回传文本](evidence/aid_swin_refit42_20260920_01/server_output.txt)、[提取报告](evidence/aid_swin_refit42_20260920_01/run.json)、[退出码](evidence/aid_swin_refit42_20260920_01/run.exit)及[来源说明](evidence/aid_swin_refit42_20260920_01/provenance.json)已归档；checkpoint未直接下载核查。

## GAViT最终拟合实测

- 用户回传EPOCH16—20/20、SCENE REFIT COMPLETE、完整run.json/run.exit，status complete、exit_code0、train5000/val0、final_epoch20。末轮loss0.006303，val_OA=None、best_epoch/best_val=null符合固定预算拟合。实际图48空间边+32特征边=80、actual_forward_checked=true，重载logit最大差4.76837158203125e-07，低于1e-5。
- 与自身校准核对架构、图捕获、参数量、数据集及全部冻结身份一致；与Swin refit核对代码、硬件环境、manifest、预训练、seed、training、counts及阶段一致。17个源码哈希匹配干净发布deb2b45；来源校准best checkpoint SHA一致。服务器报告来源run.json SHA为`ede0e4769d62eb36c23bad81b689072db8ca547fc142ec2046fea00054c97263`，不与本地格式化副本作字节比较。
- RTX2080Ti，全程819.7612688440004秒（约13分40秒）；可见末五轮train_s30.66—31.86秒。CUDA峰值allocated3925145088字节（3.6556GiB）/reserved4162846720字节（3.8770GiB），含重建；主进程RSS1776668672字节（1.6547GiB）。Swin/GAViT总轮数不同且wall time包含扫描/I/O，不能由总耗时反推GAViT更快；完整history、标准化推理效率与费用待提供或测量。
- 最终权重`/home/featurize/work/aid_gavit_refit42_20260920_01/last.pth`，SHA256 `a023563e7c4566436660a2d9bc778a9adb3e9b5a64f39f821ce81e909c913c7a`。两模型最终test均待运行。
- [原始回传文本](evidence/aid_gavit_refit42_20260920_01/server_output.txt)、[提取报告](evidence/aid_gavit_refit42_20260920_01/run.json)、[退出码](evidence/aid_gavit_refit42_20260920_01/run.exit)及[来源说明](evidence/aid_gavit_refit42_20260920_01/provenance.json)已归档，checkpoint未直接下载。

## Swin最终测试回传

- 5000张test，OA96.34%、macro-F1 96.22088514124282%，training_phase=refit、fit_scope=train_plus_val、status complete。这是完整5000训练池拟合后权重的测试结果，与4000/1000校准验证结果分开记录；不能直接把两者差值解释成同一checkpoint的泛化差距。
- checkpoint SHA为`fc04ddd58e233bc7291b6a5e54a2c3e9efc45c0584a709aa28a0eba496178018`，manifest SHA、17个源码哈希、Git deb2b45/dirty=false及RTX2080Ti环境均匹配Swin refit；batch32/workers2/float32/TF32关闭/pin_memory=false，命令明确split=test且使用last.pth。
- [报告规范化副本](evidence/aid_swin_test42_20260921_01/run.json)、[指标摘要](evidence/aid_swin_test42_20260921_01/metrics_summary.json)与[来源说明](evidence/aid_swin_test42_20260921_01/provenance.json)已保存。用户在随后GAViT回传中明确补贴Swin exit_code=0，[退出码](evidence/aid_swin_test42_20260921_01/run.exit)已归档；完整metrics.json、逐类/混淆矩阵和预测尚未回传，当前摘要不替代完整产物。

## GAViT最终测试回传与配对结论

- 用户回传完整数据扫描10000/10000（113.2秒）、SCENE EVALUATION COMPLETE及双方exit_code0；评估报告status complete、samples5000、split test。GAViT OA96.94%、macro-F1 96.7811271561119%，refit/train_plus_val，checkpoint SHA `a023563e7c4566436660a2d9bc778a9adb3e9b5a64f39f821ce81e909c913c7a`与已验收refit匹配。
- 与Swin评估核对样本数、manifest、code（17个源码哈希及deb2b45/dirty=false）、RTX2080Ti软硬件、batch32/workers2、float32/TF32关闭/pin_memory=false一致，argv只有模型来源权重及输出目录不同；两模型来自同一预训练和已冻结选轮规则。扫描113.2秒不是测试推理耗时。
- [GAViT报告](evidence/aid_gavit_test42_20260921_01/run.json)、[指标摘要](evidence/aid_gavit_test42_20260921_01/metrics_summary.json)、[退出码](evidence/aid_gavit_test42_20260921_01/run.exit)、[回传末段](evidence/aid_gavit_test42_20260921_01/reported_tail.log)与[来源说明](evidence/aid_gavit_test42_20260921_01/provenance.json)已归档。恢复聊天转义并核对复用相同评估字段，不是服务器文件字节副本；未下载checkpoint、完整metrics或预测。
- 本次单seed支持GAViT在固定AID协议上取得小幅正向测试差。下一步仅从持久化预测读取逐样本纠错/新增错误、30类准确率/F1，不追加GPU训练；按路径配对、检查重复/缺失和相同真值，再核对总体OA。多seed重复性、完整效率benchmark与既定外部方法/分组消融仍未完成。
