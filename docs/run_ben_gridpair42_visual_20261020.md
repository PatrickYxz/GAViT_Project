# BEN seed42：受训3×3／4×4同图可视化

2026-10-20，用户返回下载目录，文件已归档并核对：37个清单文件SHA、8组捕获、488条实际pre-GAT边、全部76项判断和两项run.exit=0通过。[返回文件审计](../results/evidence/ben_gridpair42_visual_return_20261020/validation.json)及[两页正式图稿/逐图分析](ben_gridpair42_visual_analysis_20261020.md)已完成。4×4纠正3个标签判断，新增1个漏判；原训练summary也已归档，K9 best10/final30和实际4090 UUID已补齐。

## 现在取回结果

**本步骤已完成，不需要再次下载、安装或捕获。** 以下保留当时的取回路径；当前结果请直接查看上面的正式图稿与分析。

本次不再执行安装或捕获命令。从Featurize文件列表下载以下一个ZIP，并返回当前对话：

```text
/home/featurize/work/ben_gridpair42_visual_20261020_01/ben_grid_pair_visual.zip
```

它包含comparison.html、captures.json、原始张量、SVG/PNG输入、run.exit/summary及原训练来源报告。本地收到后核验哈希、8组捕获及19类配对预测，再检查真实图像并制作正式图稿。现有控制台输出没有类别概率或图像，不能据此填写逐图改善/退化结果。也可先自行打开同目录comparison.html查看。

## 首次运行命令（仅留档）

以下为本次已完成运行的原始部署步骤，当前无需再次执行。

打开[单行安装及捕获命令](../results/evidence/ben_gridpair42_visual_prepare_20261020/install_and_capture_command.txt)，复制完整一行到Featurize终端执行。默认使用CPU、8线程，只复用已有两套权重对4张图做推理；不训练、不联网安装或下载、不重新解压BEN、不重跑全量test。需要此前训练使用的Python环境及持久化work文件。

结果保存到新目录：

```text
/home/featurize/work/ben_gridpair42_visual_20261020_01
```

目录已存在时会停止，保留全部旧内容。途中报错时保留run.log/run.exit，不重复安装或删除重跑；先返回报错内容。

正常结束会打印`BEN_GRID_PAIR_VISUAL_COMPLETE`、`RETURN_HTML`及`RETURN_AUDIT_ZIP`。查看完成状态：

```bash
cat /home/featurize/work/ben_gridpair42_visual_20261020_01/run.exit /home/featurize/work/ben_gridpair42_visual_20261020_01/summary.json
```

需要exit_code=0、summary status=complete、samples=4、models=[9,16]、paired_capture_checks_passed=true，以及实际HTML/ZIP文件。**下载并返回`ben_grid_pair_visual.zip`即可继续本地核对和正式图稿制作**；它包含单文件展示页、配对数据、原始张量、来源记录及训练摘要。

## 捕获与展示内容

固定沿用全部4张归档验证输入：97389、14680、28450、4439。它们来自历史诊断选样，先于本次K9/K16捕获确定，不估计总体获益率。旧seed44概率和图边不进入新的模型对照。

输入PNG来自原始RGB reader生成并Resize224后的无损保存；各PNG字节SHA、原划分val CSV SHA、对应零起始row/patch_path/19维标签均核对。直接ToTensor/ImageNet Normalize一次，同一连续FP32张量供两模型使用；无需原TIF或全量解压。归档输入和原模型记录来源见[此前原图诊断](../.worktrees/sparse-hybrid-4n-top2/results/seed44_original_feedback_diagnosis.md)。

每例展示原图、固定3×3、固定4×4；可分别开关区域覆盖、实际特征边、空间边和区域内汇总权重。箭头为neighbor→query，K9有24+18=42条pre-GAT边，K16有48+32=80条。GAT内部自环另行记录，不混入这些边；图线颜色表示边类型。汇总权重来自实际attn scorer前向并重建检查，热图以区域内均匀权重为参照，不充当类别归因或语义标签。

表格保留全部19类标签、两套概率及差值，阈值0.5下分别标记4×4纠正、4×4新增错误、两者均对、两者均错。概率差是单图变化，不是AP差；没有跨阈值时不称纠正。分区、模型预测和总体mAP分别解释。

## 权重、源码和完成检查

- 新K9/42 best SHA：`c4b77953d6dd2908d7a61ebe7a34f4004e8c219a0843de6ab05aaf207460feff`。
- 历史K16/42 best SHA：`9b9acb76d93fa8799e979a227ce1d2496573da937b9329ab0ce3ca1c3f29859e`。
- 从K9已完成目录读取冻结13文件快照及legacy_graph，验证其全部SHA；snapshot SHA为`f82d37a9cc1d18920a553fa1e4fd516e9d9ebda9d9abddb6595e12a69a944f58`。不导入当前可能有修改的研究checkout。
- 自动读取原summary，确认seed42、完整30轮、237871/122342、119825测试、train/test同一best权重、smoke重载及各阶段退出0；绑定本次已回传的全部指标。两套metadata/architecture和strict load均需通过。
- 每张图先正常前向，再开启观察hooks，再移除hooks重跑，logits最大差需≤1e-5；区域加权汇总和feedback分类独立重建也需≤1e-5。K16受训权重另检验新旧图函数前向一致性。保存实际输入、tokens、regions、图边、pooling scores/weights、feedback及logits的CPU张量。
- 记录本次软件、CPU/设备、精度、耗时、输入/权重/张量/输出SHA；原训练summary和两套metadata随ZIP归档，可补齐此前尚未回传的训练元信息。跨训练批次和单seed限制保留。

## 本地验证范围

[准备核对](../results/evidence/ben_gridpair42_visual_prepare_20261020/preparation_validation.json)及[20项相关检查日志](../results/evidence/ben_gridpair42_visual_prepare_20261020/targeted_unittest.log)保存。包含9项新增检查、已有K9扩展11项回归、真实安装解码/临时目录fixture、防覆盖、Bash/zsh语法、原始PNG及488条fixture边的SVG对应。

四张SVG已用静态渲染器转PNG并看图，**仅是明确标注FIXTURE的合成边/合成概率排版样例**，不能进入论文结果。浏览器拒绝file协议预览，因此HTML交互尚未实测，未绕过该限制；SVG导出和嵌入数据已经本地核对。

以上描述准备阶段：当时本地没有torch/timm/PyG，尚未执行真实神经网络或服务器完整导出。全套unittest运行199项，因缺torch有8项错误、缺pytest有13项错误，详细名称及trace见[全套日志](../results/evidence/ben_gridpair42_visual_prepare_20261020/full_unittest.log)；不称全套通过。现已取得真实服务器输出并完成文件/概率/图像与导出核对；本地没有重跑模型或反序列化PT。
