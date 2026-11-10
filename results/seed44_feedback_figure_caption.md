# 原图反馈案例图：已导出图稿与配套图注

依据：`pair44_feedback_k_7ehbne`原图诊断、完整终端及2026-09-19返回的真实HTML。固定展示24张入选验证图中的全部4个阈值翻转案例，顺序为(a)草地97389、(b)草地14680、(c)沙滩28450、(d)沙滩4439。四张PNG内容已检查，图边、分数与来源哈希核验通过，见[视觉分析](seed44_feedback_visual_review.md)。2026-09-19已生成并检查180 mm宽的[PDF及300 dpi PNG图稿](../output/pdf/seed44_feedback_figure/README.md)，附LaTeX片段。四张原始图像像素未变，图边/节点/文字为矢量；具体投稿模板改变尺寸后应再检查字号。

## 中文图注

BigEarthNet-19固定验证案例中的图反馈干预，使用seed44训练的GAViT。每行依次展示原始RGB输入、区域分组与空间边、区域分组与特征边。箭头来自正常前向实际捕获的有向图输入，方向为邻居到查询区域；蓝色为空间边（48条），橙色为额外特征边（32条），GAT内部自环未显示。图中比较同一已训练权重在保留图反馈与将反馈投影输出置零时的目标类别概率，阈值为0.5。(a,c)为真负例，保留反馈抑制了误报；(b,d)为真正例，保留反馈导致漏报。四例覆盖预先固定24张诊断图片中全部阈值翻转。所选案例不代表总体性能，零反馈干预不等同于重新训练的无图模型；图边本身也不构成特定边导致分类变化的因果证据。

## English caption

Graph-feedback interventions on fixed BigEarthNet-19 validation cases using GAViT trained with seed 44. Each row shows the RGB input, region assignments with spatial edges, and region assignments with feature edges. Directed edges were captured from the normal forward pass and point from a neighboring region to the query region. Blue and orange denote 48 spatial and 32 additional feature edges; internal GAT self-loops are omitted. Target-class probabilities use the same trained weights, with feedback retained or the output of the feedback projection set to zero, at a threshold of 0.5. Retained feedback corrects false positives in (a,c) but introduces false negatives in (b,d). These cases comprise all four threshold flips among 24 previously fixed diagnostic images. Selected cases do not estimate population performance; zero-feedback inference is not a retrained graph-free baseline, and the edges do not establish edge-specific causal attribution.

## 可直接使用的数值说明

| 面板 | row | 目标类 | 真实标签 | zero | normal | 观察 |
|---|---:|---|---:|---:|---:|---|
| a | 97389 | 自然草地/稀疏植被 | 0 | 0.9691 | 0.4858 | 抑制误报 |
| b | 14680 | 自然草地/稀疏植被 | 1 | 0.9170 | 0.4704 | 增加漏报 |
| c | 28450 | 沙滩/沙丘/沙地 | 0 | 0.9972 | 0.4518 | 抑制误报 |
| d | 4439 | 沙滩/沙丘/沙地 | 1 | 0.9974 | 0.2812 | 增加漏报 |

## 论文讨论段落草稿

在这些固定原图案例中，图反馈对目标类别分数的抑制表现出两种作用：它能够将真负例的预测压回阈值以下，同时也可能将真正例压成漏判。该现象提示反馈对具体样本的影响并非一致有益。不过，这批样本按诊断成败分组选择，且干预使用共同训练后的模型，不能据此推导整体mAP收益或无图模型的独立训练表现。海岸湿地的入选错例在移除反馈后仍未被纠正，测试集上该类额外退步的原因仍未由本轮案例分析确定。

真实RGB检查进一步显示，(b)包含暗绿细碎纹理与下部较浅绿色开阔带，(d)表现为左侧狭窄浅色岸带与大面积水面。固定空间区域跨越这些可见纹理边界；(d)也存在以水面为主的区域连向沿岸混合区域的特征边。这些观察使局部目标与背景干扰成为可检验假设，但不能将整支反馈置零所得变化归因到某条边，或断言某种纹理就是模型使用的分类证据。
