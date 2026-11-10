# BEN 3×3／4×4固定区域：第一批图稿

2026-10-17。复用4张本地归档BEN验证RGB输入，只生成真实位置分配规则的覆盖层。没有GPU、训练、模型推理、新预测或图边。

- [单页PDF](ben_grid3_grid4_cases.pdf)：180×254 mm，底图以原生224×224像素嵌入，覆盖层与文字为矢量。
- [300dpi全图PNG](ben_grid3_grid4_300dpi.png)：2126×3000像素，导出分辨率不增加遥感细节。
- [中文逐例分析与中英文图注](../../../docs/ben_grid3_grid4_analysis_20261017.md)。
- [图稿来源、分配矩阵与SHA](figure_manifest.json)、[导出核验](export_qa.json)。

| 案例 | 独立PNG | 可编辑SVG |
|---|---|---|
| row97389 | [对照图](row_97389_grid3_grid4.png) | [SVG](row_97389_grid3_grid4.svg) |
| row14680 | [对照图](row_14680_grid3_grid4.png) | [SVG](row_14680_grid3_grid4.svg) |
| row28450 | [对照图](row_28450_grid3_grid4.png) | [SVG](row_28450_grid3_grid4.svg) |
| row4439 | [对照图](row_4439_grid3_grid4.png) | [SVG](row_4439_grid3_grid4.svg) |

源PNG字节不变，三列嵌入相同底图；区域覆盖是额外的矢量层。叠加透明度一致为0.18，未增强亮度或改变裁剪。4×4成员与归档真实捕获吻合，3×3是由位置规则计算的几何参照，不是受训K9前向。四例是历史验证诊断子集，不能作为代表性抽样或新四seed性能证据。

`build_figure.py`保存PDF/SVG构建过程；`verify_figure.py`核对来源、原始嵌入像素、独立整数分配、PDF矩形坐标和范围标注。重建后需重新渲染PNG并做视觉检查，不能沿用旧QA。
