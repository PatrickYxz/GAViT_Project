# GAViT seed44 论文案例图

2026-09-19完成。展示预先固定24张验证图中全部4个阈值翻转例，顺序为草地纠正/草地漏判/沙滩纠正/沙滩漏判。无需服务器运行或再次解压数据。

## 直接使用

- [PDF图稿](gavit_seed44_feedback_cases.pdf)：单页，180 × 229.18 mm；图边、节点和文字为矢量，实际使用的字体已嵌入。推荐双栏通栏或整页插图。
- [300 dpi PNG](gavit_seed44_feedback_cases_300dpi.png)：2126 × 2707像素，适合Word排版。原RGB输入仍为224 × 224像素，导出分辨率不增加遥感细节。
- [中英文图注与讨论段落](../../../results/seed44_feedback_figure_caption.md)：图注与此图的a–d顺序对应。
- [LaTeX插图片段](figure.tex)：将PDF放在主.tex所在目录，再将片段复制进主文档；要求主文档加载graphicx。可按论文模板调整浮动位置。

图上Feedback off表示同一训练权重推理时将整个反馈投影输出置零，Feedback on表示正常前向。颜色仅表示固定区域身份，蓝/橙线分别为空间/特征边，不表示attention强度。阈值0.5；两个负例纠正误报、两个正例增加漏判，不能由此估计总体AP收益。

## 验收与来源

来源为用户返回HTML中归档的真实PNG、案例JSON和捕获SVG。四张原始RGB像素未改动，三个面板重复使用相同图像；未调整亮度、裁剪图像或生成遥感内容。SVG的实际边、路径、节点坐标、token矩形与区域透明度保留；仅调整印刷字号、线宽、箭头外观以提高可读性。

[export_qa.json](export_qa.json)记录：PDF中4张嵌入图片像素与源文件一致，12次图片放置、320条图边、392个透明度0.25的token矩形；概率与row身份匹配，实际文字字体均嵌入，字符未超出页面。最终PDF由Poppler转为PNG后逐行检查，未观察到遮挡、越界或布局重叠。该验收针对新PDF图稿，不表示旧HTML的浏览器预览限制已经解除。

[figure_manifest.json](figure_manifest.json)保留来源、每条捕获路径与导出哈希；[build_figure.py](build_figure.py)为本地图稿生成记录，仅依赖ReportLab/Pillow和项目内已归档案例。它不会运行模型、读取原始大数据集或重建kNN。重复执行只重建本目录的图稿及manifest；PNG和QA须针对新的PDF重新生成，不沿用旧记录。

在不同投稿模板中改变实际印刷尺寸后，仍需按该模板复查字号；当前图稿按180 mm宽验收。无需把这份目录上传到Featurize。
