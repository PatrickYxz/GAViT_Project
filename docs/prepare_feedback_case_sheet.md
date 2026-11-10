# 将原图诊断整理成一份独立HTML

状态：2026-09-19。用户已返回真实 `feedback_cases.html`；四张PNG内容、四份案例JSON、原始及筛选SVG已离线检查并归档，见[视觉分析](../results/seed44_feedback_visual_review.md)。后续已完成[PDF/PNG论文图稿](../output/pdf/seed44_feedback_figure/README.md)及渲染验收，W1-08完成。不需要再次执行服务器命令。

## 当前目的

把已固定24张原图中全部4个阈值翻转案例整理为同一展示页：草地97389（纠正误报）/14680（增加漏报），沙滩28450（纠正误报）/4439（增加漏报）。每例三幅并排图：原始224×224 RGB输入、区域+48条空间边、区域+32条特征边；下方列出真实标签、zero/normal分数和阈值0.5。

只读取 `pair44_feedback_k_7ehbne` 已保存的run.json、loaded_model.json和4例PNG/SVG/JSON，不读大型原数据、不加载Torch、不运行模型、不重新计算图边。两种边视图只是从捕获的原SVG中筛选边，保留原路径与方向；内部GAT自环不画，颜色/边不能当作attention或因果归因。

## 在服务器执行（保留复现说明，本轮已完成）

打开本地 [可粘贴命令](paste_feedback_case_sheet.sh)，复制从第一行 `python - <<PY` 到最后单独一行 `PY` 的全部文本，粘贴到服务器正常Shell提示符后执行。命令会在新建的 `work/pair44_case_sheet_随机后缀` 保存脚本及结果，不需要上传工具包，不覆盖原诊断。

它会先核对original/complete、24张源清单、GAViT44 checkpoint身份、4例标签/行号/分数/缓存误差、PNG224尺寸，以及SVG边与JSON边一致。任何缺失或不一致都会停止，不用占位图替代，也不回退JPEG。

成功打印：

```text
CASE SHEET COMPLETE: /home/featurize/work/pair44_case_sheet_随机后缀/view/feedback_cases.html
4 original cases; 12 panels; no inference. Open or download only feedback_cases.html.
RENDER_EXIT 0
```

在服务器文件列表找到打印出的 `feedback_cases.html`。它内嵌4张PNG、图层、逐例JSON与来源哈希，只需要打开或下载这一个文件，不需要同时下载images目录或PT张量。若平台预览屏蔽HTML中的内嵌图片，下载这一份后在本地浏览器打开。

本次已用返回的单文件完成实际图片检查；后续讨论可直接使用本地归档，无需再传相同HTML、数值日志或完整数据集。

## 核验与完成边界

- 6项新行为检查覆盖独立资源嵌入、实际边过滤、拒绝gallery/失败运行、身份/标签/缓存不一致、缺图、SVG边漂移、独立CLI和拒绝覆盖；使用清楚标注的合成PNG与图数据作为测试夹具。
- 新旧诊断相关测试共33项通过；Bash与zsh均实际执行粘贴入口的本地夹具版本，核对脚本按字节还原、4例HTML、输出哈希与RENDER_EXIT0。该验证不使用服务器真实图像。
- 原诊断行为不改动；已有4例与数值结论见 [原图报告](../results/seed44_original_feedback_diagnosis.md)。
- 本地浏览器对 `file://` 预览的访问被URL安全策略拒绝；未绕过，浏览器页面排版未验收。真实返回文件通过离线解析提取未改动PNG进行图片内容检查，并核对SVG边与节点坐标；这不等于浏览器排版或论文导出验收。
- [中英文图注](../results/seed44_feedback_figure_caption.md)已结合真实观察更新。新PDF图稿的矢量图边、字体、PNG像素及导出页面已核验，W1-08完成；此结论不替代旧HTML浏览器预览状态。
