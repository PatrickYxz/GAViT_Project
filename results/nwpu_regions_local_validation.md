# NWPU 固定/软分配可视化：本地验收

2026-09-29 后续：服务器真实90图已经完成，见[服务器完成记录](nwpu_regions_server_completion.md)。以下保留本地验收时的证据范围。

2026-09-29。用户确认90张验证图方案并在实施中要求继续。新增独立工具`tools/visualize_nwpu_regions.py`、15项行为检查及[服务器单行命令](../docs/run_nwpu_regions_20260929.md)。没有修改模型、训练入口或原划分，没有提交/推送、服务器连接或新增训练。

## 已实现

- 使用原manifest中每类按路径排序的前2张val，共90张，保存全部选中图；读取现有fixed/soft校准best及原验证预测。通过原manifest、run.json、checkpoint、代码逐文件SHA链保证是此前完成的那一批；不拿refit权重观察它见过的val再称未见样本。
- 实例原图存在时读原图；缺失时直接读持久化ZIP里的对应成员，不解压整个数据集。已有文件内容不符则报错，原文件和解码像素双哈希必须匹配。
- hook捕获真实S/W及48空间+32特征构造边；校验归一化、候选mask、图边/方向、区域汇总及反馈重建。无修改捕获前/后预测，hook和observer在异常时清理；与原缓存概率和预测再核对。
- HTML图册自包含，90个案例和16区域完整S图全保留；每例SVG可独立打开。S argmax、归一化分配熵、固定锚点几何、空间/特征边均明确标注；不冒充像素分割、GAT注意力、图边因果作用或正式总体指标。
- 原像素PNG、S/W/edge_index/logits PT及身份先保存；失败标记非零、保留部分结果和失败缓存张量；同名输出/启动日志拒绝覆盖。成功生成542成员的证据ZIP以及摘要。

## 验证结果

环境：独立Python3.12.13/macOS ARM CPU，torch2.2.2、PyG2.8.0等沿用本地测试环境。解释器`/private/tmp/gavit-soft-qa-20260926/bin/python`。真实服务器Python3.11/CUDA及受训权重尚未执行。

| 检查 | 结果 |
|---|---|
| 改动前分组/探针基线 | 13 passed |
| 初始新增行为RED→GREEN | 缺少新模块时9 failed；实现后9 passed |
| 最终新增15项+既有13项回归 | **28 passed / 39 warnings / 11.54秒** |
| 90张合成图片端到端 | 原ZIP读取、真实分组/PyG、180个PT、90张SVG及完整S图、HTML90案例、542成员ZIP CRC通过；用小型token骨干，输入/模型随机，不是NWPU精度 |
| 完整Swin-T/PyG真实CPU前向 | 两架构捕获/汇总/反馈重建/恢复logit最大差均0，S/W误差≤2.3842e-7；合成图与随机权重，不是训练结果 |
| 启动命令 | Python/Bash/zsh语法、源码解码逐字节一致、后台参数、重复启动拒绝、已有异样源码拒绝均通过；mock子进程，无CUDA任务 |
| 图稿 | 两张合成SVG经离线sharp渲染为PNG并实际查看，4列与16区域图布局、图例和英文文字完整，无裁切 |

39条警告是已有Torch AST/Python3.12弃用提示。最终测试命令：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /private/tmp/gavit-soft-qa-20260926/bin/python -m pytest tests/test_nwpu_region_visualization.py tests/test_soft_assignment_diagnostics.py tests/test_soft_gavit.py -p no:cacheprovider -q --tb=short
```

浏览器工具拒绝file页面；未绕过浏览器策略，用独立离线SVG光栅化查看图稿，没有浏览器交互验收。服务器只需Pillow/Torch和原项目依赖，不需要sharp、matplotlib或浏览器；sharp仅用于本机渲染QA。

## 独立审阅及修复

只读审阅发现1项P2：缓存概率不符时在PT存盘前报错，原PNG又到全部推理结束才写入，导致失败材料不完整。实际runner注入缓存不符复现该问题（原图不存在断言失败），现改为先持久化选图身份/已验证原PNG，捕获完成立即保存PT，缓存失败附error和failed标记再抛出。新增回归验证失败PT、原图、selection、exit1均留存，最后28项回归通过。没有其他待修审阅项。

测试阶段另修正测试文件缺少Counter导入，未因此修改模型。最终仅按受影响的诊断/捕获范围回归，不重复执行完整训练入口测试或付费实验。

证据：[最终测试](evidence/nwpu_regions_20260929/final_tests.log)、[完整Swin CPU前向](evidence/nwpu_regions_20260929/real_swin_cpu_capture.json)、[启动验证](evidence/nwpu_regions_20260929/launch_validation.json)、[失败保留回归RED](evidence/nwpu_regions_20260929/review_red.log)、[合成图稿QA](evidence/nwpu_regions_20260929/synthetic_qa.png)。图稿QA不能作为论文真实案例。

本工具真实运行后还需检查真实90张图、分组及连边的语义表现；通过工程检查不代表图结构合理，也不能证明软分配引起正式test差距。E05多seed/BigEarth正式消融等仍未完成。
