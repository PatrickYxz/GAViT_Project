# E05 局部原型软分配：本地实现验收

2026-09-26开始实施，09-27凌晨（Asia/Shanghai）完成收尾。用户确认实现计划后完成本地代码、测试和一次独立审阅，审阅修复后完整回归 **190 passed / 44 warnings / exit0，47.20秒**。没有新数据集精度结果，没有部署服务器或启动付费训练。

工作树：`/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2`；分支 `codex/sparse-hybrid-4n-top2`。基线 HEAD `8f2ed07671ed50387998a46a2d2167a7a4dfc2ed`，本次没有提交/推送/合并；该 SHA 不能单独代表本次未提交实现，须结合 [最终源码哈希](evidence/soft_assignment_local_20260926/final_source.json)。既有未提交研究材料保留。

## 已交付

- `soft_prototype`：旧区域注意力形成当前图片原型，经共享LN/768→128投影在局部候选内做cosine/0.1软分配；S沿区域归一化，W沿token归一化。新增参数99,840。
- 区域汇总与反馈都使用软归属，投影、原注意力和骨干可通过分配路径获得梯度。图构造预算保持48+32，空间边仍是锚点先验，非真实区域接壤的保证。
- 完整版本配置写入新checkpoint，缺字段、值/类型不符、非法组合、跨架构重建均拒绝；旧模式公共参数、构造随机数和one-hot反馈等价检查通过。
- scene/NWPU 与 BigEarthNet 训练/重载入口接通；新配置从metadata重建。旧NWPU入口拒绝新架构并指向scene。旧scene记录仍需原代码版本通过既有源码身份检查。
- 固定最多8张无增强train/val/pool探针，保存epoch0、best、final及每轮统计，不使用test；捕获后恢复RNG、逐模块train/eval状态和hook。BigEarth中断后核验并恢复已存快照别名，不重新推理伪造历史。

## 实际验证与原始证据

环境：macOS ARM / Python3.12.13 / CPU；独立解释器 `/private/tmp/gavit-soft-qa-20260926/bin/python`。torch2.2.2、torchvision0.17.2、timm1.0.27、PyG2.8.0、numpy1.26.4、pillow10.3.0、pytest8.3.5、sklearn1.6.1、rasterio1.4.4；[完整环境](evidence/soft_assignment_local_20260926/environment.txt)。这不是服务器Python3.11/CUDA环境的逐版本复现。

| 检查 | 结果 | 日志（同目录有exit文件） |
|---|---:|---|
| 修改前身份/模型/图/scene基线 | 81 passed | [baseline_tests.log](evidence/soft_assignment_local_20260926/baseline_tests.log) |
| 严格分组配置、旧架构和元数据解析 | 42 passed | [task1_green.log](evidence/soft_assignment_local_20260926/task1_green.log) |
| S/W轴、局部mask、梯度、退化输入、批独立、RNG | 9 passed | [task2_green.log](evidence/soft_assignment_local_20260926/task2_green.log) |
| 公共初始化、one-hot整模等价、真实PyG CE/BCE、重载 | 24 passed | [task3_green.log](evidence/soft_assignment_local_20260926/task3_green.log) |
| 探针无副作用、scene三模型CPU smoke及身份 | 31 passed | [task4_green.log](evidence/soft_assignment_local_20260926/task4_green.log) |
| BigEarth三模型原入口CPU训练/保存/重载/测试、受控续训 | 33 passed | [task5_green.log](evidence/soft_assignment_local_20260926/task5_green.log) |
| 审阅修复：best/final别名恢复与入口回归 | 16 passed | [review_fix_green.log](evidence/soft_assignment_local_20260926/review_fix_green.log) |
| 修复后完整tests目录 | **190 passed；无skip** | [final_tests.log](evidence/soft_assignment_local_20260926/final_tests.log) |

表内不同阶段有重叠，不能相加成测试总数。最终44条警告为39条Torch旧AST/Python3.12弃用提示和5条合成TIFF缺少地理参考提示。最终命令：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /private/tmp/gavit-soft-qa-20260926/bin/python -m pytest tests -p no:cacheprovider -q --tb=short
```

基线和各功能的RED/GREEN输出均在 [证据目录](evidence/soft_assignment_local_20260926)。初次BigEarth子进程在沙箱内遭遇OpenMP共享内存错误，不计为有效功能RED；通过正常授权在沙箱外运行CPU测试。随后修复测试工厂保持DataLoader为类以及既有状态文件后缀的断言；有效RED才用于判断缺少新入口/重载行为。未改生产代码来规避沙箱。

端到端工程测试使用真实Swin与PyG、临时合成单标签图片和两图/划分的19标签RGB TIFF，实际执行优化、验证、存盘和重载。测试工厂只把预训练下载替换为本地随机初始化、DataLoader workers设0；测试调用的是真实入口，不是模拟整个train/evaluate。独立数学/等价测试另使用有空间变化的小骨干。合成数据的分数不能作论文结果。

## 独立审阅及修复

一次fresh-context只读审阅覆盖当前工作文件和基线哈希；没有用空的HEAD..HEAD冒充审阅。审阅原判无Critical/Important，提出1项Minor：诊断写入epoch后、best/final选择前中断，可能在续训exit0时丢失命名快照。

实现者按实际交付影响提升为Important：这会让使用者收到“完成”，却缺少后续论文图所需产物。先在原入口复现epoch1后best缺失和最终epoch2后final缺失（[两个失败](evidence/soft_assignment_local_20260926/review_entry_red.log)），再添加基于已存epoch快照与checkpoint最佳轮的恢复，校验身份和epoch，缺失源文件明确失败。测试也覆盖旧别名替换、无新forward、epoch0及历史不变；16项相关检查和190项完整回归通过。本轮没有再次派审阅，也无剩余待延后minor。详见[审阅记录](evidence/soft_assignment_local_20260926/review.md)。

## 实施取舍与未验证边界

1. 增加 `select_snapshot` 和恢复别名接口以落实best/final产物要求；不改模型公式。若接口需调整，涉及两个runner调用点。
2. 未提交实现以当前文件和前后哈希审阅并交付，保留工作树与执行ledger；发布时须严格按文件清单，不能把同目录其余研究修改一起误提交。
3. GPU、真实图像、workers4、资源费用、完整真实校准→refit及精度收益留给服务器验收。CPU通过不能取代这些证据；风险是实机依赖、数据或资源问题尚未暴露。
4. BigEarth继承的续训不保存完整RNG、不自动验证所有split和源文件哈希；本次仅保证新架构/训练参数/探针身份与记录一致。服务器须外部冻结并核验源码、split和预训练，不能声称bit级无中断等价。扩大为完整续训身份改造属于后续独立事项；风险是未经外部核验的旧机制无法识别所有输入漂移。

旧分组、骨干、图构造、图推理、数据和指标模块的未改动哈希在最终清单中核对。旧研究记录和结果表中的历史协议缺口仍保留；本轮不声称审计了全部历史研究文件，也没有往comparison/ablation表填入新精度。

下一步依次执行 [服务器短测说明](../docs/run_soft_assignment_smoke.md)：确认NWPU数据路径、发布独立固定版本、NWPU原图1轮、BigEarthNet96/64原图1轮。上述准备完成后再给用户不含嵌套引号或临时变量的绝对路径单行命令。正式三模型/多seed与外部论文比较仍未完成。
