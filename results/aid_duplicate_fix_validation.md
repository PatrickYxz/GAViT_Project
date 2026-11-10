# AID重复图片修复验证

日期：2026-09-20；修复基于发布分支2ba112a。实际服务器报错来自Industrial/industrial_203.jpg和industrial_205.jpg完全相同的解码像素。没有获取或修改服务器图片；本地验证使用合成RGB图片。

- 先增加6项数据行为回归，原版本均因缺少duplicate_policy参数失败；实现后18项数据测试通过。
- 新增新schema清单到真实Swin CPU训练、checkpoint身份与独立val评估的集成回归。原Swin/GAViT流程照常测试。
- 发布工作树完整tests/：**96 passed，34 subtests passed，exit 0，10.52秒**。39条警告来自Torch与Python3.12 AST弃用接口。[日志](evidence/aid_duplicate_fix_20260920/tests.log)。命令：`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /private/tmp/gavit-aid-qa-20260919/bin/python -m pytest tests -p no:cacheprovider -q --tb=short`。
- 实际CLI prepare：20张合成图、1组跨编码重复，生成train8/val2/test10；19张唯一图片、1份多余副本，两条路径均test，退出0。未删除/改写原图。
- 独立审查未发现阻断项；审查者另报告45927个子集和检查、1400个旧版划分等价检查通过。
- 指南命令经bash/zsh -n检查通过；git diff --check通过。

新选项保留所有图片，只对完全相同解码像素分组，拒绝跨类别标签冲突。先确定性选精确pool，再选精确val；某个固定pool若无法保持组完整并满足val数则失败，不承诺为所有数学可行的三分组寻找全局解。此限制及自定义协议已记录在运行指南。

实际AID全量扫描、完整ZIP SHA256、服务器ImageNet权重加载、CUDA和双worker仍待用户执行；上述结果不能作为正式AID性能或服务器成功证据。
