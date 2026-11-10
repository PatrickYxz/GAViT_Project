# AID入口同步分支验证

日期：2026-09-20。代码基于既有sparse-hybrid分支的8f2ed07671ed50387998a46a2d2167a7a4dfc2ed。

仅引入run_scene.py、scene_classification五个源文件、两份测试、服务器指南及本记录；旧训练入口、模型源码和正式结果表均未修改。八份Python文件与已完成代码审查的本地版本逐字节一致。本次分支只整理AID代码，不作为完整实验档案；历史实验进度以原研究工作树的最新research_diary.md为准。

此前18项新增测试及相关模型回归共74项通过、31个subtest通过。两个真实模型在CPU、随机初始化、合成30类图片、workers=0下完成prepare/train/evaluate五条CLI命令；权重保存前到重建的logit最大差均为0，GAViT捕获48空间+32特征边。另验证本地生成的同一份权重可给两个backbone加载相同张量；不等同于已验证服务器ImageNet文件。

隔离分支完整tests/验证：89 passed、34 subtests passed、exit0、8.85秒。39条警告来自Torch JIT与Python3.12旧AST接口。见[完整输出](evidence/aid_entry_release_20260920/tests.log)。运行命令：`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /private/tmp/gavit-aid-qa-20260919/bin/python -m pytest tests -p no:cacheprovider -q --tb=short`。没有AID真实性能结果，没有启动服务器训练。

已知边界：本地Mac双worker检查曾发生OpenMP共享内存错误，获准在沙箱外重试后180秒超时，后者根因未定；成功的workers=0流程不能替代Linux/CUDA workers=2验收。服务器尚无已确认的AID数据；实际ImageNet文件、CUDA、吞吐和显存需按[运行指南](../docs/run_aid.md)验证。正式calibrate/refit需两模型smoke均通过后另行安排。
