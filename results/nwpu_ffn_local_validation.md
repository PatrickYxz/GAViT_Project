# NWPU FFN 候选：本地实现与发布准备

**2026-10-04更新：已发布。** 用户批准后，9文件按原候选逐字节提交到独立分支`codex/token-ffn-nwpu-20261003`，完整提交`058f1d8b79d0be538ec358eb742d2cd3992921d4`，独立远端查询确认一致。干净发布checkout144项通过/exit0（51.17秒），独立代码审阅完成且无可行动问题。[发布证据](evidence/nwpu_ffn_release_20261003/release_identity.json)与[服务器命令](../docs/run_nwpu_ffn_smoke_20261003.md)已保存。服务器GPU短测尚未执行；以下10-03准备记录保留为历史，234项与发布checkout144项是不同测试范围。

2026-10-03。状态为**本地 CPU 验证完成，服务器 GPU 短测和效果尚未测量**。用户批准的范围是 Swin+FFN / soft GAViT+FFN 两次 seed42 NWPU 校准；完整协议与停止条件见[运行说明](../docs/run_nwpu_ffn_trial_20261003.md)。

## 实现

新增 `residual_ffn_v1`：池化前逐 token 执行 `x + W2 GELU(W1 LN(x))`，768→192→768；GAViT 放在反馈残差之后。两组共享结构及独立 CPU seed42003 初始化，W2 weight/bias 为0，初始残差为0；原来的模型随机数序列和参数保持。默认不启用，旧 architecture 无新增字段、state_dict 无新增键。

第一步 W1/LN 的任务梯度为0，W2 有非零梯度；第二步梯度能到 W1/LN。AdamW 仍可能对第一步零梯度参数执行权重衰减，不把“零任务梯度”描述为参数完全不动。没有额外搜索宽度、温度或学习率。

scene 入口支持 `--token-refinement residual_ffn_v1`，严格保存并重建版本配置；拒绝错误 FFN 配置、非42 seed、refit/test、非 NWPU 数据集（本地 synthetic smoke 例外）及改变的 NWPU 清单/预训练哈希。完整训练仍执行原来的 clean Git、CUDA、batch32 门禁，并限制 workers2。关闭开关的旧流程通过回归；旧 checkpoint 的正式评估仍应使用记录的旧源码，以满足原 code identity 校验。

每次新运行额外保存 `refinement_audit.json`：初始张量 SHA、实际首次输入形状和零残差、前两次反传的梯度总量、selected/final FFN 张量 SHA。run.json 增加 best/last/训练状态文件 SHA。原训练曲线、验证预测、逐类指标、混淆矩阵、图边和软分配探针继续保存。

## 验证证据

- 先写失败检查：初始7项因缺少 FFN 接口失败；运行记录2项因缺少审计字段失败；冻结协议检查1项因未拒绝改变的 manifest 失败，再分别实现并验证。
- **完整 `python -m pytest tests -q`：234 passed，exit0，119.99秒**。包含真实 Swin-T/soft GAViT CPU 合成图像训练、反传、最佳/最终权重保存、strict 重载和独立 val 评估；以及旧 BigEarth 单轮入口、续训和 scene 兼容回归。
- 额外与改动前、哈希匹配 ed480e6 的真实模型类进行初始前向比较：两组旧参数逐张量相同、CPU RNG 相同、logit 最大差均0；未使用轻量假 backbone。这是随机权重与合成输入，未替代真实预训练/CUDA验收。
- 两模型 FFN 初始张量 SHA 同为 `566b722f15ccd6b8bc927d96d73a426f582bc320f5216eedcd1b2b13f45944ca`（本地实测；服务器应核对同环境两组，不预设跨平台逐bit必同）。实测参数：Swin+FFN **27,852,903**，soft+FFN **31,784,168**，各增加 **297,408**。
- 非零 FFN 训练后验证了位置：Swin 的输入是 backbone tokens；soft 的输入为 `tokens + S @ region_feedback`；分类头读取 FFN 输出的均值。该输出与先池化再 FFN 不相等。

完整日志与结构化记录在[验收证据目录](evidence/nwpu_ffn_prepare_20261003/validation.json)。既有依赖产生39项 PyTorch/Python AST deprecation 和5项合成 TIFF 无地理定位提示，未将警告描述为零。

验证中保留两类失败记录：直接 `pytest -q` 在既有 `baselines/swin_baseline/test_swin.py` 的顶层 argparse 处中断收集；指定 `tests/` 后，沙箱内9项子进程测试触发 OpenMP Error179 共享内存限制。获得沙箱外本地测试许可后，全部234项通过；没有修改生产模型去规避这两类环境/收集问题。设计复核已完成；独立代码审阅代理因使用额度中断，未获得其签字，主代理完成限定差异检查与上述验证。

## 发布边界与下一步

[9文件补丁](../deliverables/nwpu_ffn_trial_20261003.patch)仅含本次4个既有代码文件修改、2个新代码文件、2个测试文件和1份协议。改动前4文件 SHA 与 ed480e6 发布记录一致；其余记录文件也逐一核对，共30项。补丁在对应文件树 `git apply --check`、实际应用及逐字节比对通过；这不是完整新 Git checkout 的发布验收。清单与所有源码 SHA 见[发布候选](evidence/nwpu_ffn_prepare_20261003/release_candidate.json)。

拟从 `ed480e60fea03ed786fdba70c144de5e815c34f5` 建立独立分支 `codex/token-ffn-nwpu-20261003`，只纳入上述9项，再验证完整发布 checkout。当前未创建该分支、未提交/推送，原工作区其他修改、历史结果和服务器源码均保留。补丁供本地审阅/发布使用，不要求用户上传工具包。

发布后通过 Git 同步新目录，先两组360/90、1轮 GPU smoke；检查初始 SHA 相同、实际前向/梯度/重载/图边、退出与资源，估计当前 GPU 成本。随后仅执行已批准的两组30轮 calibration。旧对照复用，不根据新分数自动加轮、扩 seed、refit/test 或转跑 BigEarth。FFN 是否提升及是否值得后续实验均为**待测试**。
