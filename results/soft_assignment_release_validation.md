# Soft-assignment 发布候选验收 — 2026-09-27

**09-28发布完成**：用户选择推送并创建PR，独立分支`codex/soft-assignment-20260927`已推送提交`ed480e60fea03ed786fdba70c144de5e815c34f5`，[草稿PR#1](https://github.com/PatrickYxz/GAViT_Project/pull/1)的head/23文件清单已核对。base为`deb2b457e86b6fa2feccb5a305e099b6d9ec6dfe`所在`codex/aid-entry-20260920`，未合并。18个代码/测试差异保持与以下验收相同，另有5个发布文档/证据文件。尚未运行GPU。下文候选测试描述为发布前记录。

## 可审阅内容

- [18文件补丁](evidence/soft_assignment_release_20260927/soft_assignment_code.patch)：9个已有文件修改、9个新增文件。仅包含分组配置、软区域模块/反馈、诊断、scene及BigEarth入口、相应测试。补丁SHA256：`b0ac6659d601e66a5687b30a502efc69383550909f4827158731f64c57556b65`。
- [完整源文件身份与文件清单](evidence/soft_assignment_release_20260927/release_identity.json)。30个实现/测试文件哈希全部与之前本地验收相同；数据模块、划分逻辑、图构造、GAT和旧分组模块不变。当前工作区其他未提交研究材料未复制到候选。
- 核心变化：增加`soft_prototype`，S按区域归一化、W按token归一化，区域汇总和图反馈共享软归属；新增99840参数，图构造仍48+32条边。新架构metadata严格重建；保留旧模型路径和已有研究结果。

## 新验证

在`/private/tmp/gavit-soft-release-git-20260927`的本地克隆上测试：其HEAD为`deb2b45`，叠加上述18文件，尚无新提交。执行该候选的完整`tests`目录：**133 passed / 44 warnings / exit0 / 47.52秒**。[日志](evidence/soft_assignment_release_20260927/tests.log)、[退出码](evidence/soft_assignment_release_20260927/tests.exit)。旧工作区190项还含未发布工具的57项测试，此次数量差异不是跳过候选测试。

环境为macOS ARM、Python3.12、CPU，沿用`/private/tmp/gavit-soft-qa-20260926/bin/python`，OMP/MKL各2线程。`git diff --check`及完整补丁反向应用检查均exit0。此验证不替代Linux/Python3.11/CUDA/真实原图/多worker验证。

初次纯源码导出在沙箱内测试有10项失败：1项依赖Git身份（导出无`.git`），9项子进程受OpenMP共享内存沙箱限制。保留[失败日志](evidence/soft_assignment_release_20260927/export_sandbox_tests.log)。改用真正本地Git克隆、获准在沙箱外复测后通过；没有修改模型或测试断言来绕过失败。

## 已确认服务器输入

- Git：`codex/aid-entry-20260920`，`deb2b45`，干净。
- GPU：RTX2080 Ti，total22528/free22009 MiB，为用户查询时瞬时值。
- NWPU：45类、31500图，5040/1260/25200；8组同类重复全部在单个划分内。
- 清单：`/home/featurize/work/nwpu_protocol_20260927_02.json`，SHA256 `ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`。
- 预训练：`/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors`，SHA256 `f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`。

## 发布后下一步

发布已完成，先执行以下固定SHA同步；成功后再运行下方短测。新目录已存在或Git失败时保留现场并返回错误，不强制覆盖。

```bash
git -C /home/featurize/work/GAViT_AID_20260920 fetch origin codex/soft-assignment-20260927 && git -C /home/featurize/work/GAViT_AID_20260920 worktree add --detach /home/featurize/work/GAViT_SOFT_20260927 ed480e60fea03ed786fdba70c144de5e815c34f5
```

确认代码与上述数据/预训练哈希后，运行下面已核对CLI的单行命令：

```bash
nohup python -u /home/featurize/work/GAViT_SOFT_20260927/run_scene.py train --model gavit --grouping soft_prototype --phase smoke --seed 42 --train-per-class 8 --val-per-class 2 --manifest /home/featurize/work/nwpu_protocol_20260927_02.json --data-root /home/featurize/data/NWPU-RESISC45 --output /home/featurize/work/nwpu_soft_smoke42_20260927_01 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --device cuda --batch-size 32 --workers 2 >> /home/featurize/work/nwpu_soft_smoke42_20260927_01.launch.log 2>&1 < /dev/null &
tail -f /home/featurize/work/nwpu_soft_smoke42_20260927_01.launch.log
```

360张训练/90张验证、1轮，使用原图和本地预训练；训练入口仍会先完整校验manifest图片。目的为CUDA前后向、真实图边、保存重载和资源测量，不根据短测OA宣称收益。启动需模型/设备/首batch、GPU进程；完成需`SCENE SMOKE COMPLETE`、`run.exit=0`、`run.json.status=complete`、实际80边、重载差≤1e-5、epoch0/1及best/final探针产物。分配S专属梯度还需单独实机检查，不能由分类loss下降推定。

正式30轮三模型校准、BigEarth原图短测/正式消融和多seed仍未执行。旧scene结果继续使用原代码版本重载；不放松源码身份校验。
