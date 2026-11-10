# 既有产物保护审计

仅对指定路径做只读流式 SHA256 核对，没有修改既有文件，没有扫描原始数据或全仓库。

历史快照共 52 条：匹配 52，缺失 0，不匹配 0。

- historical_metrics: 19/19 匹配；依据 results/evidence/external_baselines_20261027/baseline_state.json。
- legacy_root_and_research_sources: 12/12 匹配；依据 results/evidence/external_baselines_20261027/entrypoint_comparison.json。
- dfagcn_sources_recipes_and_old_bundles: 21/21 匹配；依据 results/evidence/dfagcn_formal_resource_fix_20261030/protected_inputs.json。

root 与 research worktree 的源码分别对照各自历史哈希；没有要求两者字节相同。

另记录17个较新 DFAGCN 分析结果、导师稿及 fix1 包的当前哈希。这些文件没有本轮实施前的冻结快照，因此不声称已验证它们自本轮开始未变。

逐路径 expected/actual/counts/matches 见同目录 audit.json。
