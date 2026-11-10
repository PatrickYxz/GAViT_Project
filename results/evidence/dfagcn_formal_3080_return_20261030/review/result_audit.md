# DFAGCN formal seed42 摘要独立核验

结论：SERVER_REPORTED_FORMAL_COMPLETE_SUMMARY_IDENTITIES_MATCH，29/29 项一致性检查通过。证据为用户粘贴的 run.exit 和 summary 两个 JSON；未直连服务器、未重训或重跑 tests。

- OA **86.53174603174604%**，macro-F1 **86.45890874660422%**；seed42，selected epoch51，C10，完整5040/1260→6300 refit→25200 test。
- OA×25200/100 = 21806.0，与 **21806/25200** 一致；这是由摘要推导的正确数，不是独立读取逐图预测。macro-F1 无逐类计数不能独立复算。
- 六阶段均completed且identity一致；phase_seconds与receipts逐值相等。合计 **2025.858676642s**，wall **2035.634612280s** = 33.927244min，差 **9.775935638s** 为阶段外wrapper开销。
- allocated峰值 **616995328B = 588.412598MiB**，来自test_cache；reserved峰值 **698351616B = 666.000000MiB**，来自pool_cache。各自等于六阶段最大值；不是整进程显存。
- test完整phase **100.970574281s**；内部cached **66.657492617s**（378.052024图/s），排除VGG并排除缓存/模型前置加载。test_cache+test **796.622277846s**（31.633562图/s）是本次阶段吞吐，不能当纯模型推理速度。
- pool/test内部提取分别160.344223592s/676.178100577s，图/s由6300或25200除以时间复算一致。外部cache阶段还包括ZIP恢复/加载权重等。
- 交付fix1包SHA **79398dc4b6d703be65cd32a968d49daa6640fe6c4bbab6a33e930b312954a6c4**；全部29个payload bytes/SHA与delivery manifest一致。本地源码与包内源码逐字节一致。pipeline/source_identity/runtime三个摘要SHA均本地复算匹配；recipe byte SHA与冻结常量匹配；本地553433881B官方权重完整SHA与摘要匹配。
- 图网络参数2,957,447（包括三CE头），冻结VGG提取器117,479,232，由当前结构独立算术复算；LIBLINEAR单独计，不能只把2.957M当完整提取流水线参数。

## 范围与限制

- Source, runtime, recipe and input identity fields are internally consistent and match local delivered bytes where available; user-pasted summary is server-reported evidence, not direct remote artifact inspection.
- Full test.json, 45-class metrics/confusion matrix, predictions, calibration/refit histories and checkpoint bytes are not locally returned. OA-derived correct count is a consistency calculation; macro-F1 and predictions cannot be independently recomputed from this summary.
- Manifest value matches the frozen code constant; original full manifest bytes and remote data/ZIP are not locally inspected in this audit. Native Linux binary SHA values are recorded, not independently checked against remote binary bytes.
- Selected epoch51 and C10 are valid frozen-recipe choices. Summary alone does not reproduce the validation selection curve, 100 calibration epochs, 51 refit epochs or per-epoch timing; delivered success path requires these formal stages.
- wall_seconds starts after inspect_resources and output creation; it includes runtime/source setup, six phases and final identity checks, but excludes preflight/input hashing, packaging, transfers and final summary/exit/progress writes. It is not billed instance duration.
- phase test includes cache/model identity validation, checkpoint load and output/metrics persistence; cached_test_seconds starts only after cache/model setup and excludes VGG. Both include more than neural forward latency. test_cache phase includes recovery and weight load, while inner extraction excludes them.
- GPU peaks are maxima of six PyTorch phase allocation/reservation counters after per-phase reset, not nvidia-smi whole-process memory or a proof of whole-run peak outside phases. The allocated and reserved maxima arise in different phases.
- Single independent torchvision-based implementation on one seed42 / one RTX3080 UUID; no official DFAGCN numerical-equivalence, three-seed stability, Swin-relative benefit or cross-GPU efficiency claim follows.

## 检查明细

- PASS `parse_exactly_run_exit_and_summary_objects`
- PASS `completed_and_training_started`
- PASS `bundle_expected_sha`
- PASS `all_delivery_payloads_match_manifest`
- PASS `local_source_recipe_helpers_match_delivered_bytes`
- PASS `pipeline_source_hash_recomputed`
- PASS `recipe_byte_hash_matches_constant_and_summary`
- PASS `manifest_hash_matches_frozen_code_constant`
- PASS `weights_hash_matches_bundled_provenance`
- PASS `local_official_weight_bytes_hash_matches_summary`
- PASS `all_six_expected_phases_completed`
- PASS `phase_seconds_exactly_matches_receipts`
- PASS `all_phase_identities_match_recomputed_source_runtime_and_inputs`
- PASS `summary_allocated_peak_is_max_of_phase_peaks`
- PASS `summary_reserved_peak_is_max_of_phase_peaks`
- PASS `allocated_never_exceeds_reserved`
- PASS `full_counts_and_partition_arithmetic`
- PASS `selected_epoch_and_C_in_frozen_recipe`
- PASS `runtime_matches_actual_fixed_recipe_and_uuid`
- PASS `command_uuid_and_frozen_weight_sha_match`
- PASS `output_and_artifact_paths_match_command`
- PASS `scratch_caches_match_command`
- PASS `reported_reload_error_zero`
- PASS `cached_test_is_within_full_test_phase`
- PASS `pool_extract_count_and_throughput_arithmetic`
- PASS `test_extract_count_and_throughput_arithmetic`
- PASS `wall_larger_than_sum_of_sequential_phase_timers`
- PASS `oa_consistent_with_integer_correct_count`
- PASS `parameter_counts_recomputed_from_architecture`
