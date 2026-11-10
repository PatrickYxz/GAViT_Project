# HGTNet AID smoke本地准备

状态LOCAL_AID_SMOKE_BUNDLE_READY_TARGET_GPU_PENDING。新数据、engine、入口、builder和30类配置独立于冻结NWPU源/包；无安装依赖、下载、GPU运行、正式训练或Git提交推送。

- [设计](../../../docs/superpowers/specs/2026-11-04-hgtnet-aid-smoke-design.md)、[计划](../../../docs/superpowers/plans/2026-11-04-hgtnet-aid-smoke.md)。AID原manifest/初值身份不变，240train/60val/test0、seed42、batch2、1epoch、Adam1e-4；当前目标3060/12288MiB及用户实际UUID。
- `data_engine/validation.json`：21项测试、RED/GREEN及明确CPU240/60夹具120次更新/reload0；同像素组、不等类支持、只读安全字节解码、30类及失败后实际更新状态检查。
- `entry_delivery/validation.json`：21项最终入口/交付测试，3060/runtime门禁、NVML/CUDA容量口径差、文件锁、防覆盖、源码/成员校验。初18项通过后独立审查发现skeletal summary错误完成；P2的RED/GREEN与严格JSON/runtime/来源/预测/进度完成证据验证均保留。
- `integration/related_tests.log`：52项相关测试在上述P2修正前通过；`affected_tests_after_p2.log`：受影响21项修正后通过。不把两份日志拼成一次55项新suite，不重复无关旧386项。
- `integration/full_model_release_validation.json`：冻结17个源/配置身份，完整30类74612163参数、实际部分Swin-B初值加载、两张生成Tensor的一次CPU更新、五组梯度、30×30混淆和独立重载0，9项通过。每次夹具保留独立临时输出；旧迭代结果保留，release为最终源绑定版本。不是AID图片、GPU或性能评估。
- [独立最终审查](review/independent_review.md)、`review/p2_delta_validation.json`：P2已关闭，修正范围4项针对回归/15项检查通过；没有剩余可操作问题。原反例和历史源SHA保留。
- `integration/final_build.json`记录实际63235B包、SHA`a70b6fbf038aa82daac9398b1d75c057d37b4fe3bf18167604b95d6f6a9d3961`、包ID`cc25746e7cc34705a454bf472938d2b7ad0db30b895c4ffb5d4485d1573285e9`。`final_bundle_inspect_verified.json/status_verified.json`为实际包在禁止ML实际导入下执行，本机缺资产所以ready=false/not_started。
- 初版根验证guard把合法find_spec查询误判为ML导入；`integration/guard_adjustment.json`及旧guard保留，修正为拒绝实际import并检查sys.modules，实际最终包验证通过，没有改生产代码或掩盖失败。
- `before_identity.json`保存24个旧实际源码及原NWPU包/结果表身份；最终复核见`final_validation.json`。原manifest/初始权重/目标CUDA等真实运行门禁仍由目标机run重新检查。

[用户操作说明](../../../output/hgtnet_aid_smoke_3060_README_20261104.md)。当前尚无实际3060/AID smoke结果；local test与CPUfixture不填写正式test指标。
