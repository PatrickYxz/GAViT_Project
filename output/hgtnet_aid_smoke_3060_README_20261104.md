# HGTNet × AID：RTX3060一轮小样本验证

原ZIP、冻结manifest、初始Swin-B和240train/60val的300张图片均已按用户回传核对通过。本包新增独立30类入口；工程配置为seed42、batch2、Adam1e-4、CE、1epoch、无增强/AMP/TF32。只使用所选train/val，test0。

## 上传文件

上传`hgtnet_aid_smoke_3060_20261104.pyz`到`/home/featurize/work/`。文件63235B，SHA256：

```text
a70b6fbf038aa82daac9398b1d75c057d37b4fe3bf18167604b95d6f6a9d3961
```

包ID：`cc25746e7cc34705a454bf472938d2b7ad0db30b895c4ffb5d4485d1573285e9`。只含独立源码、配置和身份信息，无模型大文件或安装器。原manifest、图片及初始权重继续使用已通过的位置。首次调用把源码安装到同名独立目录，检查后才执行入口；旧NWPU包和输出不修改。

## 直接运行一轮

输入刚刚已通过检查，可以直接执行这一整行；run会再次预检数据/权重/设备/版本，失败则停止，不进入正式训练：

```bash
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_smoke_3060_20261104.pyz run --gpu-uuid GPU-7acce72a-93cd-e56f-35d8-5d33ecf6655e
```

如只想先看新运行包的门禁，使用：

```bash
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_smoke_3060_20261104.pyz inspect --gpu-uuid GPU-7acce72a-93cd-e56f-35d8-5d33ecf6655e
```

inspect在预检通过时显示ready=true及run_command；实际Torch/CUDA、预训练加载、模型容量和更新必须由run确认。显卡应为RTX3060/12288MiB；实际CUDA总容量允许比NVML少不超过1GiB。目标软件为Python3.11、Torch2.2.2+cu121、torchvision0.17.2+cu121、NumPy1.26.4、Pillow10.3.0、CUDA12.1/cuDNN8902。若环境或GPU不符，回传实际错误，不改变旧代码绕过检查。

## 进度和完成

预期先看到实际CUDA设备行，再看到`first_optimizer_step`（五组梯度有限且有实际更新），随后30/60/90/120步进度。一轮共240张训练图、120次更新，再评估60张validation，保存独立checkpoint并用新模型严格重载；误差须≤1e-6。

成功末尾显示`HGTNET_AID_SMOKE_COMPLETE`。输出目录：`/home/featurize/work/hgtnet_aid_smoke_3060_20261104/runs/aid_smoke_seed42_01`，含run.log、binding/runtime/来源、first_step、optimizer_progress、history、60条预测/30类指标/混淆矩阵、smoke_epoch1.pt、summary和run.exit。

完成后运行这一行，回传完整JSON：

```bash
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_smoke_3060_20261104.pyz status
```

新输出目录拒绝覆盖，不支持resume。若失败，保留日志与实际更新状态，先发最后完整错误，不反复重跑同目录。只看到PID、输入ready或训练进度不算整轮成功。

## 本地验证范围

新数据/engine21项测试；入口/交付修正后21项测试；只读输入工具13项既有相关测试。集成52项在修正前通过，完成状态门禁修正后重新运行受影响21项；独立审查关闭一个完成状态误判问题。实际最终pyz的inspect/status在禁止ML实际导入条件下通过，本机无目标资产/GPU所以如实ready=false/status not_started。

完整74612163参数的30类HGTNet已用实际Swin-B候选加载、两张生成Tensor输入完成一次真实CPU更新、五组梯度和独立重载，误差0；另有明确CPU夹具覆盖完整240/60/120次更新。这些只验证工程行为，不是AID实测或3060容量/速度证据。一轮smoke的validation分数也不进入正式性能比较表。

来源：[设计与固定范围](../docs/superpowers/specs/2026-11-04-hgtnet-aid-smoke-design.md)、[本地证据](../results/evidence/hgtnet_aid_smoke_prepare_20261104/README.md)、[300图恢复回传](../results/evidence/hgtnet_aid_inputs_restored_20261104/README.md)。
