# HGTNet3090：CUDA显存口径检查修复

用户实际诊断：CUDA12.1、is_available=True、device_count=1、设备名RTX3090；CUDA properties显示total_memory=24123MB，而NVIDIA清单报告物理总量24576MiB。原检查错误要求二者完全相等，在模型/数据训练前误拦截。显示数值未包含原始字节，完整PyTorch版本也尚未由该诊断返回；新包会分别记录并核验。

## 上传新小包

将 `hgtnet_nwpu_smoke_3090_20261102_fix1.pyz` 上传到 `/home/featurize/work/`。

- 大小53324字节。
- SHA256：`90c60dabcd185ffd0a1b575f8544e7363a469c4e3616528955cb3645c2ae7c42`。
- 原Swin-B权重、manifest和ZIP继续复用，已经通过的完整权重SHA不用重新上传。
- 原失败目录保留；新包安装/output/scratch均带fix1，不执行旧run_command或删除历史记录。

## 执行这一行

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_nwpu_smoke_3090_20261102_fix1.pyz inspect
```

`ready: true`后直接执行本次生成的 `run_command`。这次会先打印 `HGTNET_CUDA_DEVICE_PROBE`，然后在runtime中分别保存NVML物理总显存和CUDA实际暴露容量。CUDA容量允许处于物理总量减1GiB至物理总量的区间；这是明确的兼容门禁上限，不表示已经证明接口差额的系统原因。错误设备名称/数量/UUID、无效/过小容量、CUDA不可用和软件版本仍会拒绝。

## 检查结果

结束后执行：

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_nwpu_smoke_3090_20261102_fix1.pyz status
```

回传短摘要，需complete/exit0、train360/val90/test0、180steps、一轮、checkpoint回读误差≤1e-6。

新输出目录：`/home/featurize/work/hgtnet_nwpu_smoke_3090_20261102_fix1/runs/nwpu_smoke_seed42_01`。

新实例临时图片目录：`/tmp/hgtnet_nwpu_smoke_3090_20261102_fix1_rtx3090_seed42_01/images`。

模型、数据、引擎、seed42、batch2、预训练和精度/优化设置保持原字节/规则，仍只做360/90一轮工程检查。45项当前门禁/打包检查通过；此前完整数值回归仍受本机临时环境缺依赖限制，未声称新数值全量通过或真实3090训练完成。
