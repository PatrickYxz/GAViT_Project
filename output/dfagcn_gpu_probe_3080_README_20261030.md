# RTX3080单批检查

已确认当前卡为RTX3080 10GiB，UUID GPU-70be4364-0641-e89d-828e-78727c6001ec；当时空闲9874MiB，驱动610.57.04。按用户本次选择推进DFAGCN；旧4090检查结果与包保留。

## 上传及执行

只上传[新检查包](dfagcn_gpu_probe_3080_20261030.pyz)到`/home/featurize/work/`，保持文件名。包133564字节，SHA256为`0d8ff18b2ecc9cd4903eb6c4833d3757b7c4e27605e0ac18d936b9158a7eac21`。

它优先复用已经保存在work中的VGG权重、manifest和旧450条冻结缓存。无需再传大权重或编译LIBLINEAR。先运行一行：

```sh
python /home/featurize/work/dfagcn_gpu_probe_3080_20261030.pyz inspect
```

inspect只读取资源。只有`ready: true`时，才会给出绑定当前UUID的`run_command`；复制该完整一行执行即可。若有blockers，先将输出贴回，不覆盖旧目录或重新运行360/90。

## 本次实际做什么

run在新CUDA进程绑定这张10GiB3080，核验原manifest、完整VGG文件和旧冻结缓存身份/字节。先做真实VGG的batch1推理，再释放它，在一次性新图模型上以正式batch100完成一次前向、反向和Adam更新，检查有限值/参数更新及显存。

模型、seed42、结构、优化器与精度设置按原完整配方；FP32、AMP/TF32关闭。读取配方只用于模型和单批参数，整个探针不运行epoch循环、不作验证选参或test，不启动100轮训练，不写checkpoint，也不续训旧450图的GCN。

输出位于新检查包同名目录下的runs/gpu_probe_01。旧4090包、checkpoint和450条缓存只读。旧缓存仅用于新卡兼容性检查；正式6300池的特征后续在3080新身份下统一生成，避免混用硬件批次。

## 回传

成功标记为DFAGCN_GPU_PROBE_COMPLETE。summary不含混淆矩阵，可直接回传：

```sh
cat /home/featurize/work/dfagcn_gpu_probe_3080_20261030/runs/gpu_probe_01/run.exit /home/featurize/work/dfagcn_gpu_probe_3080_20261030/runs/gpu_probe_01/summary.json
```

将看到VGG与图batch各自的显存/耗时、实际GPU及软件/精度、optimizer_steps1、formal_training_epochs0、checkpoint_writtenfalse。失败保留独立failure/exit；不覆盖旧结果。

## 本地验收范围

119项DFAGCN相关测试通过，含真实CPU完整结构batch100的前后向、Adam更新和输入缓存未改；独立审查无未解决问题。发现并修复了进程CSV格式异常被误判空闲的问题。一次组合运行触发Mac libomp共享内存错误，日志保留，串行重跑119项通过。

本机资源检查正确停在缺少服务器manifest/缓存/目标GPU，未创建run目录；还不能声称3080 CUDA或显存实测已通过。旧4090约0.57GiB不作为新卡保证。3080实测与4090历史效率分别记录。
