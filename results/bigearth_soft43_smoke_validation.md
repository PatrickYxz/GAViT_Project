# BigEarth soft43 RTX4090工程短测（2026-10-11）

用户回传exit0与complete：96train/64val、seed43、1轮，RTX4090 GPU-cede40c0-f93d-7c8e-1006-cf6183e53592，ed480e6 clean、31,466,766参数、BCE/原soft配方。硬件/软件/精度与本批预检一致，FP32/noAMP/matmul TF32false/cuDNNtrue。

CUDA重载最大logit差2.384185791015625e-7；实际48+32图边；8张val、3份探针快照S行/W列归一化误差均2.384185791015625e-7。checkpoint SHA949504d4daf1d943be0dde59ff13eaea250212b0346126030d0f41146a580232。

工程train loss0.5083718399、val mAP31.6302547%、macro-F1 4.5297114%。小样本支持可能不覆盖全部19类，入口跳过无正例类；不能用于预测正式test效果或替代三模型消融。

训练入口11.863476秒，总含重建/验收14.093355秒；CUDA allocated3.662695GiB、reserved3.951172GiB；主进程RSS1927744KiB不覆盖所有worker。这不是纯训练/推理速度，不据小样本线性外推预算。

[回传转录](evidence/seed43_bigearth_smoke4090_20261011/user_return.json)、[字段核验](evidence/seed43_bigearth_smoke4090_20261011/validation.json)。本地未下载原checkpoint/探针；新的[正式启动命令](evidence/seed43_bigearth_smoke4090_20261011/bigearth_launch_checked_command.txt)在服务器只读核对工程产物、复检当前冻结身份，通过后进入原定单soft43的30轮训练→best完整test，不重复smoke、不重训旧对照。当前正式任务尚待启动回传。
