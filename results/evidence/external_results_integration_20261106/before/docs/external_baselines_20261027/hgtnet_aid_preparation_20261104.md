# HGTNet × AID：资产与smoke推进记录

2026-11-04。NWPU独立HGTNet与DFAGCN的seed42补充比较已经完成。用户授权继续后续流程，并确认46/47原实验平台、原始记录与完整F1无法回收。本阶段不再以回收46/47作为推进前置条件。

最新更新（2026-11-05）：实际RTX3060（GPU-df386e6c-dbb3-0952-8190-3265a7bd8ae8）上的240/60、batch2、1epoch smoke已完成，并补回[完整status及10项核对](../../results/evidence/hgtnet_aid_smoke_complete_return_20261105/status_validation.json)：state complete、exit0、完成产物/来源检查true；val Accuracy76.6667%、macro-F1 72.3492%、重载0、wall48.7654秒、allocated1.4314GiB/reserved1.578125GiB。该结果仅为工程smoke，原预测和checkpoint未下载。本次[原样回传与记录](../../results/evidence/hgtnet_aid_smoke_complete_return_20261105/README.md)已归档，下一步[实际3060 batch16容量门禁](../superpowers/specs/2026-11-05-hgtnet-aid-batch16-design.md)只做原train16图的4次更新，为正式配方资源准备提供证据。11月4日[原资产恢复证据](../../results/evidence/hgtnet_aid_inputs_restored_20261104/README.md)和[smoke设计](../superpowers/specs/2026-11-04-hgtnet-aid-smoke-design.md)继续保留。

## 本阶段交付

1. 从已确认的46/47主指标生成可读补录表与来源记录；只计算这些分数支持的均值、样本SD和配对差。缺失F1、预测、运行日志、UUID、精确checkpoint身份及起止时间保持空值，不从其他seed移植。补录不冒充平台原始输出。
2. 创建独立、仅标准库的AID只读资产检查包，检查旧AID冻结manifest、smoke所需图片、Swin-B候选权重、当前GPU和磁盘。输出JSON到stdout，不训练、不写服务器文件、不安装环境、不修改现有NWPU程序。

## 已有AID协议

- 原manifest：`/home/featurize/work/aid_protocol_20260920_03.json`，SHA256 `30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22`。
- schema2、`scene-pool-group-v2`；30类10000条记录，train4000、val1000、pool5000、test5000；split_seed42、val_split_seed4242。每类数量不等，不能套用NWPU的每类固定数量断言。
- 旧数据根：`/home/featurize/data/AID/AID`。同像素Industrial两图按原manifest同时留在train，不自行去重或重新划分。
- 本地仍未收到原manifest字节；目标机用户回传的SHA与结构核验已通过。旧ZIP`/home/featurize/work/AID_visionflan_20260920.zip`、2628474154字节、SHA256 `cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7`在本次链式恢复命令中显示OK；300图实际SHA检查已通过。恢复不改原manifest或像素重复组。
- HGTNet初始候选：`/home/featurize/work/swin_b-68c6b09e.pth`，SHA256 `68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893`。这是独立实现的部分Swin-B初始化，不能拿NWPU已训练45类权重充当AID30类初值。

来源：[AID原正式执行记录](../../.worktrees/sparse-hybrid-4n-top2/docs/run_aid_formal_seed42.md)、[AID三seed结果](../../.worktrees/sparse-hybrid-4n-top2/results/aid_multiseed_summary.md)、[现有外部方法协议](protocol_and_acceptance.md)、[已验收NWPU结果](../../results/hgtnet_nwpu_seed42_analysis_20261104.md)。

## 资产检查边界

检查包固定原manifest SHA、taxonomy及数量，不重新生成split。验证manifest中路径、标签、split、SHA和像素组不跨split，再按每类排序选8张train与2张val，共240/60；只检查这些300张图片的文件字节SHA，不读取test图片。若原件或图片缺失，报告实际缺失情况并继续输出其余资产信息。

模型权重做完整流式SHA；GPU只调用nvidia-smi，依赖版本只读取distribution metadata，不导入Torch/PIL/NumPy，不触发CUDA模型计算。磁盘仅盘点实际容量，不把未经测量的费用/吞吐/正式预算写成实测。即使资产均通过，也只表示可以准备smoke，不表示训练入口已实现或GPU训练已通过。

## smoke与后续阶段

正式AID包已本地完成：[新包](../../output/hgtnet_aid_formal_3060_20261105_v1.pyz)/[启动说明](../../output/hgtnet_aid_formal_3060_README_20261105.md)/[验收证据](../../results/evidence/hgtnet_aid_formal_prepare_20261105/README.md)。64项新formal范围测试和独立审查通过，旧43文件保护哈希不变。现在待用户服务器inspect&&run验证全pool/资源/实际首批；正式test与端到端资源仍为空，不能把小样本工程指标写入正式表。

batch16小包已有[真实3060完整回传](../../results/evidence/hgtnet_aid_batch16_return_20261105/README.md)：16图4次更新/exit0、完成与来源true，allocated4.406981GiB/reserved4.855469GiB、中位计算0.469230027秒、总13.4862秒；容量门禁通过，不能当完整loader ETA。现在进入[正式AID包设计](../superpowers/specs/2026-11-05-hgtnet-aid-formal-design.md)/[实现](../superpowers/plans/2026-11-05-hgtnet-aid-formal.md)，保持原数据协议和独立配方；实际正式GPU尚未启动。

AID专用数据/binding/入口及独立30类smoke包已交付，真实3060完成回传及最终status已核对，冻结NWPU包继续保留。现在准备batch16短资源门禁；通过后落实4000/1000选择、fresh5000 refit及5000 test的正式配置与预算及独立正式入口。工程smoke和探针不写入正式成绩表，固定16图重复计算速度不称完整epoch吞吐。

现有外部方法结果属于各自披露配方下的补充比较。导师要求的统一训练/评估协议、多seed外部比较及具体方法×数据集矩阵仍须单独落实，不能由增加一个AID单seed自动替代。
