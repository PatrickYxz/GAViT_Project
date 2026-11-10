# AID HGTNet：RTX3060 batch16短检查

上一阶段实际3060的一轮smoke已通过：val Accuracy76.6667%、macro-F1 72.3492%，独立重载误差0，总48.7654秒，峰值allocated1.4314GiB/reserved1.578125GiB。该工程结果已归档，不写正式AID成绩表。

现在只确认正式候选batch16的实际容量和计算耗时：16张原train图片、1次warmup+3次measured，共4次一次性更新。没有epoch、val/test、checkpoint或准确率输出。新包67224字节，保持原30类模型、同一份Swin-B候选；不载入smoke训练后的权重。

## 运行

将同目录`hgtnet_aid_batch16_probe_3060_20261105.pyz`上传到Featurize的`/home/featurize/work/`，保留原文件名。只需要上传这一个小包，原manifest、图片和初始化继续使用已有文件。

复制这一整行，完成后会自动读取status：

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_batch16_probe_3060_20261105.pyz run && /environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_batch16_probe_3060_20261105.pyz status
```

入口自动选择唯一空闲RTX3060，导入Torch前绑定当前物理UUID，并复核实际CUDA/软件版本；无需手填昨天的UUID。它会重新检查原manifest、完整初始化权重SHA和所选16图字节；不重新解压、下载或划分数据。独立输出位于`/home/featurize/work/hgtnet_aid_batch16_probe_3060_20261105/runs/batch16_seed42_01`，存在输出或锁时拒绝覆盖。

把末尾完整status JSON贴回即可。若被预检拦截，贴完整错误；不要改旧程序、门禁或覆盖原输出。

## 验收与后续

期望state complete、exit0、completion_evidence_valid/source_identity_matches true，以及unique16、processed64、optimizer_steps4、batch_size16、warmup1/measured3。记录实际UUID、三次同步计算耗时、中位数与allocated/reserved峰值。

计算耗时只覆盖已放在设备上的固定16图的forward/CE/backward/有限梯度检查/Adam更新；首步warmup、读图、预处理、transfer、callback和参数诊断不计入中位数。峰值统计从模型/输入放到设备及optimizer构建后reset，包含常驻分配、四步与诊断；不表示更早初始化峰值或完整正式流程峰值。这些数值不能直接称为完整loader吞吐、正式训练ETA或租卡账单。

通过后继续准备原4000train/1000val选择、fresh5000pool refit、5000test的独立正式AID入口与预算。正式候选沿用最多100轮calibration、验证correct严格增加/相同选最早、同初态fresh refit到选中轮；batch16/Adam1e-5/WD5e-5。完整正式trainer尚未在本阶段交付或运行。

## 本地交付身份

SHA256：`fd54b4c1a21db1731d1addb0808c49104c11113b610883b79b2c6af456fc80f5`。

bundle ID：`f07b8d6744acbd4ee25a16c170a90fa913011e1ed668c6a035837dc4c494662e`。18个源码/配置文件，19个payload、21个ZIP成员，不含权重大文件或安装器。数值12项CPU小模型测试、入口8项和交付6项标准库测试通过，独立审查通过；最终实际包的无ML导入inspect/status与成员/源码身份已核对。当前尚无3060 batch16实测，旧35个源/包/成绩表保护哈希保持一致。
