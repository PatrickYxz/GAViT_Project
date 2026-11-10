# BigEarthNet seed45：配对训练和测试完成

2026-09-24核验收尾。用户回传总队列exit0、六阶段各exit0及complete。Swin/GAViT均完成30轮，在同一RTX2080Ti上按冻结协议评估各自最佳验证权重。目录日期20260922是运行身份，不改写为核验日期。[原始粘贴](evidence/bigearth_seed45_20260922_01/reported_summary_output.txt)、[提取摘要](evidence/bigearth_seed45_20260922_01/summary.json)、[核验与逐类分解](evidence/bigearth_seed45_20260922_01/summary_validation.json)均已归档。

## 配对结果

| 项目 | Swin45 | GAViT45 | GAViT − Swin |
|---|---:|---:|---:|
| Test mAP % | 71.676930 | 70.258236 | **−1.418693 pp** |
| Test macro-F1 % | 65.084212 | 64.295477 | −0.788735 pp |
| Test micro-F1 % | 76.732264 | 76.702457 | −0.029808 pp |
| Best val mAP % | 78.6509 | 78.2951 | −0.3558 pp |
| Best epoch | 6 | 10 | — |
| Final epoch | 30 | 30 | — |
| 参数量 | 27,535,501 | 31,366,926 | +3,831,425 |

测试集双方均119825张，主要指标为macro-mAP，F1阈值0.5。GAViT本次三个测试指标均低于baseline；mAP约差1.42 pp。它是这次固定seed/划分的结果，不单独构成跨seed显著性结论。

训练末轮GAViT loss0.08172658044316794、val mAP76.61513448708149%、macro-F1 71.71295447541755%，来自此前同目录第30轮摘录；较自身best val下降约1.6800 pp。Swin最终val及完整30轮曲线尚未本地收到，留空，不推测。

## 类别差异

| 类别 | Swin AP % | GAViT AP % | 差值 pp | 对19类mAP差值贡献 pp |
|---|---:|---:|---:|---:|
| Coastal wetlands | 33.8542 | 10.6684 | −23.1858 | −1.2203 |
| Permanent crops | 63.3516 | 60.8124 | −2.5391 | −0.1336 |
| Transitional woodland, shrub | 78.9833 | 77.8972 | −1.0860 | −0.0572 |
| Complex cultivation patterns | 68.8229 | 70.0692 | +1.2463 | +0.0656 |

完整19类为8升、11降。海岸湿地一类贡献净mAP下降的约86.02%；这是各类AP差除19的算术分解，仍保留全部类别，并非删类后的新评估。其余18类合计贡献约−0.1984 pp。[完整逐类表](bigearth_seed45_per_class_comparison.csv)。AP表未提供类别支持数或逐样本预测配对，不能据此断言样本量、标签或图模块造成退化，也未重新计算F1或显著性。

## 实际资源与预算

| 阶段 | 秒 | 时长 |
|---|---:|---:|
| Swin smoke | 18.004015 | 18.00秒 |
| GAViT smoke | 14.058593 | 14.06秒 |
| Swin训练/验证/保存 | 45857.682399 | 12.738245小时 |
| GAViT训练/验证/保存 | 53606.540994 | 14.890706小时 |
| Swin测试完整测量阶段 | 270.560692 | 4.509345分钟 |
| GAViT测试完整测量阶段 | 407.965277 | 6.799421分钟 |
| 六阶段合计 | 100174.811970 | **27.826337小时** |

上述计时包括阶段中的checkpoint I/O、saved-state核验，test还包含保存预测；不含阶段计时前的身份检查/导入、全数据预检、解压、队列间隔或实例闲置。按之前查询的参考价¥0.69/小时，计时范围对应约¥19.20；这是条件换算，不是用户实际账单，真实单价/优惠及总租用时长未取得。

训练CUDA峰值allocated Swin3.556476 GiB、GAViT3.653889 GiB，reserved3.822266/3.867188 GiB；主进程峰值RSS1823848/2002048 KiB，不含DataLoader子进程。测试allocated623328768/638921728字节；范围含模型加载和后处理，不能替代标准推理效率测量。

## 实验身份与核验边界

- 数据仍为237871/122342/119825，三个CSV SHA与冻结launcher一致；预训练SHA仍f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36。报告clean deb2b45，launcher SHA与本地完整脚本8ac0083f...逐字节重算一致。本地git diff再次确认发布代码的BigEarthNet入口/模型/utils/identity与a23a566无差异。
- 两模型seed45、30轮、batch32、AdamW lr0.0003/wd0.0001、CosineAnnealingLR、BCE、dropout0.1；GAViT为16区attentive_spatial、sparse_hybrid_4n_top2、48空间+32特征边、256隐藏/4头/2层GAT、token_feedback。图信息来自metadata及已核对代码，本次未收到独立真实图边捕获输出。
- RTX2080Ti UUID GPU-0fa5e2ad-0028-881c-749b-3b88c01de00b，驱动610.57.04，CUDA12.1/cuDNN8902，Python3.11.8、torch2.2.2、torchvision0.17.2、timm1.0.27、PyG2.8.0、numpy1.26.4、sklearn1.9.0、Pillow10.3.0、8线程。FP32、无AMP、matmul TF32 false、cuDNN TF32 true，沿用BigEarthNet默认行为。
- Swin最佳checkpoint SHA `78cf048b6e2690e40789d13c78030c2350b238c899024a632347af7f0997ed8c`；GAViT为`4268de448d7fabee4fde6db042241efbf38459953f6108337b95a3025313940c`。各自训练摘要、metadata与test指标报告哈希三者一致；均区别于smoke权重。未本地收到权重字节，不声称完成本地权重重哈希。
- smoke保存重载差Swin0、GAViT4.76837e-7，小于1e-5。原始六份run.json、history及predictions.npz未本地取得。回传common_identity是汇总命令提取的共享字段；六阶段环境相同还由已匹配SHA的launcher在服务器的完成检查支持。本地可重算19类均值、差值与阶段时间和；F1引用回传值。
- 摘要原文首行缺Q（UEUE_EXIT），尾带shell提示符；原文完整保存，仅提取中间JSON，未伪装为服务器summary.json字节副本。此前的舍入日志与精确回传相容，两份证据均保留。

本轮seed45已验收，结果写入主表及[四seed汇总](bigearth_four_seed_summary.md)。不因负向结果重跑或排除seed45；没有新训练、发布代码修改、Git提交或推送。
