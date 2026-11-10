# DFAGCN独立实现

2026-10-30最新：[真实NWPU360train/90val/450subset refit在RTX4090上的工程检查已回传通过](../../results/dfagcn_nwpu_smoke_20261030.md)，交付身份匹配，重载误差0。完整正式训练/test、MatConvNet等价及正式规模成本仍待验证。

2026-10-29已接通缓存、VGG提取、三支训练、验证选择、重新拟合、LIBLINEAR与最终测试。86项测试通过，真实VGG的20图合成工程链完成。详见[运行说明](../../docs/external_baselines_20261027/pipeline_usage_20261029.md)、[固定配方](../../configs/dfagcn_independent_v1.json)和[本轮证据](../../results/evidence/dfagcn_pipeline_20261029/)。规格来源见[论文方法核对](../../results/evidence/external_baselines_20261027/dfagcn/method_spec.json)。

## 组件接口

- `features.VGG16FeatureAdapter(backbone)`：调用方注入标准VGG16-D结构和权重。`forward(images224, images448)`返回`fc/conv3/conv4/conv5`；也可分别调用`fc_features`、`convolutional_features`。输入须为预处理后的RGB浮点张量。接口不下载权重，不自行缩放/归一化。
- `graph.knn_union_adjacency(nodes, k=10, metric=...)`：输入B×N×F，输出B×N×N bool。每样本单独构图，不含自环；无向边取双向kNN的OR并集。距离明确选`euclidean`或`cosine`，相等时按节点索引选，cosine拒绝零向量。这些是披露的工程选择，不宣称已确认作者全部细节。
- `graph.normalize_adjacency`：二值邻接矩阵加一次自环，再做对称度归一化。`GCNLayer`计算A_norm×X×W，可选传播后bias，无默认激活/深度/dropout。`mean_readout`取节点均值。
- `fusion.weighted_concat(conv5, conv4, conv3, fc, alpha=..., beta=..., gamma=...)`：前三支已细化为B×512、FC为B×4096，输出B×5632；不添加归一化。`paper_coefficients("NWPU")`或`"AID"`返回原文系数。
- `identity`记录声明身份；新增cache/extraction核验实际图片、权重和npy字节，pipeline执行calibration/refit/test门禁。cache.key只表示身份，cache.content_sha256绑定完整index及数组SHA。

## 最小使用片段

```python
from external_baselines.dfagcn.graph import knn_union_adjacency, normalize_adjacency, GCNLayer, mean_readout
from external_baselines.dfagcn.fusion import weighted_concat, paper_coefficients

# nodes必须来自调用方明确声明的池化规则；本包没有默认49节点。
adjacency = knn_union_adjacency(nodes, k=10, metric="euclidean")
normalized = normalize_adjacency(adjacency, dtype=nodes.dtype)
layer = GCNLayer(nodes.shape[-1], 512).to(device=nodes.device, dtype=nodes.dtype)
branch_feature = mean_readout(layer(nodes, normalized))
# 这里只示范一个算子连接，不是论文完整GCN分支。
fused = weighted_concat(conv5, conv4, conv3, fc, **paper_coefficients("NWPU"))
```

原组件保留meta形状检查；新增缓存测试与完整smoke已实际执行CPU FC6，并核对随机真实VGG权重保存/加载。尚未验证MatConvNet预训练数值等价或真实NWPU表现。

## 本地验证

依赖限定在独立临时环境，不改现有项目环境。Python3.12、Torch2.2.2、torchvision0.17.2、NumPy1.26.4；新增测试使用Python自带unittest。实际安装结果见[证据目录](../../results/evidence/dfagcn_primitives_20261028/)。

从项目根目录运行：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/gavit-dfagcn-cpu-20261027/bin/python -m unittest discover -s tests -p 'test_dfagcn_*.py' -v
```

原表未决项已采用显式独立配置接通，不冒充原文确认。子命令为cache、calibrate、refit、test、smoke；支持每类8张train/2张val的工程子集，子集不能进入最终测试。尚无真实NWPU/AID分数、正式比较或GPU成本；旧仓库仍缺pytest/timm/torch_geometric，不能称全仓库通过。
