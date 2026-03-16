# results/figures/

存放所有论文用图表，命名规范如下：

| 文件名格式 | 内容 |
|-----------|------|
| `loss_acc_curve_{model_name}.png` | 训练/验证曲线（loss + acc） |
| `confusion_matrix_{model_name}.png` | 45类混淆矩阵（test set） |
| `graph_vis_{model_name}_epoch{N}.png` | 图结构可视化（节点/边权重） |
| `attention_map_{model_name}_{sample}.png` | GAT 注意力热图叠加在原图上 |

生成脚本位置（待添加）：`scripts/visualize.py`
