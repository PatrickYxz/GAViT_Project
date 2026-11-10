# 外部方法复现：现有协议与阶段验收

核对日期：2026-10-27。本文件是独立复现的准备材料，没有新增模型实现、运行结果或付费训练。

## 1. 已有划分直接复用

AID：30类，10000张；内部校准使用4000张train和1000张val，最终重新拟合使用5000张训练池，测试5000张。已归档manifest SHA256为`30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22`。

NWPU-RESISC45：45类，31500张；内部校准使用5040张train和1260张val，最终重新拟合使用6300张训练池，测试25200张。已归档manifest SHA256为`ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`。

两套协议均固定`split_seed=42`和`val_split_seed=4242`。训练seed独立保存；改变训练seed时不重新生成划分。NWPU正式清单已经生成且用于既有实验，旧接入文档中“尚未生成”的描述是历史状态。

本轮核对的是本地代码与已归档服务器报告，没有重新连接服务器确认持久化目录的实时状态。后续运行前需按原清单哈希核验实际文件，不重新分割数据。

来源：

- [AID三seed实际流程](../../.worktrees/sparse-hybrid-4n-top2/results/aid_multiseed_summary.md)。
- [NWPU seed43实际流程和manifest身份](../../.worktrees/sparse-hybrid-4n-top2/results/nwpu_seed43_test_comparison.md)。
- [现有数据实现](../../scene_classification/data.py)。

## 2. 文献报告值与本项目结果

文献报告值保留原数据集、训练比例、重复方式和打印精度。缺少直接可核对的标准差时填空，不用其他论文转引值或本项目标准差补齐。

本项目是在固定划分上重复训练。DFAGCN明确报告十次随机划分，包含划分变化；HGTNet本轮只确认五次重复，具体是否每次重划分仍未确认。不同重复方式的标准差不合并，也不把比例相同写成逐样本配对。

本轮不将文献数值写入`results/comparison_table.csv`、`results/ablation_study.csv`或已有多seed性能表。文献值的来源类型固定为`paper_reported`；未来本项目独立实现的实测类型为`independent_reimplementation`，在真正运行和核验前不填写性能。

## 3. 工程入口单独建设

未来代码建议位于根目录的新`external_baselines/`包。只复用已有冻结清单和指标工具，不把外部模型名称塞进旧`RunConfig`。

可以复用的数据接口：

- `load_manifest(path, root)`：读取并重新核对冻结清单。
- `select_records(manifest, split, *, per_class=None)`：按既有split选择样本。
- `classification_metrics(truth, predictions, class_names)`：计算OA、macro-F1、逐类指标和混淆矩阵。

接口分别见[数据模块](../../scene_classification/data.py)和[指标模块](../../scene_classification/metrics.py)。类序从manifest的`class_names`读取，不为外部方法另造标签顺序。

旧入口的限制：

- `architecture`和`build_model`只接受Swin/GAViT，旧checkpoint metadata使用`scene-v1`和GAViT架构规范。
- 图捕获门禁写死16节点、48条空间边和32条特征边，不能作为HGTNet或DFAGCN的构图正确性检查。
- 旧正式流程固定batch32、交叉熵、AdamW和30轮Cosine，且只有一个224输入和一个预训练权重入口。
- 旧测试分支仍可测试calibration权重；外部方法的“20%最终训练”主比较须另行强制`phase=refit`、`fit_scope=train_plus_val`和6300张拟合身份。

根目录与研究工作树的数据、指标核心实现一致。新实现不修改旧模型和旧身份规则，避免已有checkpoint失去对应源码。

## 4. 先验证的有限范围

第一篇先在NWPU 20%协议下做工程验证，训练seed为42。方法选择依据论文结构的可执行程度，不按文献分数高低或是否有利于GAViT选择。

第一阶段：合成输入上的张量形状、有限值、具体图算子、可学习参数梯度，以及保存后重载结果一致性。原方法的预训练与预处理必须有可检查身份；随机权重可以用于工程检查，但不冒充正式预训练实验。

第二阶段：只从既有内部train和val分别取最多8张/类和2张/类，即最多360张train、90张val。输出使用独立smoke目录，不读取test。DFAGCN需要走完特征提取、GCN、融合和线性分类的小链，不能只运行特征提取器。

第三阶段：工程检查和资源预算明确后，再落实一个seed的5040/1260校准、6300训练池重新拟合及25200张最终测试。正式运行前冻结配置、实现假设、预训练来源、源码哈希、停止规则及输出身份。正负结果均保留，不按test表现选择实现分支或更换seed。

本轮不启动上述训练阶段，也不提供尚未实现入口的远端启动命令。正式多seed及其费用不包含在本次文档准备中。

## 5. 选择规则与数据泄漏检查

任何有学习参数的标准化、GCN和分类器，只能在当前拟合集合上拟合；验证集用于事先规定的选择，test不得用于C、融合系数、停止轮数、架构分支或预处理选择。

DFAGCN的C和融合系数搜索必须把GCN特征生成过程纳入数据隔离：不能用已在train+val上训练的GCN特征，再声称val仍是独立选择集。最终refit按冻结规则从原预训练或初始状态重建学习产物。

可缓存的是身份明确、与标签选择无关的固定预训练特征；缓存键必须绑定样本顺序、manifest/class map、输入尺度、预处理、提取层、权重和源码哈希。校准与refit训练得到的特征不能混用。

公共约束先固定为相同数据、信息可见性、标签映射、验证选择原则、最终测试和报告规则。原方法必须保留的多尺度输入、分阶段优化和线性分类器另列配置；不能用“相同30轮端到端脚本”代替它们，也不能仅凭相同split宣称所有训练细节已一致。正式统一协议的具体预算须在实现和小样本检查后冻结。

## 6. 依赖与资源现状

只读检查了项目`.venv/bin/python`、`/opt/homebrew/bin/python3`和`/usr/bin/python3`。这三套解释器均未定位到torch、torchvision、timm、torch_geometric、numpy、scipy、sklearn或pytest；Homebrew Python可定位PIL。未安装依赖，未导入模型或执行真实张量运算。该检查不代表整台机器不存在其他环境。

后续先确定可用执行环境，再区分静态/标准库检查、真实CPU张量检查、目标GPU小样本和正式结果，不把前者写成后者完成。

新GPU实验遵循项目4090政策，记录GPU UUID、软件和精度。方法级效率覆盖完整特征提取、图聚合、融合和分类；缓存后的分类耗时另列。训练时长、峰值显存和费用目前均为待测，不套用Swin/GAViT的历史耗时。

## 7. 本阶段完成条件

本阶段交付是原论文报告值、方法模块核对、实现选择记录及首篇复现次序。以下仍不是本阶段的完成声明：外部模型已经实现、真实前向已通过、论文分数已经重现或导师要求的多seed比较已经完成。
