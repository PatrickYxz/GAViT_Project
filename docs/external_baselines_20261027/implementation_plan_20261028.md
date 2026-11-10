# DFAGCN第一批组件实施记录

用户已批准推进；本轮范围是独立组件，不含完整架构、训练器、权重转换或付费GPU任务。规格依据为[method_spec](../../results/evidence/external_baselines_20261027/dfagcn/method_spec.json)。使用test-driven-development：先运行缺失接口的失败测试，再实现，再运行数值/回归检查。

## 接口与任务

1. `graph.py`：逐样本二值无向kNN并集；metric必须显式填写；默认k=10。自环一次、对称度归一化、无隐式激活的GCN层和节点均值。
2. `features.py`：注入标准VGG16结构。224 FC6后ReLU、448三个卷积层前ReLU；实际权重和预处理由调用方提供，不声明MatConvNet数值等价。
3. `fusion.py`：三条512维图支路和4096维FC，按conv5、alpha×conv4、beta×conv3、gamma×fc拼接；没有默认归一化。
4. `identity.py`：仅缓存身份JSON接口，绑定有序样本/原图、manifest、类别、权重、预处理、取层、源码哈希；区分冻结特征、calibration和refit。它不负责生成张量缓存，不自行证明调用方提供的SHA对应真实文件或无数据泄漏。
5. 跑新增测试与现有测试入口，独立代码审查，保存验证证据，更新研究日记和状态。

Pre-flight：graph输出B×N×F，而fusion消费B×512，二者之间仍需由未冻结的完整图分支连接；本轮不擅自补层。features导出未池化卷积图；池化目标保持未决，不继承GAViT的49节点。identity不急切导入Torch，元数据接口可独立使用。

## 验收与限制

- 手算小图、梯度、图批次隔离、融合顺序和系数；真实VGG卷积数值、完整meta形状分别记录。
- 实际CPU FC6数值和MatConvNet预训练等价未覆盖；meta只验证形状。
- 每个原指标文件和旧入口源码均核对保护哈希。无Git提交、推送或旧模型修改。
- 临时环境安装依赖已通过本次正常审批；昨晚一次审批因用量限制未完成，另一次PyPI连接失败。这些不视为组件验证通过。

## 执行日志

各任务测试、审查与最终状态写入[本轮证据目录](../../results/evidence/dfagcn_primitives_20261028/)。保留此记录，不依赖Git提交保存未完成实验事实。

- graph：先见接口缺失失败，再26个方法通过；cosine有限大数范数溢出另做RED→GREEN。
- features：先见接口缺失失败，再13个方法通过。CPU卷积数值与FC meta形状分别记录。
- fusion：先见接口缺失失败，再7个方法通过；另核加权float16溢出必须拒绝。
- identity：先见接口缺失失败，再6个方法通过。
- 三条图支路接融合的组合检查1个方法通过；合计53个方法通过，不把subTest展开数当测试方法数。
- 完整仓库检查受pytest、timm、torch_geometric缺失影响，明细已归档，不能称全仓库绿色。
- Ruling：为避免额外框架下载阻碍，新增测试统一使用原生unittest，保留数值与边界断言；不使用pytest替身。
- Ruling：沿用已获批准的根目录独立包方案，不创建/切换分支，不提交；代价是本轮组件尚需用户后续决定何时纳入Git历史。
- Ruling：本批不执行完整CPU FC6数值验证、不组装未知层数/池化规则；代价是尚不能声称完整方法等价或训练就绪。这是组件范围限制，后续完整方法必须补齐或明确披露独立选择。
- Final: Ruling：审查未判断的MatConvNet/CPU FC数值及完整模型/分类器归入后续集成，原因和代价同上，不把组件通过登记为性能复现。
- Final: Ruling：审查未判断的真实张量缓存字节核对和训练/test执行门禁保留为后续任务；本接口仅声明元数据身份，代价是不能据此证明真实文件完整或无测试泄漏。
- Final: Ruling：审查未判断的全仓库绿灯/GPU行为保持未验证；依赖缺失与CPU范围已披露，代价是不能据此声明既有完整模型回归或GPU运行正常。
- Final review：独立审查无Critical/Important/Minor发现；最终53方法通过，并以标量距离参照另核27节点、k10图。无Deferred minors。审查与作者最终源码哈希一致。
