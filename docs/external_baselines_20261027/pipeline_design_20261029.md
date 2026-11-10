# DFAGCN独立流程配置与实施

用户已批准：固定缺失配置，接成完整流程，先本地小样本验证；正式单seed、多seed和GPU任务是后续阶段。既有论文报告分数保留为文献参考。

## 路线与选择

已选择独立实现路线。继续等待原表不作为编码条件；把未知层配置直接说成作者结构也不合适。本轮沿用已验证算子，使用显式工程配置，并保存全部配置与源码身份。

- 448卷积地图自适应均值池化到7×7；逐图Euclidean k10无向并集。
- 三个独立分支：GCN宽度256、256，每层ReLU/dropout0.5；均值readout，MLP1024→512，末512维ReLU导出，其前一层dropout0.5；每支独立分类头。三支CE求和，参数互不共享。
- Adam lr0.001、weight_decay0、batch100；StepLR每50轮乘0.1；校准最多100轮，按验证CE总和选最早最优轮。该上限是待测工程预算，不是本轮启动的训练。
- 固定NWPU原文融合系数0.7/0.3/0.4；LIBLINEAR solver2、bias1、tol0.01；C在10^-4至10^4九值中用验证OA选择，相同OA取较小C。池化/网络/优化初始值/求解器/停止规则均是披露的独立选择。
- 权重入口接受实际SHA校验的torchvision VGG16；RGB伸缩至224与448，bilinear，ImageNet RGB均值/方差，无增强。与原MatConvNet不声明等价。正式模式须使用对应官方checkpoint的397923af SHA前缀及完整SHA，不接受随机初始化。具体正式成本仍需真实smoke后确认。

## 接口与分工

config/model负责配方与三个分支；cache/extraction负责实际像素、已池化浮点npy memmap、文件SHA和身份；linear负责官方LIBLINEAR train/predict；pipeline/CLI负责校准、重新拟合、最终测试。

冻结特征缓存可以跨阶段复用，但每个缓存逐项匹配manifest样本顺序、图像SHA、实际权重、预处理、池化和提取源码。训练artifact另绑定配方/源码/拟合样本、缓存内容、seed。使用npy memmap避免全NWPU特征装进内存，不用pickle保存特征。

校准只拟合train；val选GCN轮数及C。refit重置同seed模型，在完整train+val上按选定轮数重新训练；最终test仅接受完整refit身份、未改变的模型/分类器和准确test样本清单。C不在refit/test重选。正式模式绑定既有NWPU/AID冻结manifest SHA。

## 本轮验证与限制

先跑接口缺失RED，再跑真实Torch与真实LIBLINEAR小样本链。无本地NWPU原图发现时，使用明确标记synthetic/random-smoke的真实像素/VGG/训练/分类器工程检查，不产生NWPU性能宣称。缓存单测可用已知数组；完整smoke必须实际执行VGG的CPU FC6与卷积分支。

各阶段输出新目录，保留失败文件，不覆盖旧checkpoint。完成后保存训练曲线、best/final、配置、缓存/模型/二进制哈希、指标和计时。无Git提交、远端训练或既有指标表改写。

## 执行记录

- Pre-flight：model消费cache输出的conv3/4/5 B×N×C，N由pooling_hw定义，输入C固定256/512/512；fusion消费512维图导出及4096维FC，顺序不变。
- Pre-flight：linear消费完整5632维融合，预测不使用LIBLINEAR打印的dummy-label准确率；项目指标由真实manifest标签计算。
- Pre-flight：cache.key是声明身份，cache.content_sha256才绑定index及npy实际哈希；训练产物均记录两者。
- 本轮证据写入results/evidence/dfagcn_pipeline_20261029；原53项组件和旧指标/源码保护哈希继续核对。
- 已完成：model/config11、cache/extraction10、实际LIBLINEAR7、pipeline4、CLI1及原组件53，共86方法通过。完整实际VGG工程链在新目录smoke_run_02完成；run01和旧源码保留。
- 额外边界：真实NWPU工程子集显式记录train_per_class/val_per_class，pool取所选两组并集；subset不能进入正式或最终test。该接口有有效RED→GREEN。
- 独立审查：无待修复Critical/Important/Minor；完整45类配方另做数值梯度检查。重复缓存复制和测试二进制临时路径两处观察已在最终审查前修正。
- Ruling：未知原配置与MatConvNet来源采用披露的独立配方，不等待回信；代价是不能声称原方法数值等价。
- Ruling：本地20图检查使用合成图片和真实随机VGG，只确认流程；代价是尚无真实NWPU/预训练或GPU行为证据，正式成本需在目标环境另测。
- Ruling：全旧仓库依赖不在新包修复范围，逐项记录9failure/9error及pytest缺失；代价是不能声明完整旧模型回归通过。新86方法与受保护文件SHA单独验收。
- Ruling：哈希绑定实验工作流而不提供恶意并发写入认证；正式运行按输出不变流程保存，代价是不能把它当签名安全系统。
- 本轮未提交Git，保留源文件快照与独立证据；纳入Git历史的时机由用户决定。无deferred minors。
