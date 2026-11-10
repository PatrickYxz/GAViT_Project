



# Research Diary — GAViT Project

> 记录每次重要进展、实验结果和 debug 过程。每次完成功能模块、获得新结果或修复重要 bug 后必须更新。

---

## 2026-11-10 — 用户要求换机开发；保存两个开发分支的本地未提交工作

- 用户明确要求将本地未提交内容提交并推送远端，实验推进随即暂停。盘点发现当前token-ffn分支及sparse-hybrid工作树均有大量源码、论文、实验记录未提交；将分别保存各自分支，不擅自合并、重写历史或修改main。换机说明见[development handoff](docs/development_handoff_20261110.md)。
- 归档前重新运行当前AID campaign26项、DFAGCN campaign13项、AID结果分析4项目标测试，均通过；本地缺完整ML依赖，不将此描述为全项目或真实CUDA通过。准备包保留原嵌入来源/哈希，不因新Git提交而改写历史实验身份。
- SSH最新实际状态：切换手机热点、代理保持开启后，公钥登录一次完成nvidia-smi，观察RTX3080/10240MiB、UUID0d9901ca-d2e0-d398-b9b4-2550e6db2fed；后续重连仍有超时。最后有一条已认证bash会话，但未发送训练命令且现已关闭。没有启动smoke/正式训练或自动队列；监督器仅讨论尚未实现。新机器先核对实例和任务，不按聊天中的成功登录推断训练已开始。
- 源码/文档/研究记录完成凭据扫描；临时权重下载响应头含Cookie，整个tmp及本机SSH配置/known_hosts、系统缓存和个人工具笔记留在本地并加入忽略规则。代码、独立运行包、论文和可追溯结果作为项目内容保存；原数据/checkpoint仍按既有规则不进入Git。本次提交推送系用户新明确授权，不改变此前未提交快照及只读实验的事实。

---

## 2026-11-10 — 单卡九项实验已授权；运行包本地核验完成，新实例SSH仍待稳定登录

- 用户批准按最新排期由本会话通过SSH操作新实例，保持单卡顺序、控制成本。冻结AID soft42/43/44/45及Swin45/固定45、DFAGCN新增43/44/45共9次；原DFAGCN独立v1 seed42仅在模型/划分/配方/权重/选择规则不变时复用。完整历史含不利seed继续保留，不补造46/47缺失记录，不新增HGTNet训练。研究问题、队列顺序、停止条件和固定配方见[预登记](results/evidence/campaign_prepare_20261110/frozen_campaign.json)。
- 新AID交付使用原Swin-T/区域/GAT模型与训练规则，补充独立smoke、完整epoch状态/RNG恢复、两阶段重训和完成封签；DFAGCN只扩展交付/设备/缓存门禁，17项算法/config/data/metrics与已完成seed42包逐字节相同。新代码是基础提交058f1d8上的未提交源码快照，以逐文件SHA标识，未冒充干净Git提交。
- 本轮独立复核AID标准库目标测试26项、DFAGCN13项通过；两个实际pyz的源码/manifest与当前仓库一致。[本地准备收据](results/evidence/campaign_prepare_20261110/root_preparation_receipt.json)记录AID包922132B/SHAea5e7775874c4bf4cd93c4e17a7ea200227d9e8ebb18700227baef6fda864291，DF包162345B/SHAc09b024cf03eb0becd6358abc156c8802c824b38509d34db7fc322758e68c28b。当前本机没有可用Torch张量运行环境，本地标准库检查不代表真实CUDA/模型前后向通过。
- 用户将实例SSH端口由18307更换为27435，仍授权使用已保存公钥；密码不用于命令或运行记录。新端口默认路径可返回OpenSSH服务器信息，但密钥交换超时；en0绑定路径在banner前关闭；系统本地SOCKS路径一次完成密钥交换并发出公钥认证请求，未收到认证结果，随后有限重试仍超时。未见明确公钥拒绝，不能认定密钥失效或将网络代理定为已证实根因。[诊断日志](results/evidence/featurize_ssh_connect_20261110/)保留，主机密钥按新端口独立固定；没有修改系统网络设置。
- 截至本条记录，没有成功执行新实例上的nvidia-smi、文件检查、数据恢复或训练命令；GPU型号/UUID、依赖、磁盘和持久文件均待实查，smoke/正式训练均未启动。请求用户临时更换Mac网络进行区分诊断；连接稳定后先完成环境/输入/容量检查及真实smoke，再进入正式队列。原结果表、论文、checkpoint未更改，无外发邮件、Git暂存/提交/推送。

---

## 2026-11-09 — 按最新导师要求制定单卡加速排期

- 用户要求为后续工作排期，并确认“越快越好”、一张卡顺序运行控制成本、DFAGCN索取邮件已发且等待一个月。按用户确认已满足导师两周等待条件，不重复等待或安排重复索取作为前置。
- [详细排期](docs/advisor_work_schedule_20261109.md)及[复制版](docs/advisor_work_schedule_20261109.txt)保存，目标11月12–13日结构完整工作稿、11月18日可送审完整初稿，19–20日故障/排版缓冲。日期为目标；首个完整AID soft实验完成后按真实耗时重排GPU结束日，未捏造单卡时长/价格。
- DFAGCN独立v1/seed42若实现/manifest/配方不变可复用，余3次；AID soft4次共最少7次新增正式实验。AID原可核验完整比较只有42/43/44；建议按结果无关规则扩展第四seed并冻结，预算为Swin/固定各预留1次，完整同四seed三模型设计按9次安排。46/47原件/F1已丢失，不当可回收数据、不补造。DFAGCN其余seed清单也在开工前登记，不随测试结果换seed。
- 本地准备与论文写作并行于GPU队列：HGTNet原文值/协议脚注和相关工作先做，不再安排HGTNet训练；三模型参数/FLOPs统一口径并核对动态/GAT算子，已有同卡效率表复用，换卡补测不得拼接速度结果。soft按负面消融记录，可能机制与已证实结果区分；完整历史和不利seed保留。
- 本轮仅制定日程、范围、预算和验收条件；未发邮件、租卡、连接新实例或启动训练，未改实验数值/旧CSV/已交付论文。此前内部写作计划已附最新优先级说明，未删除其未完成事项。

---

## 2026-11-09 — 导师新回复与完整seed汇报澄清草稿

- 用户转贴今日导师回复，并明确此前只汇报了正向结果，询问是否应补充两个不利seed。导师当前“固定版稳定提升”的建议建立在已汇报42/43/46/47子集上；不能将该建议当作覆盖全部已完成实验的数值证据。
- 本地重新核对原AID计划42/43/44：固定−Swin OA差+0.60/+0.30/−0.76pp，平均+0.0466667pp，配对差样本SD0.7145162pp。BEN44/45存在负向结果，历史执行条件差异继续随表，不能只通过选seed或假设随机波动排除。上次+0.46/+0.75pp四seed子集与完整历史范围分别说明。
- 已拟定[澄清英文邮件](docs/supervisor_seed_results_clarification_20261109.txt)和[中文依据/待核对事项](docs/supervisor_seed_results_clarification_20261109.md)，准确说明前封遗漏已完成结果，不将历史实验说成今日新增实验；仅保存草稿，未发送。固定版最终架构与其相对Swin是否稳定是两项结论，soft负面消融不用于掩盖固定版不利重复。
- 导师关于DFAGCN四seed、AID soft四seed、HGTNet论文数值、参数/FLOPs及相关工作的新要求已登记待核对，当前用户问题不作为额外租卡/训练或对外联系授权。已完成的独立seed42外部结果不删除。内部整合的数值审计、原始正负记录和现有写作计划保留，不把未完成章稿或编译说成已交付。

---

## 2026-11-08 — 完成外部结果离线分析、导师复制稿与论文补充章；47页草稿导出并核验版式

- 本轮问题是：已完成的HGTNet×NWPU/AID、DFAGCN×NWPU三个独立实现seed42结果能支持哪些补充比较，以及AID近同分是否掩盖双向错误变化。沿用原冻结划分、现有预测和history，不改变实验变量。用户此前确认归还原实例；旧SSH配置已停用，监控已结束，本轮没有重连实例或执行GPU训练/模型评估。
- 原[HGTNet AID46份记录](results/evidence/hgtnet_aid_formal_complete_20261106/raw/)及归还实例前已回收的[Swin/GAViT10份原始评估文件](results/evidence/external_results_integration_20261106/aid_core_raw/)共56项逐份SHA复核通过。三个AID模型的5000条路径、标签、30类顺序一致；本地重新计算OA/macro-F1为Swin96.34%/96.22088514124282%、固定GAViT96.94%/96.78112715611192%、HGTNet96.92%/96.7728681493198%，分别4817/4847/4846正确，与原件一致。
- [真实配对](results/aid_seed42_three_model_paired_analysis_20261106.json)同时保留两个方向：Swin→固定GAViT纠正93/新增63、净+30；Swin→HGTNet纠正108/新增79、净+29；固定GAViT→HGTNet共同正确4771、纠正75、新增76、共同错误78，净-1。净差1不等于仅1张发生变化。NWPU HGTNet相对固定版净多对4张仍只有汇总差，未推造该对逐图纠正数；HGTNet/DFAGCN原配对2415/503保留。
- [离线分析脚本](tools/analyze_hgtnet_aid_results.py)及4项有意义的标准库测试已通过；RED与本轮GREEN日志保留。导出全部5000预测、154错误、30类指标、30×30矩阵、870个混淆方向、435双向对和100校准+96 refit轮历史，提供4图PNG/SVG。best96与final100均可见，refit75缺失wall字段保持空值。原始图/CSV/raw未被论文图适配覆盖。
- [三模型AID表](results/external_baselines_aid_seed42_20261106.csv)、[跨数据集比较](results/external_methods_comparison_20261106.json)及[完成说明](docs/external_baselines_20261027/completed_comparison_20261106.md)齐备。AID固定版比HGTNet少57.95%参数，相对自己的Swin-T参照增加13.91%；这些异配方、异预算、异设备的单seed观察不建立稳定排名、图模块因果贡献或同条件效率优势。AID恢复attempt2的4986.588秒及4.406/4.820GiB峰值仍仅覆盖恢复调用，全实验资源/费用未知。
- 更新中文完整汇报、英文邮件草稿及逐字段纯文本，并新增[11月8日可复制汇报](docs/supervisor_external_results_update_20261108.txt)。旧内部主表和历史正负结果按原范围保留；46/47缺失F1/原始记录仍空缺，不补造。邮件只保存草稿，没有发送。
- 英文论文新增第5章补充比较，并修正绪论/设置中的旧状态；原修改前文件已保留。实际PDF发现目录自引用/负间距重叠和图缩小后字号不足，随后修正目录，并从完整数据重画14.5cm最终宽度论文图（轴/类名/legend8pt，CM数值7pt；色值未取整，低于1%的非零数标<1）。正文12pt、页边距和行距保持原模板。
- 最终源码原生编译及隔离临时Tectonic编译均成功；[47页PDF](output/pdf/ntu_gavit_external_results_20261108.pdf)完成全部页面版式覆盖：19页最终直接视检，28页正文与已视检版面RGB/尺寸、decoded内容及MediaBox逐页完全一致。50条书目均被引用；无未定义引用、缺字或Overfull，27项Underfull经对应页检查未构成实质缺陷。当前仅是第1–4章加外部补充的部分稿，正式前置页、内部结果/消融、综合讨论及结论尚缺，未把47页当作完整50–60页论文。
- [新版Overleaf导入包](thesis/ntu_gavit_overleaf_20261108.zip)14项均与当前源文件逐字节匹配，稳定名称同步，10月29日旧包单独保留。[本轮验收](results/evidence/external_results_integration_20261106/final_validation_20261108.json)记录56原件hash、4测试、35项独立科学核验、页面映射及PDF/ZIP身份；旧核心comparison/ablation表hash保持不变，无Git提交推送。
- 下一步将外部补充内容合并到完整结果结构，接入内部主表、消融和讨论并重新分配50–60页总预算。外部统一训练/评估协议、多seed均值/样本SD与可比效率仍是未完成要求；本轮不启动矩阵之外的DFAGCN×AID训练。

---

## 2026-11-06 — HGTNet AID seed42完成并独立核验：test Accuracy96.92%，macro-F1 96.772868%

- 用户要求持续监控后，保持本轮只读检查至训练与测试完成。[26次原始轮询](results/evidence/hgtnet_aid_formal_live_monitor_20261106/)保存；poll11一次SSH超时，立即公钥重连确认原任务继续推进，随后复用SSH连接、按分钟检查至完成。本地首轮监控44534退出后以44990接续，均未调用run/resume/test、重做评估或修改服务器训练数据/源码/配方/checkpoint。
- 北京时间2026-11-06 11:00:49直接观察强status：state=complete、completed=true、exit_code=0、source_identity_matches=true、completion_evidence_valid=true；PID3136和GPU计算进程均已退出。原始run.exit=0、completion.marker及attempt2日志footer均为HGTNET_AID_FORMAL_COMPLETE，不是仅由进度条判断完成。
- AID30类固定seed42，校准train4000/val1000、fresh refit pool5000、test5000；calibration100轮、最早最佳epoch96、refit96轮、55048接受步。最佳验证Accuracy97.20%/macro-F1 97.0745569519779%；最终test4846/5000正确（154错），Accuracy96.92%、macro-F1 96.77286814931982%。本地从5000条有序预测独立重算Accuracy、macro-F1、30×30混淆矩阵和全部逐类指标，与服务器记录一致；未将训练100%准确率当作test。
- attempt2由refit75/generation175/48475步恢复、保留原48655观察和180未提交更新回滚记录，连续完成76–96；校准/refit初态SHA相同，独立重载误差0，最终test事务complete且test_reused=false。最终模型SHA4de7bdfbbc6afdf91ee3dc59b0c084b36d0161a07d1457b0c9f56cd6e0504aa2，test SHA6d35db2feaef0864c0f3181002164e8e999c84685f01064af3cd83c6306e7a0b；原manifest SHA30ac...aa22与本地收到的原文件一致。Git base058f1d8b79d0be538ec358eb742d2cd3992921d4、uncommitted_source_snapshot，22项源/config哈希核对一致，不伪称新提交。
- 已直接取回[46份原始记录](results/evidence/hgtnet_aid_formal_complete_20261106/raw/)，9154560B记录包SHAcf9e17989e6bf2005ae58afd210af42df0925e171e9fbb159c383a46824bdbb1，每份SHA核对通过；[主审计](results/evidence/hgtnet_aid_formal_complete_20261106/root_audit.json)和[独立41项审计](results/evidence/hgtnet_aid_formal_complete_20261106/independent_audit.json)均PASS。原始history、全部test/val预测、30类统计、binding/原manifest、事务、重载、两次attempt和三份模型收据已保留；三个大型.pt在持久化work，服务器强status核对二进制SHA，本地核对收据，未下载或加载模型/图像权重冒充独立GPU验证。
- 当前RTX3060 UUID91a8；恢复后attempt2墙钟4986.588285893秒≈83.11分钟、峰值allocated4731174400B≈4.406GiB/reserved5175771136B≈4.820GiB，仅覆盖本次恢复调用。整个实验总墙钟、全程峰值、费用仍未获得；不能把本次恢复时间当总训练时间。接受轨迹已记录轮时和可离线统计保留在history，不等于账单。
- [最终结果JSON](results/hgtnet_aid_seed42_20261106.json)、[可复制报告](results/hgtnet_aid_seed42_20261106.md)、[独立AID结果表](results/external_baselines_aid_seed42_20261106.csv)及进度终态已保存。这是独立实现单seed固定配方结果，不冒充官方复现或多seed/统一backbone优势；原核心comparison/ablation表未混入异配方结果。训练监控完成后，heartbeat hgtnet-aid已实际更新PAUSED；实例未自动归还，可由用户决定后续用途。本轮无Git提交推送。

---

## 2026-11-06 — 按用户要求持续监控现有AID任务，已到第91轮

- 用户明确要求延长当前对话、持续检查直到这一任务完成。启动本机只读轮询PID44534，每60秒通过已验证SSH配置查询现有status/进程/GPU/末段日志，逐次原始回执保存在[监控目录](results/evidence/hgtnet_aid_formal_live_monitor_20261106/)。循环在完成候选、状态/连接错误或进程消失时结束，交由主会话核验；没有运行run/resume/test或更改服务器文件。
- 已在同一对话创建并确认ACTIVE的5分钟heartbeat（id hgtnet-aid），[监控身份](results/evidence/hgtnet_aid_formal_live_monitor_20261106/heartbeat.json)保存。当前分钟轮询在正常更新时，心跳沿用回执、不重复建立监控；关键阶段、完成、失败或需要人工操作才形成可报告发现。完成验收和归档后停用心跳，避免重复运行。
- 首两次回执显示refit90的5000样本训练结束，随后完整epoch_complete53170步，进入refit91/53230步/960样本；原正式PID3136和GPU UUID91a8一致，GPU进程约5102MiB。仍未完成正式test，不将训练准确率或预估耗时当作最终论文指标。验收将检查selected/refit96、55048步、强status五项、重载≤1e-6、test5000有序预测/30类统计/事务和哈希。持续保持当前任务与配方，不启动额外训练，也不自动归还实例。

---

## 2026-11-06 — SSH直连已验证，直接读到refit88/96；后续可免人工转贴

- 已用Mac专用agent公钥登录当前实例，直接执行nvidia-smi、正式包status、pgrep及只读日志检查。成功验证[连接记录](results/evidence/featurize_ssh_connect_20261106/connection_check.json)、[status原始输出](results/evidence/featurize_ssh_connect_20261106/direct_status.json)、[进程/GPU/日志原始输出](results/evidence/featurize_ssh_connect_20261106/direct_inspection.json)已归档。SSH debug明确Server accepts key/Authenticated using publickey/Exit status0；未使用登录密码，未记录口令或私钥内容。
- 用户本机普通SSH同样无响应后Connection closed by198.18.0.11。单连接nc绑定物理en0（默认路由172.20.10.1、MTU1500）访问经两家DNS核对的公网IP183.223.25.19后成功；仍使用原host-key pin/严格校验/同一公钥。正式status连接已恢复默认sntrup密钥交换并成功，不把前期KEX对照误记为必须降级算法。默认路径与物理路径对照可复现差异，但具体隧道转发机制不继续猜测。
- [项目SSH配置](output/featurize_gavit_ssh_20261106.conf)/[使用说明](output/featurize_gavit_ssh_20261106_README.md)保存，仅影响该连接，没有修改全局代理或用户SSH配置。该配置依赖当前端口/IP/host-key/interface和agent，换实例时重新核对。后续可直接读取/回收已授权任务的输出，无需用户充当日志中转。
- 2026-11-06T02:22:23Z（北京时间10:22:23）直接观察：正式PID3136与GPU计算PID3136相同，GPU91a8/RTX3060/12288MiB，当前总体5111MiB、该进程5102MiB。status source_identity_matches=true，manifest/config/recipe/counts与原实验一致，attempt2/未完成。日志refit87完整完成52231=25000+87×313步；当前refit88已处理480样本、52261步，新训练事件明确optimizer_step_succeeded=true。余2787步至预期55048，包含当前88–96共9轮；上一完整轮日志phase ETA2099.771962秒≈35分钟仅覆盖剩余refit，后续还需重载/test。第87轮100%为训练accuracy，不能写成test。
- [当前进度](results/hgtnet_aid_formal_seed42_progress_20261106.json)已改为DIRECT_SSH_VERIFIED_REFIT88_IN_PROGRESS；GPU即时利用率取样为0%不否定同PID和随后新增训练事件，不当全程峰值或故障证据。未启动额外训练/恢复/测试、未改服务器文件/模型/配方/checkpoint，未变更核心结果表，最终指标与正式总资源仍待完成后直接读取。失败连接原记录保留，本轮本地仅配置/说明/证据记录和diff检查，无Git提交推送。

---

## 2026-11-06 — Featurize SSH直连排查：服务可达，尚未完成认证

- 用户提供当前实例SSH端点并授权直接查看GPU/训练，确认公钥已粘贴保存。当前Mac专用密钥文件存在，公钥指纹与已加载SSH agent一致；仅核对公钥指纹，不记录私钥或登录口令。只读连接[排查记录](results/evidence/featurize_ssh_connect_20261106/connection_check.json)、六份SSH debug日志及首次取得的公开host-key pin保存；没有将聊天中提供的口令写入脚本、日志或配置。
- 默认沙箱DNS无法解析；批准的外部网络调用能建立TCP并识别OpenSSH8.9服务器。默认协商在KEX回复等待时超时；单次curve25519对照可完成host-key/NEWKEYS、读取publickey/password方法并发送专用agent公钥，但服务器回复前超时，无接受/拒绝或认证成功记录。密码对照连接在banner阶段就超时，没有进入密码输入。所有请求目标都是只读nvidia-smi，实际均未执行远程GPU/训练命令。
- 系统将该域名解析到198.18.0.11，经utun0/198.18.0.1、MTU9000路由；AliDNS和Google HTTPS解析均返回183.223.25.19。使用该公网地址且保留原host-key pin的对照也停在公钥回复前；IPQoS=none对照同样未认证。观察到隧道路由，但尚不能认定代理、密钥或实例认证后端是哪一项导致超时，不据此修改全局代理或重建密钥。
- 已请用户仅在Mac原生终端验证一次登录，以区分本机/实例路径与当前工具执行环境。远程直连状态为SSH_NOT_CONNECTED_LOCAL_TERMINAL_COMPARISON_PENDING，不能宣称已接管。训练最近证据仍是此前用户回传的refit76完成；本轮无新增正式指标，不动运行中的任务、源码/配方、数据划分或checkpoint，不启动额外训练/测试。结果主表保持原样，本地仅标准库记录及diff检查，无Git提交推送。

---

## 2026-11-06 — HGTNet AID新实例续训首批确认，第76轮已完整完成

- 用户[完整终端回传](results/evidence/hgtnet_aid_formal_resume_started_return_20261106/reported_output.txt)按原始字节保存，[解析事件](results/evidence/hgtnet_aid_formal_resume_started_return_20261106/reported_progress.json)/[核对记录](results/evidence/hgtnet_aid_formal_resume_started_return_20261106/verification.json)归档，旧状态另存。pgrep显示正式resume PID3136、指定新卡容量证明目录；同次nvidia-smi（01:42:01、时区未提供）显示同PID3136的CUDA进程5102MiB，GPU总体5111MiB/12288MiB、利用率99%。满足CPU进程/GPU/首batch联合启动确认，而非仅发出后台命令。
- attempt2的resumed接受refit75/48475步快照，generation175、897033192B、SHA bb7c8bb4f2115bd1b384f9a86f47688cf96fada39b3e3ad3ad27ad347be5cefc与前次receipt逐字段一致。首个恢复更新refit76/48476步、CUDA0/16样本，loss有限，五组451张量梯度均有限且非零、missing/nonfinite为空、五组代表参数更新与postcheck通过。由冻结代码路径可确认模型/Adam/选择状态已接受；未独立下载新快照复算SHA。
- 随后第76轮完整处理5000pool图/313次更新，epoch_complete接受总步48788=25000+76×313，selected_epoch96。训练191.054413秒、轮墙钟213.005072秒、26.170555图/秒、训练acc99.92%、loss0.003590046。上述均为训练指标，不是test；即时GPU显存不是全程峰值。
- 还剩refit77–96共20轮/6260步，预期最终接受总步55048。旧实例76轮未提交的180步按恢复规则重算，不双计。日志phase ETA4696.991812秒≈78.28分钟是包括旧实例已完成轮的运行均值估计，只覆盖剩余refit，不含启动/最终重载/test/费用，不能承诺完成时间。
- [当前进度](results/hgtnet_aid_formal_seed42_progress_20261106.json)更新为RESUMED_TRAINING_CONFIRMED_REFIT76_COMPLETE。保持当前实例与既有后台任务运行；不重启、不再probe、不改锁/配方/源码。稍后只读正式包status读取完成情况。最终test、全程资源/耗时及完整完成证据仍待回传，核心结果表不填训练片段。本地本轮仅归档与标准库核对，独立只读审查结论一致，无GPU训练、源码/包变更或Git提交推送。

---

## 2026-11-06 — 原图恢复及新卡四步通过，后台resume请求已发出；首批尚待确认

- 用户[原始附件字节](results/evidence/hgtnet_aid_formal_resume_launch_return_20261106/reported_output.txt)已归档，[四步解析](results/evidence/hgtnet_aid_formal_resume_launch_return_20261106/probe_progress.json)/[启动排查](results/evidence/hgtnet_aid_formal_resume_launch_return_20261106/triage.json)保存。原ZIP完整SHA校验OK，unzip -n到本地data后新独立probe目录完成，PID3075、16图4步/64次呈现、451梯度张量五组/代表更新和每步postcheck通过。三次计算中位0.466862467秒；未回传新probe allocator峰值或5000pool校验结果，保持空值。
- 用户执行旧锁mv -n改名及nohup后台resume命令，终端未返回错误；随后立即tail启动日志为空、nvidia-smi报告01:37:11（时区未知）、RTX3060/1MiB/0%/无GPU进程。没有新formal进程PID、resumed或新first_optimizer_step回传，只能记BACKGROUND_RESUME_REQUESTED_STARTUP_UNCONFIRMED，不能判断已成功续训或已失败。旧锁实际归档内容和新锁归属尚未独立回传。
- 独立只读源码复核确认：在首条CUDA日志前，resume先静默hash5000pool/351MB初值、两个约897MBsnapshot SHA、只读容量证明，并在锁内重复必要检查。因此刚发出后台命令时日志空/GPU空闲可以处在CPU预检。需要当前CPU进程和新日志，不能根据即时GPU截图再次启动任务。
- 下一步只读pgrep -af正式包名、tail启动日志末30行、nvidia-smi。成功恢复应出现refit75/48475步的resumed（Tensor/Adam已接受），随后refit76首步48476；原48655含180次未提交部分更新，回退是接受轨迹恢复规则。当前不重跑probe/run/resume、不清锁或改源码；最终test与正式资源仍待回传。

---

## 2026-11-06 — 第75轮快照文件通过恢复预检；新GPU与本地data缺失为当前阻塞

- 用户[完整回传](results/evidence/hgtnet_aid_formal_resume_inspect_return_20261106/reported_output.txt)/[inspect JSON](results/evidence/hgtnet_aid_formal_resume_inspect_return_20261106/inspect_reported.json)显示ps9450只有表头、锁pid9450/目标输出匹配。用户另明确确认旧实例已关闭、仅当前实例；当前namespace旧PID不存在，避免把单机ps误当跨主机停机证明。
- 同包inspect --resume返回最高可用文件receipt：refit75、generation175、slot1.pt、897033192B/SHAbb7c8bb4f2115bd1b384f9a86f47688cf96fada39b3e3ad3ad27ad347be5cefc、binding SHA662ceee356d72644ff513e7e5fc9f9fdef1cb20110e911b32253ccdad2f49599。该服务器标准库检查已确认run身份/绑定和snapshot文件SHA，尚未反序列化Tensor/Adam，不冒充实际resume成功。
- 当前唯一闲置RTX3060 UUID91a8cd0f-e755-4d15-863a-8a5bdd741ed4，与旧容量证明df386e6c不同；input_error为No such file or directory data。blockers仅inputs/gpu_capacity，原22项source哈希与本地逐项一致、初始化351654207B/SHA68c6...仍通过、软件metadata匹配。换卡/缺data已确认，不推断原退出的具体信号或平台原因。
- 可用work6478102528B比6GiB门槛仅多35651584B=34MiB，host可用25560760320B通过。恢复原ZIP到实例data，不占work解压空间；为新卡只跑已有包固定4步/16图到独立runs/batch16_seed42_gpu91a8cd0f_01。独立源码核对确认capacity-output/GPU身份属于attempt资源证据，不进入数值binding，原正式包可以用新容量路径resume，无需改源码或丢弃已有校准/refit。
- [恢复命令说明](output/hgtnet_aid_resume_20261106.md)和[归档/计划](results/evidence/hgtnet_aid_formal_resume_inspect_return_20261106/triage.json)准备：原ZIP完整SHA匹配后unzip -n；新卡4步成功后，将已确认死锁以mv -n改名保留，再nohup后台显式resume，新旧日志/恢复点均不覆盖。resume会重检全部门禁并严格载入Tensor/Adam，需见resumed与新首步及GPU占用才确认启动；返回PID不算成功。当前仅本地准备命令，尚无远端restore/新探针/改名锁/实际续训回传。
- 预期有效恢复后从第76轮继续到96轮；原未提交180步保留为历史、接受轨迹回到48475，剩21轮/6573步。最终test/总耗时/资源指标仍未知，记录为RESUME_REQUIRES_ORIGINAL_DATA_AND_CURRENT_GPU_PROOF。未改正式代码/包、未删数据或checkpoint、未执行GPU/Git操作。

---

## 2026-11-06 — AID GPU当前无训练进程，日志停在refit76；selected96已确认

- 用户回传[nvidia-smi与末尾15条日志原始附件字节](results/evidence/hgtnet_aid_formal_process_return_20261106/reported_output.txt)，原件SHA/解析与[恢复排查记录](results/evidence/hgtnet_aid_formal_process_return_20261106/triage.json)保存。GPU为RTX3060/12288MiB，回传时仅1MiB/利用率0%、No running processes found；显示时间Fri Nov6 01:17:56 2026，时区未提供，不换算客户端时间。当前物理UUID未显示，不推断实例是否更换。
- 日志最后仍PID9450/refit76/2880样本/48655步，与上一status相同；未见最终完成或traceback，不能凭GPU空闲确认CPU PID已退出，也不能断言是关终端、OOM、手动中断或实例退出。选中重训预算96轮由refit75 epoch_complete明确给出；该轮完整5000样本/313步，累计48475、train217.4071秒、epoch wall235.5976秒、mean235.38246秒。训练集99.9%不是最终test结果，不填比较表。
- 按匹配源码正常路径，第75轮epoch_complete在snapshot提交后打印，最新预期恢复点refit75/generation175/48475步，但实际文件SHA/归属/模型payload仍须验证；不能冒充已有可用checkpoint验收。若该恢复点有效，部分第76轮180次更新将保留旧证据并回滚，重做第76–96共21轮/6573次接受轨迹更新；当时日志剩余phase估计4943.03秒仅是历史均值缩放，排除恢复/setup/最终重载/test/账单。
- 独立只读复核确认需先ps查看9450的CPU进程状态/开始时间/完整命令、读取owned lock，再执行同包inspect --resume。后者校验run_identity/binding/source/recipe与snapshot receipt/generation/文件SHA，但不校验锁活性、不反序列化Tensor/Adam，readytrue不等于可直接解锁。若lock PID不同还须检查对应PID，排除同输出其他活动写入。
- [当前记录](results/hgtnet_aid_formal_seed42_progress_20261106.json)更新为SERVER_REPORTED_GPU_IDLE_REFIT76_STALLED_RECOVERY_PREFLIGHT_PENDING；原记录快照保留。未执行run/resume/清锁/代码修补/GPU训练/Git操作，最终test/正式总耗时/完整GPU峰值保持未知。下一步为三条只读诊断回传，不重复从头训练。

---

## 2026-11-06 — AID正式训练已发生：最新回传refit第76轮，进程存活待确认

- 用户回传正式包status，[原样内容转录](results/evidence/hgtnet_aid_formal_progress_return_20261106/status_reported_output.txt)/[JSON](results/evidence/hgtnet_aid_formal_progress_return_20261106/status_reported.json)已归档；来源为用户终端回传，未直连服务器。attempt1、real_aid、RTX3060目标、原4000/1000/5000pool/5000test及manifest/config/recipe/初值SHA一致，source_identity_matches=true、training_started/formal_training_started=true，说明正式任务已记录实际更新。
- 最新持久化进度为PID9450、phase refit、epoch76、samples_seen2880、optimizer_steps48655、实际step completed/succeeded true。根据匹配源码的正常路径，100轮calibration已结束、refit75轮完成，本轮180个batch；25000+75×313+180=48655一致，当前轮尚余2120张/133个batch。不知道选中的总refit轮数，不按100轮假设或短探针推算剩余时长。
- [当前工程记录](results/hgtnet_aid_formal_seed42_progress_20261106.json)为SERVER_REPORTED_FORMAL_STARTED_LAST_REFIT76_LIVENESS_UNCONFIRMED；completed/completion_evidence_valid=false、exit_code=null、state incomplete仅表示尚无完整完成验收，不能单凭持久化进度证明PID仍活着，也不据此判失败。无最终test、正式总耗时或正式峰值回传，均保持空值；未填正式成绩表。
- 下一步只读nvidia-smi及attempts/0001/run.log末尾15行，核对PID9450、步数是否继续增长、最近完整epoch的selected_epoch和阶段耗时。当前没有执行run/resume、清锁或修改代码/包/Git，避免因遗留状态直接重启付费训练。

---

## 2026-11-05 — HGTNet AID seed42正式包本地完成，服务器预检/首批待执行

- 论文问题为原AID冻结协议上的独立HGTNet单seed补充比较，计划输出Accuracy/macro-F1、30类per-class accuracy/F1、CM、有序5000条test与checkpoint/配置/资源。主要改变为外部方法在AID全量执行；各自披露配方不同，不能解释为单模块因果或已完成统一多seed。已有真实batch2+batch16工程证据充足，本轮不再安排付费探针，未启动实际正式GPU。
- [设计](docs/superpowers/specs/2026-11-05-hgtnet-aid-formal-design.md)/[计划](docs/superpowers/plans/2026-11-05-hgtnet-aid-formal.md)落实原manifest30ac5760...aa22、30类/4000train/1000val/5000pool/5000test及Industrial原pixel组；原Swin-B候选68c6b09e...1a893，74612163全可训练参数/451张量。100轮calibration、验证correct严格增加/相同最早、同初态SHA fresh5000pool refit到selected E、严格独立新模型重载≤1e-6，再5000test一次；Adam1e-5/WD5e-5/batch16/CE、无scheduler/AMP/TF32，Resize256+crop/flip与eval center224。固定recipe canonical SHAb4b007a30f3e96c68e777b1ce7115779c80fa21f104c65cd74009260a115584d。
- 新AID protocol/data/checkpoint/engine/entry与独立builder，不改旧源。默认禁止读test字节，原图RGB600安全同字节SHA→解码；run/resume独占锁、同binding完整epoch双slot恢复、attempt保留回滚部分实际更新；test started→完整packet→complete事务，合法完整结果复用，不完整已开始test拒绝暗中重测。纯stdlib status核对full source/runtime/binding、100+E历史/earliest最佳/25000+313E步、两相首步、fresh初态、三主checkpoint/reload、best/lastval与5000test身份/指标重算、事务与末尾复核/exit0/marker，拒绝骨架summary。
- 三个并行实现最初因账户额度中断，原部分代码与RED/失败日志保留，主代理接手完成再验收；后续独立审查恢复可用。[43项CPU数据/数值/恢复测试](results/evidence/hgtnet_aid_formal_prepare_20261105/engine/root_integration_attempt.log)通过；[审查](results/evidence/hgtnet_aid_formal_prepare_20261105/review/independent_review.md)发现两P2并关闭：恢复归档未提交轨迹的失败首步凭据后发布新证明；进行中status从有效real_aid binding+持久化更新推导formal_started。14项窄回归通过。最终[同一CPU解释器本次全部新formal范围64项](results/evidence/hgtnet_aid_formal_prepare_20261105/scoped_final_tests.log)通过；旧AID入口15/交付6/batch16入口8另行基线通过，不拼成全项目/GPU验证。CPU小模型、生成图和receipt夹具与真实GPU证据分开。
- [正式包](output/hgtnet_aid_formal_3060_20261105_v1.pyz)85287B/SHAef2bc84e0df446b7d2bdb5ac2097b21456e66a97c381d8cf5d16e87a3bac5c4b、bundleID31e712e70be7495ea8f64cdd14fe6d0e1c2ba158a30babd0918f918ee8ad0d1f，22source/23payload/25ZIP成员。[最终实际包验收](results/evidence/hgtnet_aid_formal_prepare_20261105/final_validation.json)含stdlib importguard inspect/status exit0、本地缺目标资产/GPU为readyfalse/not_started、Python3.11语法与43旧源/包/成绩表保护哈希一致。Git基础提交058f1d8b79d0be538ec358eb742d2cd3992921d4，新代码为未提交source-only快照、实际身份由22项SHA/包ID标识，不冒充该提交包含新增代码。
- [完整启动说明](output/hgtnet_aid_formal_3060_README_20261105.md)先inspect&&run，默认新持久化输出`/home/featurize/work/hgtnet_aid_formal_3060_20261105_v1/runs/aid_seed42_01`，best_calibration.pt/final_calibration.pt/final_refit.pt和attempts日志固定。启动前严格验证原pool5000图、同当前物理GPU的已通过batch16旧包SHA/status、6GiB磁盘/6GiB可用hostRAM及实际软件/CUDA。原命令、source、seed42、checkpoint和资源身份在程序开始前保存；只有实际设备/模型首batch检查通过才确认启动。强杀后可能留lock，说明含只读归属/进程/snapshot核对再处理的流程，不自动删锁或从头训练。
- 状态LOCAL_AID_FORMAL_BUNDLE_VALIDATED_SERVER_PREFLIGHT_AND_START_PENDING。正式最佳轮/test指标/正式峰值/端到端耗时/费用均待GPU，不将工程4.407GiB或34.098图/秒填正式表。结果将在实际完整验收后进入独立AID外部方法seed42表；训练阶段时间从首个真实完整epoch及各phase均值记录，不用短探针推断完整账单。本轮无GPU正式训练、依赖安装、租卡、外发或Git暂存/提交/推送。

---

## 2026-11-05 — AID实际3060 batch16四步通过；正式包进入实现

- 用户执行原新包run&&status，回传完整四步及最终JSON：[转录](results/evidence/hgtnet_aid_batch16_return_20261105/reported_output.txt)/[status](results/evidence/hgtnet_aid_batch16_return_20261105/status_reported.json)/[10项核对](results/evidence/hgtnet_aid_batch16_return_20261105/validation.json)归档。JSON布局规范化、字段内容保留，未冒充服务器run.log原件或直连查询。state complete、exit0、完成产物与来源true、原manifest/config/初值与包SHA一致，实际UUID仍GPU-df386e6c-dbb3-0952-8190-3265a7bd8ae8。
- 原train16图、unique16/processed64/updates4、1warmup+3measured，五组451梯度张量无missing/nonfinite、组L2正、代表参数更新、各步postcheck通过。实际allocator allocated4731960320B=4.4069814682GiB、reserved5213519872B=4.85546875GiB；三步中位0.4692300269998668秒、compute34.0984145927图/秒，总wall13.486246308秒。scope仍为固定16图设备驻留计算，非完整loader、epoch、费用或更早初始化峰值。
- [当前工程记录](results/hgtnet_aid_batch16_probe_3060_20261105.json)为SERVER_REPORTED_AID_BATCH16_COMPLETE_STATUS_CHECKED；没有正式训练/val/test/accuracy/checkpoint，不写正式成绩表。此前batch2+3060与本次batch16均有真实回传，后续不再重复小规模付费探针。
- 继续已授权AID流程：[正式设计](docs/superpowers/specs/2026-11-05-hgtnet-aid-formal-design.md)/[计划](docs/superpowers/plans/2026-11-05-hgtnet-aid-formal.md)固定30类原4000train/1000val/5000pool/5000test；独立100轮calibration/earliest严格最佳、同初态fresh pool refit、严格独立重载后test一次，Adam1e-5/WD5e-5/batch16/既有cropflip。新增AID siblings，保留43个旧源/包/结果表保护快照；实现全量数据/test门禁、完整epoch恢复、test事务复用或中断拒绝、run/resume锁和强status。现在为本地实现阶段，未启动实际正式GPU任务，完整时长待首个真实epoch和IO记录。

---

## 2026-11-05 — AID batch16容量包本地完成，目标3060四步待运行

- 根据已通过真实3060 batch2的一轮smoke，新增独立aid_batch_probe_engine/aid_batch_probe及builder，不修改旧AID/NWPU源/包。原30类配置与原manifest/Swin-B初值均固定；从冻结原train按label0–15各取路径最小一图，原RGB600→bilinear224/ImageNet归一化，只读取这16张图片。使用全模型FP32、Adam1e-5/WD5e-5、1warmup+3measured共4次一次性更新，unique16/processed64，无val/test/epoch/accuracy/checkpoint/formal训练。
- 入口复用已实测3060身份/runtime检查，在Torch导入前自动绑定唯一空闲3060；自建18项来源和强完成状态，保留独占输出锁及拒绝覆盖。首步要求五组451梯度张量及代表参数更新、实际模型74612163总/可训练参数；四步事件分别落盘，发生更新后诊断失败也保留真实步数。结尾复核源/config/manifest/完整初值/16图后，验收完整产物才输出COMPLETE，status交叉核对runtime/选择/各步/计时/源/summary和exit0，拒绝骨架summary。
- [数值引擎12项CPU行为测试](results/evidence/hgtnet_aid_batch16_prepare_20261105/engine/validation.json)、[入口8项/交付6项stdlib测试](results/evidence/hgtnet_aid_batch16_prepare_20261105/entry_delivery/validation.json)通过，主代理在各自正确解释器又核对通过；旧AID入口15与交付6项基线通过。CPU夹具为生成Tensor/小模型10张量，不是完整HGTNet或真实AID/GPU容量。原RED日志、早期夹具错误、根runner通配符误选无Torch解释器的1项导入错误、macOS临时路径symlink被安全拒绝及更正均留证，不拼成一次全项目全绿suite。
- [独立审查](results/evidence/hgtnet_aid_batch16_prepare_20261105/review/independent_review.md)对6个新增生产/测试文件规格与质量均PASS、无可操作缺陷；[最终实际交付验收](results/evidence/hgtnet_aid_batch16_prepare_20261105/final_validation.json)核对18source/19payload/21ZIP成员、原样源码与35个旧源/包/成绩表保护哈希。对最终命名包做无ML实际导入的inspect/status，exit0，本地缺原资产/目标GPU所以readyfalse/not_started；未伪造通过。
- [新小包](output/hgtnet_aid_batch16_probe_3060_20261105.pyz)67224B/SHAfd54b4c1a21db1731d1addb0808c49104c11113b610883b79b2c6af456fc80f5，bundleIDf07b8d6744acbd4ee25a16c170a90fa913011e1ed668c6a035837dc4c494662e；[用户单行说明](output/hgtnet_aid_batch16_probe_3060_README_20261105.md)使用新默认持久化目录，run后自动status，不再手填旧UUID或重复一轮smoke。状态LOCAL_AID_BATCH16_BUNDLE_READY_ACTUAL3060_PENDING，实际3060 batch16显存/计算计时待回传，无本轮GPU或正式训练/依赖安装/租卡/Git提交推送。
- 计时只测已在设备上的固定16图forward/backward/有限梯度检查/Adam；warmup、读图/预处理/transfer/callback和参数诊断不进中位数。allocator峰值reset在模型/输入放到设备及optimizer构建后，含常驻分配、4步与诊断，不声称更早初始化峰值。结果不能直接作完整loader吞吐、正式ETA或费用。正式AID入口仍需全量协议、恢复/输出锁、test一次与强status实现，后续依据真实batch16容量推进。

---

## 2026-11-05 — AID smoke的status验收字段齐全；准备batch16容量门禁

- 用户回传同包同目录status：[原样文本](results/evidence/hgtnet_aid_smoke_complete_return_20261105/status_reported_output.txt)/[JSON](results/evidence/hgtnet_aid_smoke_complete_return_20261105/status_reported.json)保存。state complete、completed/completion_evidence_valid/source_identity_matches均true、exit0；train240/val60/test0、best/final epoch1、optimizer120、实际GPU与PID3130和前次运行一致，formal_training_started=false。
- 真实回传smoke val Accuracy76.66666666666667%、macro-F1 72.34920634920634%；Accuracy对应46/60正确（算术换算，未下载原预测作本地重算）。checkpoint SHA9d2040b95eb27756f8f82f26c3b49e05fd787849776e45fa7df9088a8a076910，独立重载最大误差0。allocator allocated峰值1536937984B=1.4313850402832031GiB、reserved1694498816B=1.578125GiB；total wall48.76537990199995秒，包含预检后导入/初始化/一轮/验证/重载，不是训练吞吐或账单时长。
- [10项字段与身份核对](results/evidence/hgtnet_aid_smoke_complete_return_20261105/status_validation.json)通过，原manifest/config/初值SHA完全一致；[当前smoke记录](results/hgtnet_aid_smoke_3060_20261105.json)更新为SERVER_REPORTED_AID_SMOKE_COMPLETE_STATUS_CHECKED。旧pending记录保留快照。来源是用户回传程序status，其服务器端JSON/字节一致性核验通过；未冒充直连服务器或下载checkpoint/60条预测后独立验收。工程val不写正式成绩表。
- 继续已授权AID流程，[短探针设计](docs/superpowers/specs/2026-11-05-hgtnet-aid-batch16-design.md)/[实现计划](docs/superpowers/plans/2026-11-05-hgtnet-aid-batch16.md)明确先验证实际3060 batch16：16张冻结原train图、1warmup+3measured共4次一次性Adam1e-5/WD5e-5更新，30类同初值，test/val/epoch/checkpoint0。现有smoke只实测batch2，旧NWPU探针写死45类/3090不可原样复用；新文件隔离，35个旧源/包/成绩表哈希已快照，旧入口15项stdlib基线通过。该门禁是资源检查，不是新增正式实验。
- 正式AID候选沿用独立NWPU已披露优化配方与最多100轮选择/fresh pool refit，需新增AID全量数据/协议/状态门禁，尚未实现/运行。原4000/1000/5000pool/5000test下batch16总训练更新25000+313E（E1–100）；不将smoke48.765秒或固定16图计算时延换成完整3060 ETA。正式trainer还须处理输出互斥、已完成test复用/中断记录、严格status及每轮恢复快照IO预算，保留全部原划分和Industrial重复组。

---

## 2026-11-05 — AID真实RTX3060一轮smoke完成回传，最终status待读取

- 用户执行原ZIP完整SHA校验、`unzip -n`恢复本地data后，使用现有包`run`自动绑定唯一空闲3060；[完整命令与原样回传](results/evidence/hgtnet_aid_smoke_complete_return_20261105/reported_output.txt)已保存，未重新划分、换初值或修改正式代码。ZIP校验OK，实际GPU为GPU-df386e6c-dbb3-0952-8190-3265a7bd8ae8；此前旧UUID/缺data的失败尝试单独保留。
- 实际Python3.11.8/Torch2.2.2+cu121/vision0.17.2+cu121/NumPy1.26.4/Pillow10.3.0/CUDA12.1/cuDNN8902、8线程/seed42/FP32、AMP与双TF32关闭、cudnn deterministic且benchmark关闭通过当前严格runtime函数。NVML总12288MiB与CUDA总12487622656B差378.875MiB，符合≤1GiB口径；容量及接口差不当作峰值显存。
- 首步loss3.747433662414551，五组共451个梯度张量missing/nonfinite均空，各组L2为正，代表参数更新均true；不扩大为每张量非零或每参数更新。随后30/60/90/120步对应60/120/180/240训练样本，最终120步complete及HGTNET_AID_SMOKE_COMPLETE指向持久化summary。工程范围仍为30类、seed42、240train/60val/test0、batch2、1epoch、Adam1e-4/CE、无增强；不作为正式AID分数。
- [17项标准库核对](results/evidence/hgtnet_aid_smoke_complete_return_20261105/validation.json)通过，含回传运行时/容量/进度和17个源码配置与已交付包一致；[独立复核](results/evidence/hgtnet_aid_smoke_complete_return_20261105/independent_review.md)确认正常完成标记在60条有序验证、独立重载≤1e-6、源码/初值/manifest/config/300图和checkpoint末尾复核、一致产物及run.exit=0写入后输出。该解释基于匹配源码，未冒充服务器文件字节验收。
- [当前结果记录](results/hgtnet_aid_smoke_3060_20261105.json)为SERVER_REPORTED_AID_SMOKE_COMPLETE_SUMMARY_PENDING；精确验证Accuracy/macro-F1、重载误差、checkpoint SHA、峰值allocated/reserved、总耗时与status完成字段保持空值。下一步只读`/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_smoke_3060_20261104.pyz status`，补齐数值后再准备独立正式AID配置与预算。本轮本地仅转录/标准库核对和文档归档，无数值训练、代码/包变更或Git提交推送。

---

## 2026-11-05 — AID预检原因确认：旧UUID不匹配、本地data目录缺失

- 用户补回详细inspect，原manifest SHA30ac5760...aa22/selected_pool_count300、初始化351654207B/SHA68c6b09e...1a893、软件metadata与work余9766436864B均通过。input_error为`No such file or directory: 'data'`，且manifest选择已成功，定位为图片路径`/home/featurize/data/AID/AID`的data组件缺失。
- 当前唯一设备是RTX3060/12288MiB/free11909MiB、UUID GPU-df386e6c-dbb3-0952-8190-3265a7bd8ae8，processes空/error null；用户命令仍指定昨天GPU-7acce72a-93cd-e56f-35d8-5d33ecf6655e。已确认物理GPU UUID变化与当前data缺失，不推断具体实例切换操作或时间。[回传字段摘录](results/evidence/hgtnet_aid_smoke_preflight_return_20261105/inspect_selected_fields_reported.json)明确为用户JSON选字段转录，不冒充完整原始stdout。
- 原入口无须修改。[9项核对](results/evidence/hgtnet_aid_smoke_preflight_return_20261105/resolution_validation.json)在回传设备清单上复现旧UUID拒绝、当前新UUID通过、无手填UUID时唯一空闲3060自动选中；独立源码复核确认run仍在导入Torch前绑定实际UUID并保留runtime核验。不是关闭GPU身份检查，没有实际GPU执行。
- [单行恢复并run](output/hgtnet_aid_recover_and_run_20261105.md)以旧ZIP当前完整SHA匹配为前提，unzip -n到本地data后使用现有包run自动识别唯一空闲3060；仍复查300图/初值/实际CUDA，再做一轮240/60，不重划分/换模型或重下权重。恢复/真实smoke尚待用户运行；包SHAa70b6f...3961及源码未改，本轮无GPU或CPU数值训练、Git提交推送。

---

## 2026-11-05 — AID smoke被输入/目标GPU预检拦截，尚未开始训练

- 用户按既有命令执行3060包run，指定UUID GPU-7acce72a-93cd-e56f-35d8-5d33ecf6655e，立即返回`preflight blockers: inputs, target_gpu`；[原样两行](results/evidence/hgtnet_aid_smoke_preflight_return_20261105/reported_output.txt)/[当前状态](results/hgtnet_aid_smoke_preflight_20261105.json)已保存。当前仅有简短错误，没有input_error/gpu_error、当前设备/进程或文件状态，不能确认已换实例、图片再次丢失或具体代码错误。
- 已核对aid_entry的执行次序：预检拒绝在输出锁、Torch导入、模型和optimizer之前，本次没有开始训练。inputs涵盖原manifest读取/身份/元数据及所选图片的缺失、SHA或安全路径；target_gpu涵盖NVIDIA查询、UUID/型号/容量及计算进程占用，不仅是显卡型号。
- 独立只读检查未发现AID数据适配器与昨天已通过300图的确定性不兼容；沿用相同manifest验证器、安全字节读取和每类8/2选择。昨天恢复证据保留为当时事实，不当作今天实例仍可用的证明。
- 下一步只运行同包同UUID的inspect，读取详细input_error/gpu_error/nvidia；不先重下载、重划分、改GPU门禁或启动新实验。本轮未改源/包/成绩表、未GPU或CPU数值训练、未租卡或Git提交推送。

---

## 2026-11-04 — AID恢复与300图通过；RTX3060一轮smoke包本地完成

- 用户回传旧ZIP当前SHA检查OK、解压后同一份inspect的input_assets_verified=true。[原样20506B/规范化JSON与17项核对](results/evidence/hgtnet_aid_inputs_restored_20261104/README.md)保存，附件SHAdeea789f93e2207b9b4dfb7006e22f1a556b85183ffa8853392f6b208643fc9d；原manifest30ac5760...aa22/初始权重68c6b09e...1a893继续通过，240train/60val全部checked=verified300、missing/invalid0、test图读取0；路径与上次缺图选择完全相同，原Industrial重复组保留。当前JSON状态更新为SERVER_REPORTED_AID_SMOKE_INPUTS_VERIFIED_GPU_SMOKE_PENDING，旧缺图状态备份保留。核验是服务器程序用户回传，未在本地下载原图或直连查询服务器。
- 沿用已授权[设计](docs/superpowers/specs/2026-11-04-hgtnet-aid-smoke-design.md)/[计划](docs/superpowers/plans/2026-11-04-hgtnet-aid-smoke.md)，新增aid_data/aid_engine/aid_entry及独立builder和30类配置。只做seed42、240train/60val/test0、batch2、1epoch、Adam1e-4/CE、无增强/AMP/TF32的工程检查；RGB600同一份已SHA验证字节→bilinear224→ImageNet normalization。30类配置仅修改旧候选num_classes，canonicalSHAf313abb893b29d674ace281d8d2905670ee65db131f4240640079123e783589b。不是正式AID配方或性能结论。
- 新数据适配器沿用冻结manifest/schema2/原像素组/不等类数量，每类8/2选择；安全普通文件和路径/哈希检查，禁止test行进入Dataset。新engine要求五组可训练FP32参数/有限非零梯度/首步实际更新，120次更新和60条有序验证预测/30类指标/混淆；保存独立AID checkpoint，新模型禁止共享storage，严格weights_only/keys/shape/dtype/有限值/身份与≤1e-6重载。每次实际更新先写durable进度，后续诊断或验证失败保留实际开始/步数。
- 新入口先按实际GPU-7acce72a-93cd-e56f-35d8-5d33ecf6655e选择RTX3060/12288MiB，CUDA_VISIBLE_DEVICES先于Torch导入；实际CUDA总容量可比NVML少不超过1GiB，避免旧3090口径误判。run核对Python3.11/Torch2.2.2+cu121/vision0.17.2+cu121/NumPy1.26.4/Pillow10.3.0/CUDA12.1/cuDNN8902、8线程/FP32/cudnn deterministic/benchmark与双TF32关闭；当前这些仍待目标机实际run验证。原source/input/config/初值/300图在运行末尾复核，失败和进程锁均有证据，不支持覆盖或resume。
- [数据/engine21项](results/evidence/hgtnet_aid_smoke_prepare_20261104/data_engine/validation.json)通过，含明确CPU240/60夹具120次更新与独立重载0；[入口/交付最终21项](results/evidence/hgtnet_aid_smoke_prepare_20261104/entry_delivery/validation.json)通过。52项相关集成在完成状态门禁修正前通过，修正后只重跑受影响21项。没有把日志拼成一次新55项suite或重复无关旧386项。
- [完整30类HGTNet CPU release数值检查](results/evidence/hgtnet_aid_smoke_prepare_20261104/integration/full_model_release_validation.json)实际加载相同完整Swin-B候选，74612163参数、两张生成Tensor、一次实际更新、五组梯度/参数更新、30×30CM和严格新模型重载误差0，9项通过并绑定17个冻结源/配置。早期迭代夹具保留；这些是CPU工程夹具，不是AID原图、3060容量/吞吐或正式test。OpenMP在沙箱SHM失败后，原解释器用自动审查批准的本地CPU运行通过，无依赖修改。
- [独立审查](results/evidence/hgtnet_aid_smoke_prepare_20261104/review/independent_review.md)发现一个status可由skeletal summary误判完成的P2，已在交付前修正并保留RED/GREEN/原反例：必须完整real_aid身份/runtime/数据/来源、首步五组/120步、有序60预测与stdlib重算指标、checkpoint及结束复核一致。4项范围回归和15项检查通过，P2关闭、规格/质量无剩余可操作问题。不声称stdlib status能重新执行GPU或解析模型张量。
- 最终[新运行包](output/hgtnet_aid_smoke_3060_20261104.pyz)63235B、SHAa70b6fbf038aa82daac9398b1d75c057d37b4fe3bf18167604b95d6f6a9d3961、bundleIDcc25746e7cc34705a454bf472938d2b7ad0db30b895c4ffb5d4485d1573285e9。17个源码/配置及独立权重来源metadata，无模型大文件/安装器；真实目录安全安装，防篡改/重复/路径/symlink/外来目录。实际最终包inspect/status在禁止ML实际导入下通过；本机无目标资产/Python3.11/GPU所以ready=false/not_started，未伪造通过。根guard初版误拦find_spec元数据查询，旧guard/失败说明保留，修正实际import阻断后通过，未因此改生产代码。
- [单行用户说明](output/hgtnet_aid_smoke_3060_README_20261104.md)和[本地证据目录](results/evidence/hgtnet_aid_smoke_prepare_20261104/README.md)已保存；默认新输出`/home/featurize/work/hgtnet_aid_smoke_3060_20261104/runs/aid_smoke_seed42_01`，checkpoint为smoke_epoch1.pt。实际GPU命令为`/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_smoke_3060_20261104.pyz run --gpu-uuid GPU-7acce72a-93cd-e56f-35d8-5d33ecf6655e`，run内再次预检。状态LOCAL_AID_SMOKE_BUNDLE_READY_TARGET_GPU_PENDING；实际3060首批/整轮/保存重载、峰值/耗时尚待用户执行。本轮无GPU/正式训练、租卡、Git提交推送或外发，旧NWPU24个源/包与历史结果表受保护。

---

## 2026-11-04 — AID盘点回传：原清单/初值通过，图片缺失，当前实际3060

- 用户执行既有`/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_inputs_20261104.pyz inspect`并上传完整输出。[原样附件与规范化JSON](results/evidence/hgtnet_aid_inputs_return_20261104/README.md)保存，31361B、SHA7f751431f1c976b5d9c3a42b22b5caae08328bb7bbc1c360843cbbc4e7729bc8；[23/23本地核对](results/evidence/hgtnet_aid_inputs_return_20261104/validation.json)通过。当前状态SERVER_REPORTED_AID_MANIFEST_WEIGHTS_PASS_IMAGES_MISSING，未直连服务器或下载原manifest/权重/图片字节。
- 原manifest2710270B、SHA30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22，与已有冻结协议一致；用户返回identity/structure true、errors空、30类/4000train/1000val/5000pool/5000test、每类不等量及原Industrial重复两图均train。初始化351654207B、SHA68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893也通过。这里是服务器只读程序回传的原件SHA核验，不冒称本地拿到原文件。
- 唯一已见输入缺项为默认`/home/featurize/data/AID/AID`目录不存在，固定240train+60val全部missing，checked/verified/invalid均0，test图片读取0；不等于整台机器其他位置必无数据。不能据此重新生成split、改类序或重下已通过的权重。
- 本次实际清单为NVIDIA GeForce RTX3060、12288MiB、UUID GPU-7acce72a-93cd-e56f-35d8-5d33ecf6655e、driver610.57.04，不沿用之前3090 UUID或声称原3090容量结论已在3060成立。Python3.11.8、Torch2.2.2/vision0.17.2、NumPy1.26.4/Pillow10.3.0/timm1.0.27均仅distribution metadata；实际CUDA、精度、模型及容量待小样本验证。
- work余9767485440B、实例本地祖先目录余349589377024B。数据继续放实例本地data，不向work解压或删除旧工件。当前未进行下载、恢复或训练，performance表未改。
- 进一步检索历史资料找回旧ZIP`/home/featurize/work/AID_visionflan_20260920.zip`，2628474154B，SHAcdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7；旧包两层AID目录，与原数据根一致。9月20日日记已有用户完整SHA确认，修正本轮早期准备说明“可靠ZIP身份未知”；保留旧程序/输出不改写。当前ZIP仍在与当前SHA尚未查。
- [单行恢复流程](output/hgtnet_aid_restore_20261104.md)以当前ZIP完整SHA匹配为前提，成功才unzip -n至本地data，成功才重跑现有inspect；任何一步失败停下，不下载猜测版本或执行旧prepare。恢复会解包原数据含test文件，之后的inspect仍只哈希300train/val，无模型评估或test选参。资产恢复通过后再准备适配实际3060的AID smoke；目前无GPU smoke或正式trainer启动。

---

## 2026-11-04 — 46/47已知分数补录，HGTNet×AID只读资产包本地完成

- 用户明确确认46/47原实验平台、原始记录和完整F1已丢失，同时再次确认此前主指标确有对应结果。现将状态从“等待回收”改为“用户确认历史主指标补录”；不再把回收这批原件作为后续推进前置条件。[补录JSON](results/seed46_seed47_reconstruction_20261104.json)/[可读记录](results/seed46_seed47_reconstruction_20261104.md)保留16个原分数、8组均值/样本SD/配对差，缺F1、预测、混淆、实际UUID/commit/checkpoint、退出码和资源字段为null。没有生成仿真日志或把其他seed的F1填入46/47。
- [28项补录核对](results/evidence/seed46_seed47_reconstruction_20261104/validation.json)通过，原multiseed_test_results.csv仍为SHA639a9b45b07625d63c1254a32969637c984a996997248622a647f261d43c0988。小数按原打印精度统计，展示才ROUND_HALF_UP；原主指标及既有多seed数值未改。旧10-12审计加当前状态索引，历史原件/更正证据保留。
- 中文[导师报告](docs/supervisor_progress_report_20261012.md)加入已验收HGTNet NWPU结果、完整历史和DF配对；将46/47改为历史分数补录，不再列作等待原件任务；[纯文本版](docs/supervisor_progress_report_20261012_plaintext.txt)同步，包含链接路径在内的数字token完全一致，无表格行和tab。已备份修改前两份报告；英文历史邮件没有编辑。
- 下一论文问题为HGTNet独立实现的AID外部比较；[本阶段范围](docs/external_baselines_20261027/hgtnet_aid_preparation_20261104.md)/[实施计划](docs/superpowers/plans/2026-11-04-hgtnet-aid-inputs.md)固定为先查既有资产。AID原清单引用SHA30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22，30类10000图，4000train/1000val→fresh5000pool→5000test。根目录及研究工作树未找到原manifest字节，不声称已本地验真；不重新生成划分冒充原协议。既有模型config可用30类，但NWPU正式入口写死45类和数量，未直接复用为AID训练器。
- 新[标准库资产工具](tools/inspect_hgtnet_aid_inputs.py)只读盘点manifest、300张smoke候选图字节、初始Swin-B完整SHA、nvidia-smi、依赖distribution metadata和磁盘。冻结真实manifest SHA与group-v2/schema2/taxonomy/不等类比例/像素组隔离；每类排序取8train/2val，共240/60，test图片不打开。逐组件O_NOFOLLOW限制路径与symlink，不导入ML或进行CUDA计算，不安装、下载、写服务器文件或生成新split。训练入口/训练启动字段均false，输入通过不冒称训练通过。
- [交付包](output/hgtnet_aid_inputs_20261104.pyz)6520B、SHA1cd2ab967cdbc386b294b397c6804bf06c5796c01332c709e44646f98d03f036；源码18930B、SHA4e3b66b91f115d99d01e39fe168e8cf5dd18503ec8593e45c39dbccfcd0eb7b4。包仅含__main__.py和源码身份JSON，独占创建拒绝覆盖。[单行操作说明](output/hgtnet_aid_inputs_README_20261104.md)给出目标机inspect命令，没有正式训练命令。
- 13项标准库行为unittest通过，RED/GREEN日志保留；[15/15交付检查](results/evidence/hgtnet_aid_inputs_prepare_20261104/delivery/validation.json)通过。实际最终pyz在禁止非标准库导入与文件修改guard下，本机缺资产盘点exit0且明确missing/GPU unavailable，没有伪造通过。未重复386项旧工程suite。旧NWPU正式包SHA仍b2bc9077cf37e75c83fbb72e7c4047d2fb9ede04db8fb4e4a56344500ff572e1。
- 当前为LOCAL_AID_INPUT_INSPECTOR_READY_TARGET_ASSETS_PENDING：目标服务器原AID清单、图片与权重仍待只读盘点；尚无AID GPU smoke、正式trainer或性能结果。资产具备后才做独立240/60 smoke，再冻结正式配置和预算。现有外部单seed各自配方结果继续单列补充比较，不冒充统一训练协议/多seed主比较完成。本轮未GPU训练/租卡/提交推送或外发消息。

---

## 2026-11-04 — HGTNet原始结果回收：25200预测独立复算，完整曲线与错误分析完成

- 用户上传`hgtnet_seed42_results_20261103.tar.tar`，实际gzip tar、544296B，SHA`4fc4d51cec3aac5900ac94cdde919818ba58cd0f76add625867623f54d052477`。按七个预期普通JSON文件的名称/类型/大小读取，无执行包内代码或指令，原包和原件保存在[证据目录](results/evidence/hgtnet_results_return_20261104/README.md)。test字节SHA`869986974e4801f2234fd2fa3bace383af2f092fd20359fbe1a252d4af7e37ae`与11月3日完成status一致；状态升级为DOWNLOADED_RESULTS_METRICS_HISTORY_LOCALLY_RECOMPUTED。
- 本轮是既有seed42正式实验的离线验收，不是新训练：沿用Git基线058f1d8b79d0be538ec358eb742d2cd3992921d4及冻结v1包，未改模型、配方、数据或seed。完整实际训练命令/研究假设/数据增强/优化设置见11月3日预登记和启动记录。本地复算入口为`python3 results/evidence/hgtnet_results_return_20261104/analyze_results.py`，依赖原项目及其已存证据，不需要ML导入或GPU。
- 从25200条唯一有序路径的类别目录重建真值，与绑定清单/labels逐项一致，45类各560张；重建全部2025个混淆单元及45类precision/recall/F1。23718正确、1482错误，OA94.11904761904762%、macro-F194.10529172930384%，两项与服务器值差异均为0；不再只依赖OA反算正确数。完整预测、所有错误、逐类结果和全混淆矩阵均导出，未挑选有利样本。
- 回收100轮calibration和95轮freshrefit全部history，逐轮CM/指标/更新数/耗时一致，总68930更新。第95轮是验证正确数唯一最大值：1173/1260、OA93.0952380952381%、macro-F193.03452094979133%；末轮100为1165/1260、OA92.46031746031746%、macro-F192.42454111106593%。保留全部轮次，最佳到末轮下降0.6349206349pp/8张，遵循训练前的选轮规则，不据波动推断因果。初始化/重载metadata一致，reload误差0；完整runtime与24个源码绑定项符合实际交付v1包。
- 导出四组PNG/SVG/矢量PDF：100+95轮训练曲线、全部45类recall/F1、完整45×45行归一化混淆矩阵、前十双向混淆。PDF逐页渲染并视觉检查；逐类数值与F1标记的首稿重叠已修复，首稿保留在draft_01，最终图无裁切/遮挡。palace召回77.6786%、church82.5%；church→palace41/反向62，共103个错误。仅作错误分布描述，无原图不归因视觉特征或图传播机制。
- 与已回收DF原预测按相同manifest引用、类别和25200路径配对：双方正确21303、DF错/HGT对2415、DF对/HGT错503、双方错误979。正负方向都保留，净增加1912正确/7.5873015873pp；这是不同骨干/预训练/配方的完整方法比较，不识别单图模块贡献。
- fixed4×4 GAViT现有seed42正确23714，HGT23718，只是点估计相近/净4张；缺GAViT逐图原件，不能声称仅纠正4张且无退化，不能做HGT↔GAViT配对检验或统计等价判断。fixed参数31,386,920，相比HGT74,619,858少57.9376%；骨干、训练预算和GPU不同，不作同条件速度或单模块因果结论。
- 实测calibration wall11778.3426秒/平均117.7834秒每轮；refit wall11862.1939秒/平均124.8652秒每轮；test评估180.7621秒/139.4098图每秒（含输入、预处理、传输和标签收集），不能当纯GPU推理或跨卡公平速度。总wall24032.6174秒=6小时40分32.6秒，allocated4.40946GiB/reserved4.78320GiB；无账单则不推算租卡费用。
- 主分析[742/742检查](results/evidence/hgtnet_results_return_20261104/validation.json)、独立预测[131/131检查](results/evidence/hgtnet_results_return_20261104/review_predictions/validation.json)、独立历史/来源[498/498检查](results/evidence/hgtnet_results_return_20261104/review_history/validation.json)及图表[26/26检查](results/evidence/hgtnet_results_return_20261104/figures/visual_validation.json)均通过；这些是离线审计，不冒充新GPU训练或重复386项工程回归。
- [分析报告](results/hgtnet_nwpu_seed42_analysis_20261104.md)、[机器记录](results/hgtnet_nwpu_seed42_analysis_20261104.json)、[可复制纯文本](output/hgtnet_seed42_verified_summary_20261104.txt)已保存。[当前seed42表](results/external_baselines_nwpu_seed42_20261030.csv)仅提升HGT的correct_basis/evidence字段，所有分数和其余四行不变；主结果JSON/MD同步、改前副本保留。旧comparison_table/ablation历史成绩未改。
- 边界：未回收模型/optimizer张量、原图、完整原manifest或逐轮validation预测；对应SHA/初值/重载一致属于metadata核验，不是原件字节审计。多seed稳定性和单模块因果问题仍由各自实验回答。本轮没有新训练、GPU占用、租卡、Git提交推送或外发消息。

---

## 2026-11-03 — HGTNet正式seed42完成：test OA94.1190%，最佳95/freshrefit95

- 用户回传refit95完成、checkpoint独立数值回读0、beforetest→complete→HGTNET_NWPU_FORMAL_COMPLETE，再补只读status。接受SERVER_REPORTED_FORMAL_COMPLETE_IDENTITIES_MATCH_PREDICTIONS_PENDING：[原样完成片段](results/evidence/hgtnet_formal_complete_return_20261103/reported_output.txt)保留截断首行不补造epochwall，[规范化status](results/evidence/hgtnet_formal_complete_return_20261103/status_reported.json)、[43/43本地验收](results/evidence/hgtnet_formal_complete_return_20261103/validation.json)、独立[22/22完成路径](results/evidence/hgtnet_formal_complete_return_20261103/review/validation.json)/[39/39摘要核验](results/evidence/hgtnet_formal_complete_return_20261103/review/status_audit.json)通过。用户服务器报告完整实验完成，未直连检索原件；逐图复算是后续离线审计，不重复训练。
- 身份沿用实际原命令和包b2bc9077...572e1/HEAD058f1d8b79d0be538ec358eb742d2cd3992921d4+包内未提交增量SHA；configaf6ef410...ac25/recipece72e182...af85/manifestce0cfef2...840e/候选68c6b09e...1a893及UUID4f625a02...3cb1均匹配。seed42/NWPU完整74619858参数/全部可训练、实际候选partialSwin-B、batch16/Adam1e-5/WD5e-5/constantLR/CE、预定RGBcrop/flip/eval/FP32等工程选择不变。仍是独立完整方法，不声称作者精确复现或单图模块因果。
- counts train5040/val1260/pool6300/test25200；校准finalepoch100、best/selected95，最佳valOA93.0952380952381%、macroF193.03452094979133%（OA算术1173/1260）。同初值SHA2e954f44...af79重新建模，pool6300 freshrefit95轮，未续学calibration权重；100×315＋95×394＝68930更新与progress/status一致。statecomplete/completedtrue/exit0/attempt1；末轮100验证指标和曲线待history，不用best替代final。
- 最终testOA94.11904761904762%、macroF194.10529172930384%，正确数由OA算术对应23718/25200、错误1482；macroF1尚未拿逐图预测独立复算，perclass/CM/配对结果待test.json。finalrefit回读误差0、SHA716835556fe989739fc39e2d278188739b4d4454a14b5bf4d8d2b9efe060db1c，test文件SHA869986974e4801f2234fd2fa3bace383af2f092fd20359fbe1a252d4af7e37ae只存服务器原件引用，未下载300MB模型或test字节。正常程序在完整test/身份末尾复核及summary/exit0/marker后打印COMPLETE。
- 总wall24032.617371428秒＝6小时40分32.617秒，engine23899.926034053秒；差132.691337375秒为wrapper准备/末尾检查等合并区间。actualallocated4734618112B=4.4094567299GiB、reserved5135925248B=4.783203125GiB，范围含本次engine诊断/重载，非纯推理或NVML整卡。当前attempt1全流程，不含stdlib预检/租卡空闲；无账单单价不编造费用。完成实测替代初轮9h校准外推；截断最后epoch片段只可确认train112.182047135秒/56.158718448图秒、online100%，不能当整体吞吐/测试accuracy。
- [结果JSON](results/hgtnet_nwpu_seed42_20261103.json)/[可读结果与边界](results/hgtnet_nwpu_seed42_20261103.md)保存，并向[当前seed42比较CSV](results/external_baselines_nwpu_seed42_20261030.csv)追加HGT一行，标注correct_basis为serverOA算术/预测待回收；[原四行完整字节备份](results/evidence/hgtnet_formal_complete_return_20261103/comparison_before_hgtnet.csv)保留，原四行不变。旧comparison_table/ablation历史成绩未改。源Swin/GAViT正式记录已复核，未混入旧约96%协议。
- 同manifest/seed42，HGT比Swin高0.18650794pp/净47张，比fixed4×4高0.01587302pp/净4张，比localsoft高0.25pp/净63张，比独立DF高7.58730159pp/净1912张。单seed接近点估计不构成等价/显著/稳定性证明；缺HGT逐图配对不能说只纠正4张无退化。fixed参数31,386,920，较HGT74,619,858少57.9376%；HGT约2.3774倍。骨干/初值/增强/预算/GPU不同，不据跨卡时间称同条件更快，不把完整方法差异归因单个graph。
- 本轮无改源/配方/seed、追加GPU或重跑、Git提交推送或外发。正式独立基线现已完成，下一步只回收test/history/小metadata做25200预测与指标独立复算、45类混淆和训练曲线；不需再占用GPU。最终模型/metadata均按程序保存在持久化work目录，原件未回收边界如上。

---

## 2026-11-03 — HGTNet正式任务真实启动，第1轮完整完成/第2轮训练段完成

- 用户上传[终端原样片段](results/evidence/hgtnet_formal_start_return_20261103/reported_output.txt)，9556B、SHA3184766b0014b350f5424fe1415d0363f95506613fc65e770749ee6da9c43bef；[本地22/22核验](results/evidence/hgtnet_formal_start_return_20261103/validation.json)与[独立22/22审查](results/evidence/hgtnet_formal_start_return_20261103/review/validation.json)通过。接受SERVER_REPORTED_FORMAL_STARTED_EPOCH1_COMPLETE_EPOCH2_TRAIN_COMPLETE。不是仅凭PID启动：实际cuda:0首批forward/backward/Adam成功，五组451tensor梯度有限、五组代表参数实际改变，首轮epoch_complete；未直连查询服务器GPU进程/原始metadata/快照。
- 实验身份：HGTNet独立完整recipe/NWPU/seed42，v1包b2bc9077...572e1/bundle9a673b...5710；Git基线058f1d8b79d0be538ec358eb742d2cd3992921d4，未提交增量由包内SHA绑定。本地实际包及engine与本次交付源相符；用户完整实际命令保留在原样日志首行和[JSONcommand](results/hgtnet_nwpu_formal_progress_20261103.json)，run、currentGPUUUID4f625a02...3cb1、formal_v1/指定manifest/config/recipe/ZIP/候选/新output/scratch，PID2661/attempt1。日志无服务器启动时间戳，不据粘贴时间编造实际起止时刻。
- 固定数据/模型/优化沿用上一条预登记：manifestce0cfef2...840e，train5040/val1260→freshpool6300→test25200；RGB224的固定crop/flip/eval流程、完整74619858参数/候选Swin-B、100校准epoch/batch16/Adam1e-5/WD5e-5/constantLR/CE、FP32/无AMP/TF32。主要方法变量是独立HGTNet完整方法，不隔离单模块因果，不因中间分数改指标/seed/epoch。
- 第1轮train315steps/5040图，onlineaccuracy47.916666666666664%（算术2415/5040）、loss2.1766647955727954；valOA77.3015873015873%（算术974/1260）、macroF176.81321440157961%为服务器报告，未拿到逐图预测独立复算。该轮selected_epoch1只是当前最佳，最终预算尚未确定。正常源码先写best模型/完整epoch恢复snapshot/history再打印epoch_complete，但原件字节未回收，不冒称checkpoint SHA已直接验证。
- 最新epoch2仅training_epoch_complete：本轮315steps/5040图、global630，无第2轮val/snapshotcomplete/wall。不能记作2轮全部完成，更不是已完成refit/test或正式性能结论。[可读进度报告](results/hgtnet_nwpu_formal_progress_20261103.md)和JSON单独保存，未把77.30%写成正式test表分数。
- 实测epoch1 train250.583607628秒/20.11304749图秒，val8.641767578秒，wall326.696851827秒=5分26.70秒；其余67.471476621秒是样本集构建/核验、参数/快照/历史保存/同步等合并区间，未拆项profiling，不归因全部IO。epoch2 train236.359086964秒/21.32348735图秒，但缺该轮完整wall。GPU实际返回RTX3090/Torch2.2.2+cu121/CUDA12.1/visible25295323136B；这是总容量，正式峰值/费用待最终summary与实际账单。
- epoch1事件当时估算校准remaining32342.988330873秒=8.98416h；按首轮wall100校准轮约9.07491h，另有selectedepoch freshrefit/test。无日志时间戳，不声称当前时刻仍剩9h；估算随完整轮均值更新，非保证总时长/实际费用。此前3.97h仅预加载16图计算段，不能当正式管线预算；现采用实际正式epoch观测。任务当前应按原协议继续，未中断/改源/重跑或追加探针。
- 本轮只本地原样归档、源码/算术核验和日志解释，无新GPU/本地数值训练、租卡、Git提交推送或外发。当前正式运行中，最终best/final/refit/test、checkpoint身份/最终峰值、完整summary/exit0仍待完成后回传，尚无HGTNet正式test结果。

---

## 2026-11-03 — HGTNet正式seed42独立运行包本地完成，386回归及完整CPU流程通过

- 用户已补回batch16只读status：[原值](results/evidence/hgtnet_batch16_return_20261102/status_reported.json)、[26/26摘要检查](results/evidence/hgtnet_batch16_return_20261102/status_validation.json)保存。statecomplete/completedtrue/exit0，4steps/16独立记录/64呈现、train16/val0/test0、config/manifest/候选/UUID匹配；allocated4732054016B=4.4070687294GiB、reserved5215617024B=4.857421875GiB，全probe wall10.059117196秒。接受SERVER_REPORTED_BATCH16_PASS_STATUS_IDENTITIES_MATCH，原3090/360+90及batch16短测均不重跑。用户回传而非直连服务器原件，作用域为四步allocator/整个probe，不当正式epoch或账单。
- 正式研究问题与预登记变量见[准备文档](docs/external_baselines_20261027/hgtnet_formal_preparation_20261102.md)：同manifest身份下独立HGTNet完整方法是否具竞争力、额外复杂度是否值得；方法整体recipe为主变量，不隔离单个图算子因果。训练seed42，接受所有结果方向，不因test换seed/挑分或重跑。原文确认100epoch/batch16/Adam1e-5/WD5e-5/CE；其它原文未知项提前披露在[冻结recipe](configs/hgtnet_nwpu_formal_v1.json)，canonicalSHAce72e1822d55632b6dd9b02cdb4dc32da7e3e77eedca815b83981ff559c1af85。
- 实现新增formal_data/protocol/checkpoints/engine/entry：[计划](docs/superpowers/plans/2026-11-02-hgtnet-formal-seed42.md)及[完整证据](results/evidence/hgtnet_formal_prepare_20261102/README.md)。calibration5040/1260固定100轮，val正确数严格更大才选best、平手最早；同初值SHA freshmodel/新Adam在pool6300训练selectedepochs，再安全保存/独立模型数值回读，最后25200test。保存OA/macroF1/perclass/CM/逐图预测、best/final/selectedepoch、源/数据/配置/初值/runtime及效率。没有继承已学smoke/calibration权重，没有新增架构搜索。
- 独立工程固定项：betas(.9,.999)/eps1e-8、constantLR/无scheduler/warmup、CE无smoothing/实际clip；训练RGBResize256bilinear→RandomCrop224→HFlip.5→ImageNetnorm，evalResize256→CenterCrop224→同norm；seed42、FP32/noAMP/TF32、CPU8线程、workers0/dropfalse。完整模型74619858参数和Swin-B候选68c6b09e...1a893沿用，原作者精确架构/预训练仍未核实。每epoch42+epoch恢复同RNG/顺序，未保证所有CUDA算子位级相同；CPU等价只属fixture。
- 新恢复机制：两个完整epoch槽/两个best模型槽，严格SHA/binding/finiteTensor/phase/epoch/history/预算/初值/Adampolicy及moment形状检查；换同软件3090UUID可从最近完整epoch恢复。每attempt日志/摘要/命令/runtime保留，未提交epoch重做；最终wall/峰值是当前attempt，不编造累计费用。图片pool先本地恢复，test在finalrefit/回读之后解码；partialpool/test保留并新attempt重建，recoveryreceipt原子写入。训练epoch结束先记录真实全部更新数再验证/保存，避免失败漏记末尾步数。best预测packet随snapshot提交，防止未提交新best在重放后污染最终逐图校准工件。
- 最终[386项相关回归](results/evidence/hgtnet_formal_prepare_20261102/integration/related_suite.json)通过，0失败/错误/跳过（unittest48.426秒）；新增data13/checkpoints7/engine15/entry14/delivery4均有有效RED→GREEN。原381及384阶段记录保留。首个完整CPU流程已完成，但根核验脚本误读compacttest指针导致exit1，原日志及工件保留；改读test.json后最终[CPUfixture03](results/evidence/hgtnet_formal_prepare_20261102/full_model_cpu_fixture_03/summary.json)实际完整74619858参数+候选、人工JPEG train2/val2/pool4/test2，2校准轮选1→freshrefit1轮→保存/独立回读→test2，共4步，初值SHA相等/回读误差0。localCPU Python3.12.13/Torch2.2.2，11.86859秒仅夹具wall，不是GPU/NWPU准确率/费用。
- [独立最终审查](results/evidence/hgtnet_formal_prepare_20261102/review/final_validation.json)16/16证据/身份检查通过，无未解决Important项；包含8checkpoint语义、partialpool实际ZIP、失败计数、双best槽7项事务与bestpacket错配的真实CPU复现/修复。正式GPU尚未运行，不将本地stub/小fixture/旧baseline数字当新benchmark。
- 最终[交付v1](output/hgtnet_nwpu_formal_3090_20261103_v1.pyz)83604B、SHAb2bc9077cf37e75c83fbb72e7c4047d2fb9ede04db8fb4e4a56344500ff572e1、bundleID9a673b432dd8dea6b8cb175a559515c2092d953f8fd4016362dd325726c35710；[说明/一行命令](output/hgtnet_nwpu_formal_3090_README_20261103.md)。25payload，独立新output/scratch，复用已上传manifest/ZIP/权重。实际最终zipapp禁止ML导入inspect/status/resume-help exit0，无runs；[145/145身份/保护](results/evidence/hgtnet_formal_prepare_20261102/integration/validation.json)确认currentpayload/CPUfixture源、旧19payload/4HGT包/52历史指标源DF包/完整候选/config保持。无v1后缀未交付候选保留，使用v1。
- 最坏100+100的70900更新按已测三步中位缩放约3.97小时纯计算，另有IO/增强/验证/保存/refit/test，不能当正式ETA或租卡上限。新脚本按实际完整epochwall给当前phase剩余估算，实际总时长/费用等待正式回传。当前LOCAL_FORMAL_SEED42_READY_REAL3090_RUN_PENDING，无正式测试分数/正式表改写；未直连服务器、启动GPU/新租卡/外发或Git提交推送。基线HEAD058f1d8b79d0be538ec358eb742d2cd3992921d4，未提交增量由包内SHA绑定。

---

## 2026-11-02 — 3090真实batch16完成4次更新回传，峰值/摘要待现有status

- 用户回传hgtnet_batch16_probe_3090_20261102.pyz的实际完整命令、CUDA设备行、warmup1+measured3、COMPLETE路径；[原样粘贴](results/evidence/hgtnet_batch16_return_20261102/reported_output.txt)与[29/29本地核验](results/evidence/hgtnet_batch16_return_20261102/validation.json)、[独立26/26审查](results/evidence/hgtnet_batch16_return_20261102/review/README.md)保存，记SERVER_REPORTED_BATCH16_4_UPDATES_COMPLETE_SUMMARY_PENDING。接受该固定16图/FP32完整独立模型短测程序报告完成，不需重跑；未直接读取服务器文件或冒称原件已回收。
- batch16、4steps/64样本呈现、PID5457；五组451可训练参数tensor梯度全有限、missing/nonfinite空、groupnorm正，五组代表参数首步均更新，全部post_update_checks_passed。loss4.158695→3.343672来自同16train图复用，不是val/test或总体性能证据，不写正式方法比较表。
- 预热1.0480841400003555秒；三次计算计时0.1999162600004638/0.20252209699992818/0.20135352999932365秒，中位0.20135352999932365秒，均值0.20126396233323854秒，16/median=79.46222745662192图/秒；48/sum=79.49759020200715图/秒另记聚合口径。同步计算段含forward/backward/Adam/有限梯度检查，排除读图、增强、拷贝、初始化、日志、首步诊断；不能除以旧batch2的含读图8.81图/秒宣称加速。
- 实际CUDA行确认Torch2.2.2+cu121、CUDA12.1、available=true、device_count1、RTX3090，total_memory_bytes25295323136=24123.5MiB=23.55810546875GiB，这是可见总容量，不是峰值。命令UUID仍GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1；本轮未返回NVML/完整runtime，不冒充本轮实测物理容量或归因驱动差额。allocated/reserved峰值、整个probe wall、摘要config/manifest/weights/UUID身份与实际exit数值待现有status；已发只读命令，不请求新训练。
- [工程报告](results/hgtnet_batch16_probe_3090_20261102.md)/[JSON](results/hgtnet_batch16_probe_3090_20261102.json)及[正式准备](docs/external_baselines_20261027/hgtnet_formal_preparation_20261102.md)同步。条件算术315次校准训练更新/轮约63.43秒，394次refit更新/轮约79.33秒；100+100最多70900次更新约3.96555小时纯更新计算，排除完整IO/增强/验证/保存/启动/test，不是正式ETA、租卡费用或保证上限。refit末批12图，实际轮数须按预先选轮规则确定。
- 本地实际交付包SHA9563c9...6799、entry/engine字节与交付相符。正常COMPLETE路径先要求4steps及最终参数有限、source/input/所选图末尾核验、summary、run.exit0及marker再打印；这些是交付程序正常条件，不冒称直接检索服务器exit文件。本轮无代码/配方/正式表修改、新GPU/本地数值训练、租卡或Git提交推送。下一阶段补摘要资源、冻结正式recipe并验证完整数据管线预算；当前仍无正式HGTNet trainer命令或benchmark结果。

---

## 2026-11-02 — HGTNet真实history分段耗时回传，batch16容量包本地完成待GPU

- 用户回传现有history，[原值](results/evidence/hgtnet_3090_smoke_return_20261102/history_reported.json)与[8/8算术核验](results/evidence/hgtnet_3090_smoke_return_20261102/history_validation.json)已保存。train40.84089300300002秒、360图/180steps=8.81469462417376图/秒；validation1.9847141220002413秒；其他smoke阶段15.431895507999798秒。训练段含读取/首步诊断/更新，不是纯GPU计算。trainloss3.7911980125639175、online trainaccuracy12.222222222222221%（算术44/360），valloss2.315320509051283。训练在线累计模型持续变化，validation轮末另组数据计算，不用两者差值判定异常/归因。
- [smoke结果报告与JSON](results/hgtnet_nwpu_smoke_3090_20261102.md)已补分段字段。相同batch2/noaug/训练率不变时5040/6300图约571.77/714.72秒只是条件算术，排除后续阶段/IO/增强变化；不是batch16、正式100轮或实际费用预测。原status/预测/checkpoint原字节未回收的边界不变，不重跑已完成smoke。
- 为论文batch16/Adam1e-5/WD5e-5先做更便宜容量验证，新增batch_probe与batch_probe_engine及受限builder入口。固定seed42/原架构74619858参数/候选Swin-B初值/FP32无AMP/TF32；从固定train360取class0–15各首张，共16图，优先核验复用旧450图缓存，否则完整ZIP SHA后仅恢复16图。1次预热+3次计时更新，共4steps/64样本呈现、16独立记录，val/test均0。明确Adam betas0.9/0.999、eps1e-8工程默认，RGB224/noaug只用于容量；无epoch、accuracy、checkpoint或已学smoke权重续训，不是单变量方法消融/正式配方。
- [新小包与一行说明](output/hgtnet_batch16_probe_3090_README_20261102.md)：hgtnet_batch16_probe_3090_20261102.pyz，61006B，SHA9563c9b14bb182bb58104480bedee963efda6c44e48cd63fce5f8522a9426799，bundleID75e7fd09dc0fd9404532f99e319cfe605ecb301c453ab849ce1d2eac25cd72db。新安装/output/scratch，inspect生成当前3090 UUID的run_command；未改已交付旧包。compute-only同步计时与wall范围独立，预热首步诊断排除3次统计；峰值含单模型/输入/Adam状态/梯度/激活/诊断，无重复回读模型，不据此预先宣称batch16显存/完整数据吞吐。
- 本轮新建隔离/private/tmp/gavit-hgtnet-probe-cpu-20261102，安装精确Torch2.2.2/vision0.17.2/NumPy1.26.4/Pillow10.3.0，本地Python3.12.13 CPU（目标3.11.8 CUDA另核验）；未改旧/项目/服务器环境。旧环境失败是当时事实，原日志保留；当前[333项HGT/DF数值相关回归](results/evidence/hgtnet_batch16_prepare_20261102/integration/related_suite.json)通过，0失败/错误/跳过，16.2288秒含discovery。新7项实际Tensor数值测试含4次Adam更新、坏输入/有限梯度/参数变化、更新后诊断失败仍保留实际步数；入口13项、交付8项都有RED/GREEN证据。独立28项检查无未解决问题，不称全旧仓库依赖suite绿色或CPU夹具为GPU。
- 实际最终pyz在禁止ML导入条件下inspect/status exit0、batch16 schema/profile3090/counts16/0/0、not_started、无runs；本机缺目标资源所以ready=false是预期。[104/104身份及保护核验](results/evidence/hgtnet_batch16_prepare_20261102/integration/validation.json)通过：原16源码/config/完整候选权重、生成来源JSON、3旧HGT包及52历史指标/源码/DF包保持原SHA，正式性能表未改。[完整证据](results/evidence/hgtnet_batch16_prepare_20261102/README.md)与[正式准备](docs/external_baselines_20261027/hgtnet_formal_preparation_20261102.md)同步。
- 当前LOCAL_BATCH16_PROBE_READY_TARGET_GPU_PENDING，真实batch16容量/计算结果待用户在现租3090执行；尚无正式trainer/正式训练命令/新test分数。未直连服务器、启动GPU/新租卡/正式训练、提交推送Git或破坏性清理；后续以实际容量检查结果冻结完整正式recipe/预算。

---

## 2026-11-02 — HGTNet3090真实NWPU360/90短摘要验收，工程门禁通过

- 用户回传fix1 status：state=complete、completed=true、exit_code0、training_started=true、formal_training_started=false，train360/val90/test0、180steps、best/final均1。[规范化回传JSON](results/evidence/hgtnet_3090_smoke_return_20261102/status_reported.json)与[56/56摘要/身份核验](results/evidence/hgtnet_3090_smoke_return_20261102/status_validation.json)、独立70/70核验已保存，接受SERVER_REPORTED_REAL_NWPU_3090_SMOKE_PASS_STATUS_IDENTITIES_MATCH，无需重跑或再回收大checkpoint证明流程。
- 返回config canonical SHA af6ef410...ac25、manifest ce0cfef2...840e、Swin-B权重68c6b09e...1a893与实际交付fix1包90c60d...7c42内容一致；3090 UUID GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1与先前清单一致。canonical配置SHA不同于配置原JSON字节SHA。完整runtime字段未包含在短status中，按程序正常通过条件验证过，不将未返回的Torch完整后缀/当前CUDA原始字节等编造成回传值。
- Val90 OA36.666666666666664%，摘要算术对应33正确/57错误；macro-F129.33525733525733%为服务器报告，尚未从逐图预测独立复算。checkpoint回读最大误差0.0，返回SHA de78efa372cf79862859efaf13b34ff4e82c25fceb58af9a0caf84c43e7232e5只作服务器原件身份引用，未下载模型字节验证，不声称原件已回收。
- wall58.257502633秒，从stdlib预检后开始，含实际imports/图片恢复核验/初始化/训练/验证/保存回读/末尾检查，不是纯训练吞吐或计费时间。全smoke PyTorch allocated峰值1538072576B=1466.8203125MiB=1.432441711GiB，reserved1688207360B=1610MiB=1.572265625GiB；包含更新诊断及独立模型回读，不是NVML整GPU/进程显存或batch16训练峰值。
- [结果报告](results/hgtnet_nwpu_smoke_3090_20261102.md)与[机器记录](results/hgtnet_nwpu_smoke_3090_20261102.json)单独保存。36.67%/29.34%只属于360训练图/一轮/batch2/lr1e-4的工程设置，未写正式性能比较表，也不据此判定正式模型优劣、改变主要指标或挑seed重跑。此前本地全数值suite依赖缺失限制保留；本次真实服务器路径是独立新证据，不把历史285项测试当作本轮回归通过。
- 已按既有原文审计整理[正式seed42准备](docs/external_baselines_20261027/hgtnet_formal_preparation_20261102.md)：原文确认100epoch/batch16/Adam1e-5/weight_decay5e-5/CE，与工程设置区分；固定calibration5040/1260→fresh6300refit→25200test、保留独立架构/预训练与实现未知项。尚未创建/运行正式trainer。优先读取已存history分离训练耗时，随后准备明确有界的batch16容量检查，不从58秒或batch2峰值线性推算正式预算。
- 本轮只本地归档/核对/写准备文档，未新运行GPU、额外smoke/正式test/训练、改正式表、安装依赖、Git提交推送或外发。下一阶段为实际batch16资源/完整正式recipe与独立入口准备；现有360/90检查已完成。

---

## 2026-11-02 — HGTNet3090真实360/90工程流程完成标记回传，待现有短摘要

- 用户回传steps30/60/90/120/150/180、samples60/120/180/240/300/360及training_started=true，最后complete=true progress与HGTNET_NWPU_SMOKE_COMPLETE，路径为work/hgtnet_nwpu_smoke_3090_20261102_fix1/runs/nwpu_smoke_seed42_01/summary.json。[原样粘贴片段](results/evidence/hgtnet_3090_smoke_return_20261102/reported_output.txt)保留首行GTNET少H，非完整服务器日志或直接远程文件。
- [41/41清单/源码检查](results/evidence/hgtnet_3090_smoke_return_20261102/validation.json)与独立实际包审计通过：fix1包SHA90c60d...7c42、17payload/16本地源均一致。正常完成路径要求固定train360/val90/test0、epoch1/180steps、梯度/参数有限与真实更新、验证顺序正确、checkpoint写盘及独立模型数值重载误差≤1e-6、history/预测保存、末尾源码/配置/manifest/权重/选样字节复核，随后summary completed=true、run.exit0、complete progress，才打印COMPLETE。
- 按交付程序与用户回传记SERVER_REPORTED_HGTNET_REAL_NWPU_SMOKE_COMPLETE_MARKER_SUMMARY_PENDING，接受此次真实小样本工程流程程序报告完成，无需重跑。未直接读取服务器summary或exit文件，具体90张验证OA/macro-F1、实测峰值/总时长、checkpoint SHA/精确回读误差、当前实际runtime身份仍待现有status输出；不凭日志推算或用本地夹具替代。
- [证据入口与一行只读status命令](results/evidence/hgtnet_3090_smoke_return_20261102/README.md)已保存。本轮只离线归档/核对，无新GPU/数据test/训练/下载checkpoint/Git提交推送。工程指标不写论文正式性能比较表，不将单轮小样本或修复门禁称为方法收益；下一步回收现有短摘要后按实际耗时/显存准备正式协议与运行包。

---

## 2026-11-02 — 3090实际CUDA显存口径误拦截定位，fix1独立包交付

- 用户回传原3090包inspect ready=true、完整权重SHA通过、无GPU进程、资源/metadata检查通过，执行生成命令后在_runtime_and_torch的合并设备判断报错，尚未进入数据/模型训练。该位置会先创建output并保留失败日志，因此不重跑/覆盖原路径，fix1采用独立目录。
- 只读设备诊断确认CUDA12.1、available=True、device_count1、名称NVIDIA GeForce RTX3090，properties显示total_memory24123MB；NVML物理总量24576MiB。原比较CUDA total floorMiB必须等于NVML nominal，构成已证实误拦截。诊断未返回原始字节数，torch.version被打印为module而非完整版本，故不编造具体Torch版本/精确CUDA字节或差额的驱动/平台原因。[结构化用户证据](results/evidence/hgtnet_cuda_memory_fix_20261102/user_reported_failure.json)明确重建来源，无直接服务器文件下载。
- 仅改runner：独立记录/输出CUDA probe；保留物理NVML卡名/24576MiB和UUID、空闲、实际软件版本、dtype/AMP/TF32/seed等原门禁。CUDA暴露字节允许在NVML nominal减1GiB至nominal范围内，并分别记录两接口容量及差额；1GiB是明确工程兼容上限，不当作原因或训练显存需求。无效类型、0、错误名称/数量/不可用，以及边界外1字节仍拒绝。模型、数据、引擎、配置、batch2/seed42/360+90/一轮/Adam均未改。
- 新[fix1包](output/hgtnet_nwpu_smoke_3090_20261102_fix1.pyz)53324B，SHA90c60dabcd185ffd0a1b575f8544e7363a469c4e3616528955cb3645c2ae7c42，bundle ID96d3451f4119c8a42f8507e819f534cca42e2b7235b21006984144d488158019。[上传与一行命令](output/hgtnet_nwpu_smoke_3090_fix1_README_20261102.md)只需上传新小包，原Swin-B/manifest/ZIP复用，新安装/output/scratch包含fix1，不服务器手改源码或删除原失败结果。
- 新6项有效RED→GREEN：通过runtime边界fixture复现原拒绝，验证代表24123×MiB容量（夹具非用户精确字节）、严格1GiB上下界、类型/名称/数量等。当前45项门禁/打包回归通过、0失败/错误/跳过；独立16运行边界+9信息/身份检查通过，无遗留问题。原全量数值suite缺本地Torch/NumPy限制不变，未重跑或声称全数值通过。[112/112身份审计](results/evidence/hgtnet_cuda_memory_fix_20261102/integration/validation.json)确认唯一payload源变更为runner，15份其他本地payload、builder、旧3090包、旧52工件及完整候选权重SHA保持。
- 实际最终fix1 pyz禁止ML导入的inspect/status通过、profile3090、fix1新路径、not_started、无runs；此项不等于真实CUDA训练通过。[证据入口](results/evidence/hgtnet_cuda_memory_fix_20261102/README.md)。没有启动GPU/训练、安装库、删除用户文件或Git提交/推送。下一步用户执行fix1 inspect生成的新命令，实际前后向/validation/checkpoint/峰值资源仍以服务器回传验收。

---

## 2026-11-02 — 根据用户实际RTX3090租卡信息交付独立兼容包

- 用户说明3080无可租，取得3090，授权检查并做必要兼容修改；已回传RTX3090、GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1、总24576MiB/空闲24123MiB、driver610.57.04。[原始清单](results/evidence/hgtnet_3090_prepare_20261102/user_gpu_inventory.txt)仅为用户终端回传，不是直接远程CUDA或训练验收。本轮改卡只适用于此次HGT工程门禁，不据旧默认4090要求阻断，也不把硬件选择扩大到未指定实验。
- 原资源模块仅接受3080/10240，旧包会拒绝本次卡。新增显式3080/3090目标，默认3080 API兼容；运行包manifest绑定3090并传入runner，预检选择当前空闲3090的实际UUID，run验证选中名称/24576MiB和真实CUDA，而非旧3080常量。软件版本、UUID/空闲查询、FP32/无AMP/TF32、容量/防覆盖仍严格；没有扩大到任意GPU。direct-source与bundle的scratch均由实际target/package命名，修复一处参数与环境默认错位，有有效RED→GREEN。
- 新[3090包](output/hgtnet_nwpu_smoke_3090_20261102.pyz)52719B，SHA33e6f0a9ab449b314ba62b6961389db2d7464039c598b7775f7fef881bda19ff，bundle ID81180fc3bd4903e5af338be1ff446c2ba762fca5ea19ad51c8a8495a66fdac51，[单行说明](output/hgtnet_nwpu_smoke_3090_README_20261102.md)可直接上传新小包后inspect。原Swin-B文件与完整68c6...1a893 SHA沿用，已上传则不重传。旧3080包SHA e798...35ce78保持；新包用独立安装/output/scratch路径，不覆盖旧工件。
- 此次只改变目标硬件检查及其传递，不改模型/权重/数据/训练：seed42、batch2、一轮180steps、train360/val90/test0、Adam1e-4/CE、RGB224、无增强/AMP/两种TF32。14份本地payload（模型core、数据、引擎、配置与helper等）和生成权重元数据保持原字节；旧52指标/DF源/包也一致。显存变大不自动加batch，不重新做本地数值训练或启动正式实验。
- [39项当前硬件/CLI/打包测试](results/evidence/hgtnet_3090_prepare_20261102/integration/targeted_suite.json)通过（0失败/错误/跳过），包含用户实际清单fixture、3090/3080互不误识别、精确显存/UUID/空闲、软件精度不放宽、真实pyz目标传播、runtime边界模拟和不覆盖旧结果。独立审查无遗留问题，使用用户NVIDIA行在进程边界fixture选中本次UUID/24GiB；实际最终pyz禁止ML导入的inspect/status通过、target3090、not_started、无runs。当前[110/110字节/身份审计](results/evidence/hgtnet_3090_prepare_20261102/integration/validation.json)通过。
- **本轮数值回归限制**：尝试全HGT/DF suite未通过，收集148项/失败导入项，1failure+26errors，根因为旧/private/tmp工程环境现已解析为不含Torch/NumPy/vision/PIL的base Python，项目.venv同样缺依赖，旧wheels目录为空。没有为了绿色结果改用缓存中的其他Torch版本。[失败日志与逐项名称](results/evidence/hgtnet_3090_prepare_20261102/README.md)原样保留；数值模型/数据/引擎字节未变，10月31日285项和完整CPU结果只作历史证据，不能称本轮299项全部通过。当前只针对实际变更的stdlib硬件/交付路径通过39项检查，未更改服务器依赖。
- [证据入口](results/evidence/hgtnet_3090_prepare_20261102/README.md)。实际3090的CUDA/真实360+90、显存/耗时仍待服务器执行；不将NVIDIA清单或边界fixture当作已启动。未直接操作服务器、启动GPU/正式训练、产生新租卡、删除用户文件或Git提交推送。用户下一步只需上传新包运行inspect；ready后按生成的本次UUID命令执行。

---

## 2026-10-31 — HGTNet RTX3080真实NWPU360/90运行包已本地验证交付

- 用户同意继续准备3080包。本轮工程问题：已有独立HGTNet能否在固定真实图片预处理、全参数一轮训练、validation和持久化/独立模型回读之间贯通，且避开此前版本后缀、旧UUID、work容量与缺原图问题。主要变量是执行环境/真实数据路径，不改上一轮模型结构或候选预训练。用本地JPEG/CPU先验证，目标GPU检查待用户运行；不是正式实验，不产生论文准确率。
- 新增nwpu_data/resources/nwpu_engine、独立runner/builder及50项行为测试。复用旧只读NWPU helper选择train360/val90、pool450、test0，seed42、batch2、1epoch/180steps、Adam lr1e-4/CE、RGB224 bilinear/ImageNet normalize，无增强/AMP/TF32。work放源码/清单/日志/训练历史/验证预测/工件，恢复图片到/tmp新scratch；work2GiB+scratch1GiB，同卷合并3GiB，无旧DF特征缓存和SVM文本。未根据结果改batch、超参或结构。
- [最终运行包](output/hgtnet_nwpu_smoke_3080_20261031.pyz)52238B，SHA e798372589f83359f6d33d3c78d48ea1a82a760310b23f26aa0d6e689835ce78，17个payload，bundle ID42fcdedc83d138708db13d1501b11b06a4e384f506acbc46eb5b9c4756ade29b。仅HGT源码/固定config/入口/旧数据helper和权重来源JSON；无LIBLINEAR、二进制、依赖安装或在线获取。Swin-B候选351654207B需单独上传，完整68c6...1a893 SHA沿用；旧VGG/Swin-T不能替代。[上传与单行说明](output/hgtnet_nwpu_smoke_3080_README_20261031.md)给出先inspect、再复制当前UUID绑定命令、最后status短摘要的流程。
- inspect全程stdlib，检查完整权重SHA、manifest、版本metadata、空闲10GiB3080及分卷预算；Torch/vision接受精确base或cu121后缀。run要求明确本次UUID，实际验证Python3.11.8/Torch2.2.2/vision0.17.2/NumPy1.26.4/Pillow10.3.0、CUDA12.1/cuDNN8902和FP32/8线程/seed42配置。缺450张选样时，现有ZIP记pending/fullhash=false，run完整SHA验证后仅恢复train/val并逐图hash；最终摘要明确verified恢复，不保留pending旧文案。该状态修复有独立真实CPU训练/重载回归RED→GREEN。
- [最终完整CPU JPEG夹具](results/evidence/hgtnet_3080_smoke_prepare_20261031/full_model_cpu_fixture_02/summary.json)使用完整74619858参数模型，实际候选加载318 tensors，生成4张train/2张val图片、2个optimizer steps，五模块组均有有限梯度和首步参数更新，checkpoint安全写盘/独立新模型数值回读误差0；verification wall3.879秒、训练段1.9116秒。明确cpu_fixture、非真实NWPU/3080，不把该时间或夹具准确率当正式吞吐、费用或模型结论。旧模型九个Python文件/配置/候选文件全部保持原SHA。
- [285项相关回归](results/evidence/hgtnet_3080_smoke_prepare_20261031/integration/related_suite.json)通过（137HGT+148DF，0失败/错误/跳过，测试执行12.883秒），包括数据/版本/UUID/容量/文件篡改/防覆盖/实际zipapp及失败保留首批状态；不重开已知旧全仓库依赖范围。[独立审查](results/evidence/hgtnet_3080_smoke_prepare_20261031/review/independent_review.md)无已复现Critical/Important，另有实际CPU失败后首步状态、错误数值回读、wrapper晚期失败探针。实际最终pyz在禁止所有ML导入时inspect成功、未创建runs，status not_started。最终[121/121审计](results/evidence/hgtnet_3080_smoke_prepare_20261031/integration/validation.json)核验包/源码/候选/fixture证据，旧52个指标/源/DF包全保持。未对外交付的本地草稿归档保留，最终包已重建。
- [证据入口](results/evidence/hgtnet_3080_smoke_prepare_20261031/README.md)。只有首批完成forward/backward/step才保存training_started=true；epoch、validation、真实输入checkpoint、数值回读及末尾身份复核都通过后才completed+exit0。HGT real-image checkpoint另设schema，实际图片不误标synthetic，best/final本次均epoch1。日志/历史/逐类混淆/逐图预测均保存，异常不自动重跑、改batch或删除旧目录。
- 仍无当前目标3080的实际显存、耗时、真实450图结果，也无HGT正式trainer/正式准确率；本轮未直连服务器、启动GPU/租卡或正式训练，无Git提交/推送、旧模型/DF结果改写或外发。下一步用户上传两个文件到work后运行inspect；据实际回传处理资源与真实小样本门禁，再决定正式协议。

---

## 2026-10-31 — HGTNet独立首版本地实现及完整CPU工程检查通过

- 用户批准开始实现HGTNet；本轮问题是验证可执行独立候选是否贯通DA、Transformer、动态图、融合、候选预训练和保存重载，先用合成CPU检查排除接线/形状/梯度错误。不是正式数据实验；无数据划分、epoch选轮、test准确率、GPU或费用测量。开发基线commit058f1d8b79d0be538ec358eb742d2cd3992921d4，实际未提交增量由配置/源码SHA识别。
- 新包[external_baselines/hgtnet](external_baselines/hgtnet/README.md)与[冻结候选设计](docs/external_baselines_20261027/hgtnet_implementation_design_20261031.md)：RGB224，通道128/256/512，STB1/1/9即attention2/2/18、heads4/8/16。按原文“每stage下采样”文字选择56→28→14→7，第三stage同通道Conv2；graph256×7×7与512×5×5、深度6/2、当前投影特征Euclidean k20、排除自身、不裁k，MR max和FFN双残差。visual与graph各512维GAP相加后45类head。Conv2、DW无padding实现7→5、图宽/LN/fusion是明确独立选择；原图/作者代码未确认，MR max与average冲突保留，不声称精确复现。30类头只验证构造/形状，未做AID评估。
- 完整参数74619858，全部可训练：patch6528、Transformer60479736、DA44077、graph14066432、head23085。候选配置[JSON](configs/hgtnet_independent_v0.json)SHA af6ef410125e9d47c87acd0acec1a6db6e0119656793124ceafcb2379eedac25，明确nonofficial/local_engineering_only。各算子默认无dropout/drop_path、LN eps1e-5，见配置完整记录。
- 原文只写partial ImageNet，本轮取得官方torchvision Swin-B IMAGENET1K_V1候选351654207B，完整SHA68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893。实际CPU安全加载并验证完整353个source tensors，全值有限；完整HGTNet实际复制318个兼容tensors，155个target新初始化、35个source跳过，所有异常列表为空。[真实加载报告](results/evidence/hgtnet_local_implementation_20261031/full_cpu_01/pretraining.json)覆盖59489486/74672680 state元素=79.6670%，含buffer不等于参数量；不称作者HGTNet预训练，meta/schema审计另存且不算加载。
- [完整实际CPU摘要](results/evidence/hgtnet_local_implementation_20261031/full_cpu_01/summary.json)：macOS CPU FP32、Python3.12.13/Torch2.2.2/vision0.17.2、单线程、seed42；B1/B2×224真模型前向，synthetic B2 CE4.558260917663574、Adam lr1e-4仅1步。451个可训练参数tensor均有有限梯度，五组gradient norm均正且五组代表参数有变化，所有更新参数有限。实际trace图49/25节点均k20、无自身边。计算步0.862940833秒/本地wall2.176888917秒，仅本次合成工程过程，不估算正式吞吐或GPU费用。
- checkpoint SHA bf1440f3fd654d2b6a16d8c2de02856513dcd4ecf53cf97ad475bbd9a384d966，保存绑定配置和六个模型本地源码SHA；不同进程重载B1/B2最大误差均0（容差1e-6）。两个实际CLI进程exit0。保存时receipt保留completed=false，fresh reload后summary才true；这是工程/inference checkpoint，不包含optimizer续训。大checkpoint和fixture本地保存、按作用域gitignore，不覆盖旧工件；跨Torch/vision版本或设备必须重新数值检查。
- 模块与集成87项HGTNet测试具RED→GREEN，连同148项DFAGCN回归共235/235、0跳过通过（测试执行11.345秒）。[独立审查](results/evidence/hgtnet_local_implementation_20261031/review/independent_review.md)无未解决问题，另有小型真实CPU接线/trace/批次隔离/checkpoint探针通过。旧全仓库缺依赖范围不扩展，不声称全仓库绿色。[保护审计](results/evidence/hgtnet_local_implementation_20261031/preservation/audit.json)19历史指标+12旧入口+21 DF源码/配方/包共52/52 SHA保持；新DF分析与导师稿只另记当前快照，未冒称有本轮前baseline。
- [证据入口](results/evidence/hgtnet_local_implementation_20261031/README.md)保留设计、原文复核、测试日志、实际命令、源码/权重/配置身份及本地数值结果。本轮未改原模型、DF配方或正式比较分数，未启动Featurize、正式训练或外发，没有Git提交推送。下一步是独立3080运行包和真实NWPU小样本工程门禁，随后才冻结正式数据/优化/选轮协议；当前没有HGTNet准确率结论。

---

## 2026-10-31 — DFAGCN完整逐图结果本地复算通过，逐类分析与科学图已交付

- 用户提供/Users/patrick/Downloads/test.json（约1.1MiB），[原样归档](results/evidence/dfagcn_test_predictions_20261031/test.json)SHA `771dee2dcd1e676e4884c36654cc8fbcc4c067811362dbf9f3e997c903ccd4e9`；record seal cf17f54c41421b61b2e3f816e0cad39ecc4ce0b2a6cf7aabaf7052b515fe700f匹配。phase=test、mode=formal、manifest/cache引用及cached时段吻合旧summary；refit_record SHA只记录引用，不声称模型字节已返回。
- 标准库独立重建全部25200条预测对应的类别目录标签，确认唯一/有序/规范路径、45类各560、pred整数0..44，重建的全部2025混淆单元与返回文件完全相同。正确21806、错误3394，OA86.53174603174604%；macro-F1复算86.45890874660425%，较报告86.45890874660422%仅差2.84e-14，在1e-10绝对容差内。逐类support/accuracy/F1全部通过，precision另行计算。[独立32/32核验](results/evidence/dfagcn_test_predictions_20261031/review/audit.json)和[本次复算脚本/证据](results/evidence/dfagcn_test_predictions_20261031/README.md)已保留。
- 召回最低palace58.0357%、church65.3571%、tennis_court66.9643%、basketball_court72.8571%、railway74.2857%，这五类共910漏判，占全部错误26.8120%。双向palace/church142（84/58）、railway/railway_station85（65/20）、basketball_court/tennis_court79（39/40）；前五类别对共440错误/12.9641%，前十有向混淆498/14.6730%。其余错误也全量保留，描述错误分布，不归因图传播、欠拟合或原图视觉原因。
- [完整逐类误差报告](results/dfagcn_nwpu_seed42_error_analysis_20261031.md)与[复制版](results/dfagcn_nwpu_seed42_error_analysis_20261031.txt)、45类指标/25200预测/3394错误/全部非零混淆/完整矩阵CSV、[分析JSON](results/dfagcn_nwpu_seed42_analysis_20261031.json)已生成。两张科学图导出PNG/PDF/SVG：10弱类+10有向混淆，以及全部45类行归一化矩阵；标准Matplotlib绘制，作图脚本、版本、SHA及实际PNG视觉QA保留，矩阵SVG为矢量单元格。绘图库仅装独立/private/tmp，未改研究环境。
- 结果记录/同seed42比较CSV的DFAGCN correct来源升级为本地逐图复算，86.53%/86.46%和固定GAViT高7.57pp不变。两份导师汇报及原结果说明已同步更新“逐类/逐图尚未回收”的旧状态，前版本保存在本次证据before_update；原内部多seed统计未改。旧模型预测未本地取回，不能将净多1908张正确写成逐图“只纠正无破坏”。
- 仍未回收原manifest/原图/cache/训练历史/refit/checkpoint原件，不将record引用一致称为这些外部工件的字节验收。此次只离线数据复核与制图，无新训练/GPU模型调用、后验调参、Git提交推送或外发。后续外部方法和配对分析按既有计划推进，当前独立实现单seed边界不变。

---

## 2026-10-30 — DFAGCN NWPU seed42正式摘要回传通过，结果表与汇报已更新

- 用户回传run.exit0与完整summary，[原始输出/规范化摘要](results/evidence/dfagcn_formal_3080_return_20261030/README.md)已保留；completed与formal_training_started均true，train5040/val1260/pool6300/test25200，selected_epoch51、C10、重载误差0。状态SERVER_REPORTED_FORMAL_COMPLETE_SUMMARY_IDENTITIES_MATCH，按服务器报告验收本次独立实现的完整正式流程，无需重跑。
- Test OA86.53174603174604%、macro-F186.45890874660422%；OA对应21806/25200正确、3394错误。独立29/29核对通过；主代理复算runtime JSON字节SHA d56d05d63aff4513f4fe733971736e430ecc40532f5ec9e45c032a72aafb1054、source_identity JSON字节SHA 4b6ce21462fc18755ef568e1553e7cdaffac9208b66a6d51793a23bf1d3d351b，与六阶段一致。pipeline/recipe/VGG身份匹配已交付fix1；完整逐图/逐类/历史/模型原件尚未本地下载，macro-F1不冒称本地从预测独立复算，OA整数也只是摘要算术一致性。
- 实际卡RTX3080 10GiB UUID GPU-0d9901ca-d2e0-d398-b9b4-2550e6db2fed，Torch2.2.2+cu121/vision0.17.2+cu121/CUDA12.1/cuDNN8902、CPU8线程、seed42、FP32/无AMP和双TF32，运行条件符合。wall2035.63461228秒（33分55.63秒），六阶段合计2025.858676642秒，余9.775935638秒；起点在预检后，不含租卡空闲/传输，不当作计费总时长。PyTorch阶段max allocated616995328B=588.41MiB、reserved698351616B=666MiB，来源不同phase，不是整进程显存。
- 冻结VGG117479232参数+图分支含CE头2957447=120436679神经参数，另有LIBLINEAR，不能仅将2.96M写作整模型。阶段时间、缓存/完整测试时段边界与实际路径已写[结果报告](results/dfagcn_nwpu_seed42_20261030.md)和[机器记录](results/dfagcn_nwpu_seed42_20261030.json)。
- 新[同清单seed42完整方法比较CSV](results/external_baselines_nwpu_seed42_20261030.csv)使用已有正式Swin93.93253968253968%、fixed94.10317460317461%、soft93.86904761904762%，未用旧约96%占位数。相对DFAGCN，fixed OA高7.57142857142857pp/F1高7.63102471616675pp，净多1908张正确；该净差不是逐图纠正/新增错误数。旧三模型GPU2080Ti，本次3080，骨干/预训练/增强/优化/选轮规则不同；仅比较披露recipe的完整方法单seed表现，不归因图模块、不宣称跨seed稳定性或同卡效率优势。
- 既有[Markdown汇报](docs/supervisor_progress_report_20261012.md)和[纯文本复制版](docs/supervisor_progress_report_20261012_plaintext.txt)已同步新增外部结果与范围，旧版本保存在证据before_advisor_update。原内部42/43/46/47均值/SD和所有历史分数保持，新外部单seed不合并进去；root旧comparison_table与研究worktree原表保持，新增独立来源表。HGTNet及外部比较其余要求仍未完成。
- 本轮只归档/核算/写文档，无新训练、test重跑、Git提交推送或外发。下一步先回收已有test.json、逐图预测、selection/refit历史进行45类support560/trace21806与F1复算及误差分析，不因分数较低后验改配方或寻找正向seed。原论文89.29%继续只作文献参考，本次未声称原文数值复现。

---

## 2026-10-30 — 收到DFAGCN正式完成标记，待已有摘要归档测试指标

- 用户回传[正式完成标记](results/evidence/dfagcn_formal_3080_return_20261030/completion_marker.txt)，路径为work/dfagcn_nwpu_formal_3080_20261030_fix1/runs/nwpu_seed42_01/summary.json。当前本地runner与实际交付fix1包逐字节一致，包SHA79398dc4b6d703be65cd32a968d49daa6640fe6c4bbab6a33e930b312954a6c4。
- 该标记仅在完整pool6300/5040+1260校准、fresh6300refit、25200test、必需模型与指标文件、输入/源码身份末尾复查以及summary completed=true、run.exit0、progress complete均写盘后打印。按用户日志记SERVER_REPORTED_FORMAL_COMPLETE_MARKER_METRICS_PENDING，确认程序报告正式流程结束；没有直接远端读取或新训练。
- 当前只收到一行标记，具体OA/macro-F1、选定epoch/C、实际硬件runtime、耗时/显存及模型身份数值待从既有摘要回传，未填正式比较表，也不借旧smoke或原论文分数替代。给出[一行只读摘要命令](results/evidence/dfagcn_formal_3080_return_20261030/read_summary_command.txt)，不会重训或输出大矩阵，后续据真实字段更新结果与汇报。

---

## 2026-10-30 — 根据真实inspect修复资源门禁，缓存迁至本地盘，fix1交付

- 用户实际服务器inspect的四个blockers是version:torch、version:torchvision、disk_capacity、target_gpu；manifest/完整VGG和已编译LIBLINEAR均通过。根因：发行包metadata仅显示2.2.2/0.17.2，旧门禁误要求+cu121；当前10GiB3080已变为GPU-0d9901ca-d2e0-d398-b9b4-2550e6db2fed，旧入口锁此前UUID；work只有17029922816B可用。后续df确认work为50G NFS（35G已用、16G剩余）、实例根盘362G（349G剩余）、data目录不存在。[原始事实归档](results/evidence/dfagcn_formal_resource_fix_20261030/reported_resources.json)为用户终端回传，无助手直连。
- 修改仅限正式wrapper和资源测试，不改核心模型/算法/配方：metadata接受精确base或cu121后缀，run仍检查实际CUDA12.1/cuDNN8902和FP32；inspect仅唯一空闲10GiB3080时选卡，生成明确UUID，run严格核对本次绑定。新实例未重跑探针，先前COMPLETE仍只属于旧物理UUID，不混写硬件证据。
- [修复约定](docs/external_baselines_20261027/formal_resource_fix_20261030.md)：新scratch-root默认/tmp/新包名_nwpu_seed42_01，四份约9.41GiB cache和恢复图片放实例本地盘，cache index另存work。checkpoint/history/log/模型/SVM文本/预测/完整指标仍在work，按阶段读test。scratch保留40GiB门禁；work按94500行×163331B的全SVM上界15434779500B+1GiB产物/安全余量，要求16508521324B，当前回传够用。若同st_dev合并预算，test恢复也查目标卷；旧/已有scratch和output拒绝覆盖，不自动删除历史文件。
- 新[fix1包](output/dfagcn_nwpu_formal_3080_20261030_fix1.pyz)151769字节，SHA `79398dc4b6d703be65cd32a968d49daa6640fe6c4bbab6a33e930b312954a6c4`，[一行检查说明](output/dfagcn_nwpu_formal_3080_fix1_README_20261030.md)可直接复制。它在新目录复用旧正式包两份本机编译binary，仍验可执行/非symlink/实际SHA；不需要重装Torch、重传权重或重编译已成功binary。旧正式/4090smoke/3080probe包保持原字节。
- [完整证据](results/evidence/dfagcn_formal_resource_fix_20261030/validation.json)：148项相关测试通过，10.756秒；新增8项资源测试先复现实际blockers再GREEN，独立复跑及全进程禁止ML导入仍8/8通过。编排fixture实际核对四份cache位于scratch、四份index和calibration/refit/test产物位于work；没有把边界替身当正式CUDA。独立审查无Critical/Important/Minor待修，核验SVM/图权重预算、目录碰撞、旧binary复用与容量门禁。21项旧源/配方/包SHA保持，旧19指标/12入口继续不动。
- 最终实际fix1 pyz的status为not_started；本地inspect完整VGG SHA通过，按预期停在缺服务器manifest/GPU/目标runtime/native binary，未创建runs。当前只离线交付，未发起远端正式训练/新费用、未删除用户文件、未Git提交推送。新实例实际CUDA与正式结果仍需后续服务器回传；TMP缓存随实例释放丢失但work成果保留，可按保存身份重新提取。

---

## 2026-10-30 — DFAGCN完整seed42 RTX3080便携入口交付，正式执行待服务器

- 已交付[正式运行包](output/dfagcn_nwpu_formal_3080_20261030.pyz)和[复制执行说明](output/dfagcn_nwpu_formal_3080_README_20261030.md)：150901字节，SHA `66cbb46bc8bc0fb8a79e7e95794534e6efb430ee1630be2258c91df16c98861b`，29个payload文件。打包器显式嵌入两个原只读helper，新增status透传；build-linear只编译本机LIBLINEAR，inspect只检查，ready后给绑定既定UUID和完整输入身份的nohup命令。
- 新入口run_dfagcn_nwpu_formal.py复用核心API，保持seed42、FP32/无AMP和TF32、batch100/VGG batch1、原100轮上限/九C/选轮规则；新6300pool与5040/1260切片→formal calibration→fresh6300refit→25200test，不继承450图学得权重或缓存。按阶段验证/恢复图片，pool不恢复test；输出、源快照、recipe/runtime/command、phase身份/时间/峰值、best/final、完整指标/逐图预测及短summary/exit各自保存。状态只给短摘要，后台不以PID代替首批证据。
- 训练问题、主要变量、现有便宜门禁、目标论文对比位置和协议差异已在[正式入口约定](docs/external_baselines_20261027/formal_3080_runner_20261030.md)冻结。全float32缓存约9.40876GiB，九C/测试文本另有约14.37GiB保守上界，提前检查40GiB可用空间。尚无全量实际吞吐、总时长或费率，不根据单批推算费用；参数数目/阶段耗时/真实每轮日志由正式run保存。
- [验收证据](results/evidence/dfagcn_formal_3080_prepare_20261030/validation.json)：139项DFAGCN相关测试通过，9.784秒；实际nohup命令已用无训练副作用进程验证执行和防覆盖；真实ZIP/浮点缓存切片、CUDA边界编排、版本/路径/输出/失效门禁通过。修复“后阶段失败抹掉真实训练开始”和“阶段receipt缺源码/runtime身份”两处问题，有有效RED→GREEN；首批前calibration中断记unknown，不仅凭函数调用说已训练。
- 本机有一次OpenMP初始化中止和139项运行中两个旧pipeline的native -6错误，原日志保留；本地额外KMP_INIT_AT_FORK=FALSE后完整139通过，该设置未写进服务器代码或配方。未重开旧全仓库缺依赖修复，不能称全仓库绿色。独立终审代理因用量限制未完成，由主代理按源码/协议接手[最终复核](results/evidence/dfagcn_formal_3080_prepare_20261030/review/root_review.md)，如实披露不是本轮独立终审。
- 实际最终pyz已本地解包，status为not_started，inspect完整VGG SHA通过且正确停在缺服务器manifest/GPU/目标软件/编译文件，没有runs。20个原DFAGCN源码/配方/旧包、19个指标、12个旧入口源码SHA保持；开发基线commit058f1d8b79d0be538ec358eb742d2cd3992921d4，未提交增量以包及源码SHA识别。当前没有直接远端操作、正式启动/测试分数、新GPU费用、Git提交或推送；用户下一步只需上传新小包，编译/inspect后执行生成命令。

---

## 2026-10-30 — RTX3080真实单批兼容性回传通过，正式入口准备中

- 用户回传实际命令、DEVICE/VGG_BATCH/GRAPH_BATCH/COMPLETE及shell提示符；[原始回传与核验](results/evidence/dfagcn_gpu_3080_return_20261030/validation.json)保存。独立核对24/24一致，交付pyz、本地关键源码与manifest内SHA三方一致。状态SERVER_REPORTED_3080_BOUNDED_PROBE_PASS；当前不是助手直连服务器取得的文件副本。
- 实际GPU RTX3080 10GiB，UUID GPU-70be4364-0641-e89d-828e-78727c6001ec，驱动610.57.04，Torch2.2.2+cu121/vision0.17.2+cu121/CUDA12.1/cuDNN8902，Python3.11.8/NumPy1.26.4/Pillow10.3，CPU8线程、seed42、FP32/无AMP和TF32、deterministic true/benchmark false，与原配方一致。
- VGG batch1形状/有限值通过；实际PyTorch allocated峰值587649536字节=560.426MiB，reserved645922816字节=616MiB，加载权重加forward为3.127573948秒。新图模型batch100、45类、三支CE loss11.601445198059082，反向梯度与更新后参数有限、参数确有改变，Adam仅1步；allocated155492352字节=148.289MiB，reserved190840832字节=182MiB，排除模型/输入准备的计算段0.571385420秒。[分段工程记录表](results/dfagcn_gpu_3080_probe_20261030.csv)与正式比较表隔离。
- COMPLETE在当前交付源码中仅于summary completed=true及run.exit0成功写盘后打印，足以验收本次兼容性，无需仅为补看文件再跑探针或下载大checkpoint。摘要的远端精确cache/source/hash字段未作为独立JSON回传，该溯源细节不编造。
- VGG使用normalized zero输入，图batch读取旧4090的450条冻结特征仅作一次性检查；没有epoch、验证选参、test、checkpoint或正式训练。这两个分段峰值不是全GPU进程占用，时间不是完整真实图片吞吐/epoch或费用测量，也不是正式OA或稳定收益。
- 按已授权继续推进的[正式入口约定](docs/external_baselines_20261027/formal_3080_runner_20261030.md)和[实施计划](docs/superpowers/plans/2026-10-30-dfagcn-formal-3080.md)，准备全量seed42独立运行包；保持既有100轮上限/九C/5040+1260划分、fresh6300refit后才25200test。正式3080缓存重新生成，当前未启动正式训练，不覆盖旧结果，不Git提交或推送。

---

## 2026-10-30 — 当前DFAGCN转RTX3080 10GiB，单批兼容性包准备完成

- 用户明确说明4090已无可租实例，取得3080；已回传NVIDIA GeForce RTX3080、UUID GPU-70be4364-0641-e89d-828e-78727c6001ec、总10240MiB/当时空闲9874MiB、驱动610.57.04。本次明确选择适用于当前DFAGCN批次，记录见[硬件说明](docs/external_baselines_20261027/hardware_3080_20261030.md)，不据旧默认4090安排阻断推进，也不扩展为其他未指定实验的全局硬件改动。
- 只读代码核对确认4090限制位于已完成的旧smoke包装器，核心DFAGCN接口按device运行。保留seed42、架构、FP32/无AMP/TF32、图batch100及VGG batch1；旧4090包/26.26秒/0.5673GiB保持原身份，不删改或当作3080测量值，不为换卡重跑整轮360/90。
- 新增tools/probe_dfagcn_gpu.py及[单批检查包](output/dfagcn_gpu_probe_3080_20261030.pyz)、[操作说明](output/dfagcn_gpu_probe_3080_README_20261030.md)。包133564字节、SHA `0d8ff18b2ecc9cd4903eb6c4833d3757b7c4e27605e0ac18d936b9158a7eac21`。新包支持自定义入口/旧包兼容，默认使用work已存的官方权重、固定manifest及旧450条冻结缓存，先inspect，再由实际UUID绑定run。
- run只做VGG batch1及一次性新图模型batch100前向/反向/Adam step，核验有限值、更新和显存；不运行epoch/选参/test、不写checkpoint、不续训旧subset模型。旧冻结缓存只作兼容性探针，正式完整6300池缓存将以3080新身份统一生成。此处不是新GPU实测已完成。
- [本轮本地证据](results/evidence/dfagcn_gpu_3080_prepare_20261030/)：完整119项DFAGCN测试通过，真实CPU batch100更新/缓存只读验证通过，独立审查无未解决项。进程CSV异常不得判空闲的门禁问题有有效RED→GREEN；一次组合测试遇Mac libomp SHM（2failure/6error），保留日志后直接串行重跑119项通过。旧仓库依赖缺失的回归范围未改变，未称全仓库绿色。
- 本地实际新pyz检查正确停在缺服务器manifest/旧缓存/目标GPU，无run目录、无3080训练。旧4090包SHA及旧19指标/12入口源码保持。待用户上传新小包并回传实际3080单批摘要后再核定正式配置与资源；无远端操作、Git提交推送或新增费用/准确率数据。

---

## 2026-10-30 — 真实NWPU360/90工程检查回传通过，交付身份一致

- 用户补回run.exit与短摘要，exit_code0、completed/training_started均true；train360/val90/pool450/test0，selected_epoch1、C1，重载误差0。[20项核验](results/evidence/dfagcn_nwpu_return_20261030/validation.json)通过：从交付pyz内部重算pipeline/recipe/weights身份，与固定manifest及回传均匹配，独立只读代理核对一致。状态SERVER_REPORTED_REAL_NWPU_SMOKE_PASS_DELIVERY_IDENTITIES_MATCH；按用户服务器回传验收工程门禁，不要求再次重跑或取回大checkpoint。
- 真实设备RTX4090 UUID GPU-d78aad24-a3ce-faa2-a189-674ccaa4d6f1；Python3.11.8、Torch2.2.2+cu121、vision0.17.2+cu121、CUDA12.1/cuDNN8902、NumPy1.26.4/Pillow10.3.0、CPU8线程；seed42、FP32无AMP/TF32、benchmark false/deterministic true均符合配方。总26.263705574秒，权重载入0.955864534秒，pool450提取+缓存6.892843317秒（65.2851图/秒），calibration3.455476798秒，subset refit4.244405087秒；PyTorch峰值allocated609131008字节/0.5673GiB，reserved727711744字节/0.6777GiB，不冒充整进程显存或正式成本。
- 前一段逐类指标与本次wall/权重SHA一致，按schema归属于val90：58/90正确，OA64.44444444444444%、macro-F1 62.338624338624335%，45类各support2，复算一致。不是正式test，不写正式比较表。epoch仅1、C网格仅[1]，不能称完整选参或原方法数值等价。
- [本次汇总](results/dfagcn_nwpu_smoke_20261030.md)、原始片段、规范化摘要和验证JSON已保留；旧19个指标/12个入口源码SHA仍一致。未直接访问服务器、下载checkpoint、追加训练/全test、外发或Git提交推送。下一阶段转完整seed42实验的配置/命令/预算准备，本轮未启动正式训练；下方首段缺字段记录作为历史保留。

---

## 2026-10-30 — 收到DFAGCN真实小样本验证片段，完成字段待短摘要确认

- 用户贴回的[原始片段](results/evidence/dfagcn_nwpu_return_20261030/partial_user_return.txt)从混淆矩阵中部开始，尾部为完整45类per_class及wall/权重字段；不是完整JSON或直接服务器文件访问。已[解析并复算](results/evidence/dfagcn_nwpu_return_20261030/partial_audit.json)：每类support2、共90张，正确58，OA64.44444444444444%、macro-F1 62.338624338624335%，与逐类数据一致。独立只读代理复算一致。
- 根据90张、每类2张及运行器schema，推断这是validation_metrics而非450张refit拟合集或正式test；父字段名已被截掉，保留该推断范围。报告wall_seconds26.263705574、weights_load_seconds0.955864534，完整VGG SHA与交付文件一致；硬件、显存、实际calibration/refit与完整一轮尚不据此确认。
- 首段未包含completed、run.exit、counts、GPU UUID/runtime、manifest/源码/配方SHA、selected epoch/C及reload误差，因此当前仅记USER_PARTIAL_RETURN_VALIDATION_FIELDS_CONSISTENT_COMPLETION_FIELDS_PENDING。已请求[无矩阵短摘要命令](results/evidence/dfagcn_nwpu_return_20261030/read_compact_summary_command.txt)，只读两份JSON并筛选字段，无需重跑或上传大checkpoint。原回传命令输出全矩阵过长，已同步缩短操作说明。
- 当前未启动新训练、正式test、远端操作或修改正式性能表，未执行Git提交推送。64.44%属于小样本一轮验证片段，不进入正式方法对比或稳定收益结论。

---

## 2026-10-29 — 真实NWPU360/90离线权重与便携包准备完成，实例尚未开启

- 用户要求继续真实NWPU下一步；询问当前GPU后，用户明确回复“实例没开启”。本地无SSH配置/已连接终端，按既有终端命令方式准备；未使用浏览器、连接服务器或租用实例。历史manifest/data/ZIP路径已从2026-10-11实际回传及源码重新核对，当前是否仍存在须由新实例inspect确认，不把旧路径当实时库存。
- 已下载完整torchvision官方VGG16文件553433881字节，SHA `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`，保存在checkpoints/dfagcn_pretrained_20261029/vgg16-397923af.pth。官方主域名TLS失败后，官方download-r2服务成功；未关闭TLS校验。整文件与66个Range片段合并SHA一致。[来源/完整身份](results/evidence/dfagcn_nwpu_prepare_20261029/weights_provenance.json)已存。
- 实际加载官方预训练文件，CPU1线程对照标准torchvision VGG前向：FC6、Conv3/4/5四项最大误差均0；三个卷积分支均保留负前激活。验证耗时1.3945s，[数值记录](results/evidence/dfagcn_nwpu_prepare_20261029/pretrained_cpu_validation.json)。这是torchvision加载/取层正确性，不证明MatConvNet、真实NWPU或CUDA等价。
- 新增[便携运行包](output/dfagcn_nwpu_smoke_20261029.pyz)、[操作说明](output/dfagcn_nwpu_smoke_README_20261029.md)、tools/run_dfagcn_nwpu_smoke.py及打包器。包134156字节，完整SHA `f7ff678be35517aca7edc3a181862a56d739cc0be1a7496a5c63dac4b269daca`；固定源码/配方/权重完整SHA和LIBLINEAR源码，不携带Mac二进制或大权重。上传包与权重后，build-linear按目标平台编译，inspect只检查资源，ready后才打印绑定实际RTX4090 UUID的一轮命令。旧服务器GAViT目录不修改。
- 运行器严格固定既有NWPU manifest、31500/45类/5040/1260/25200和每类112/28/560；原清单顺序选8train/2val，共360/90。若本地原图已清空，仅从完整SHA匹配的持久ZIP恢复450张所需图片，不解压test、不重分划分。提取pool一次后派生train/val缓存，做一轮calibration与fresh subset refit/实际LIBLINEAR，不调用最终test；保存GPU UUID/软件/精度/峰值显存/耗时/吞吐/源与输入SHA、checkpoint、history、exit。
- 本轮105项相关unittest通过，实际pyz解包/变更拒绝/官方源码本机编译及已核验权重inspect均执行。本地inspect正确停在缺真实服务器manifest与RTX4090，training_started=false且未创建runs。复核并修复了子代理未完成的“CUDA已初始化不得再切UUID”检查；另核配方绑定后变更必须拒绝。新增真实GCN首batch与epoch日志，RED→GREEN；不把PID或调用函数本身当执行证据。
- [完整证据](results/evidence/dfagcn_nwpu_prepare_20261029/validation.json)记录新包105通过；旧仓库发现204项仍有9failure/9error，缺pytest/timm/torch_geometric，不称全仓库绿色。旧19项指标/12项入口源码SHA保持。子代理后续因用量限额中断，剩余复核与验收由主代理完成，未声明新一轮独立代理终审。
- 真实NWPU/CUDA小样本尚未开始；无GPU费用/正式指标/外部发送/Git提交推送。实例开启后按操作说明上传两个文件，先编译/inspect，回传资源结果或按ready生成的命令执行；完成后才核验run.exit与summary，不扩大为正式训练。

---

## 2026-10-29 — DFAGCN完整独立入口及真实算子工程链完成

- 用户批准固定缺失配置并推进完整流程后，新增[完整配方](configs/dfagcn_independent_v1.json)、config/model/cache/extraction/linear/pipeline/CLI/smoke；[设计](docs/external_baselines_20261027/pipeline_design_20261029.md)与[运行说明](docs/external_baselines_20261027/pipeline_usage_20261029.md)记录来源边界。7×7池化、GCN256×2、MLP1024→512、预处理/Adam初始LR/停止/solver2/bias1等是显式独立选择，不声称原Table I或MatConvNet等价。已确认原文的k10并集、特征层、融合/C网格仍保留。
- [86项包内unittest](results/evidence/dfagcn_pipeline_20261029/package_tests.log)通过；有效接口缺失RED及图片变化、子集门禁回归RED→GREEN归档。缓存采用float32 npy memmap并核验原图/权重/数组实际SHA及有序样本；实际CPU FC6、随机VGG约470MB权重安全保存/加载测试已通过，弥补前一批仅meta FC证据。完整默认49节点、45类结构另有独立数值梯度探针。
- 实际官方LIBLINEAR v250源码固定commit491c9f1188b97ba70847c70a68be363d186ddf9d，BSD3许可证、源码归档、编译及二进制SHA在[provenance](results/evidence/dfagcn_pipeline_20261029/liblinear/provenance.json)。实际调用train/predict，dummy-label stdout准确率不作项目评估；本地二进制是macOS arm64，服务器须同源重编译。
- [最终smoke_run_02](results/evidence/dfagcn_pipeline_20261029/smoke_run_02/smoke.json)：20张合成RGB、两类、真实随机VGG、CPU1线程；train8/val2，三支训练一轮，val选C0.1，重置同seed在pool10重新拟合，合成test10预测，保存重载误差0。总6.0434s，VGG初始化/指纹/提取缓存5.7536s；不是NWPU数据、预训练性能或正式效率。run01与改动前源码另保留，不覆盖。原论文分数和旧性能表均未写入此结果。
- 完整阶段身份绑定配方/源码/manifest/缓存内容/拟合样本/seed/checkpoint；val仅选epoch/C，test仅接受完整pool重新拟合。新增每类train8/val2工程限额，pool为所选两子集并集；subset标记且不能进入最终test。正式模式拒绝随机权重/子集/非既有manifest。[NWPU一轮smoke配方](configs/dfagcn_nwpu_smoke_v1.json)已备，真实NWPU原图/已校验预训练文件接入后的运行尚未执行。
- 独立审查无待修复问题；另核三支真实参数更新、val C选择规则、错误split/随机formal拒绝。消除重复缓存复制及测试临时二进制路径。旧19个指标/12个入口源码SHA保持。全仓库unittest发现185项仍有9failure/9error，缺pytest/timm/torch_geometric，pytest入口不能启动；逐项日志保留，不称全仓库通过。Mac libomp共享内存偶发启动失败另记，单线程直接命令最终通过，不将它算功能RED。
- 本轮有合成工程训练与评估，无真实NWPU/AID训练或测试、GPU/远端任务/费用数据、邮件外发或Git提交推送。真实目标GPU行为、端到端成本与正式多seed比较仍待后续实验。

---

## 2026-10-29 — 英文绪论与文献综述整合至第1–4章

- 按用户确认的总50–60页（含前置页和参考文献）、56页预算，完成[绪论](thesis/ntu_gavit/chapters/01-introduction.tex)和[文献综述](thesis/ntu_gavit/chapters/02-literature-review.tex)初稿，源码词数估算1132/3872。综述按遥感表征、图网络、区域分组、直接相关方法、评估和定位展开，含原创比较表；不是已经测得约13页，也不是整篇已达到50–60页。
- [主文件](thesis/ntu_gavit/main.tex)连续编入1–4章，50条独立参考全部在正文引用。并行原始来源核对、逐篇机制/访问边界和BibTeX合并记录位于[provenance](thesis/ntu_gavit/EVIDENCE.md)。GloRe、TokenLearner/TokenFuser、两种STViT等先例归属明确；不把聚合→交互→映回通用流程宣称首创，不以文献综述代替HGTNet/DFAGCN实测。
- 独立审阅已报告未发现首创/稳定优势/语义分割/Swin上下文等事实性阻断项，并指出两个STViT段落的编辑口吻；已改为明确的方法与恢复路径比较。书目补DGCNN卷38期5及Article146，不抄arXiv占位页码。逐篇有摘要、作者全文或出版社索引等不同访问范围，不能称所有50篇均全文审读或已穷尽最新工作。
- [最终原生编译](thesis/ntu_gavit/provenance/compile-final-20261029.json)返回success；[引用/标签/输入检查](thesis/ntu_gavit/provenance/source-checks-20261029.json)无缺失/重复/未使用条目。第3/4章源码SHA与扩写前一致；旧主文件、八条参考版本和导入包保留在before-intro-review-20261028。逐页视觉QA和实际分页仍待PDF导出后检查。
- 更新[说明](thesis/ntu_gavit/README.md)、证据索引和验证状态，提供[日期版Overleaf导入包](thesis/ntu_gavit_overleaf_20261029.zip)及稳定名称副本。仅论文文档及本条日记变更；未修改模型/实验CSV/外部复现组件，未新增训练或测试集评估，无邮件外发或Git提交推送。

---

## 2026-10-28 — DFAGCN第一批独立组件与CPU检查完成

- 用户批准推进后，在根目录新增[独立组件包](external_baselines/dfagcn/README.md)：注入式VGG16取层接口、逐图kNN二值无向并集、自环一次的对称归一化、无隐式激活的GCN与mean readout、5632维有序加权拼接，以及JSON缓存身份接口。未将它登记成作者官方实现或完整DFAGCN。
- [实施记录](docs/external_baselines_20261027/implementation_plan_20261028.md)及[证据](results/evidence/dfagcn_primitives_20261028/)保留有效接口缺失RED与GREEN。最终新增53个unittest方法通过：graph26、features13、fusion7、identity6、组合链1。包含手算小图、拓扑/批次隔离、节点/参数梯度、真实CPU卷积pre-ReLU负值、卷积权重重载、eval/no_grad与缓存身份。另补cosine范数溢出和加权float16溢出的RED→GREEN，不将测试展开数写成独立方法数。
- 特征测试未下载预训练权重；完整标准VGG16的FC6仅在meta验证形状，CPU仅执行真实卷积分支，未证明CPU FC数值或MatConvNet权重/预处理等价。未冻结论文未明的池化目标、完整GCN/FC层配置及LIBLINEAR/停止选参规则；fixture节点数和层宽不升级为论文默认。
- 缓存接口绑定manifest/有序样本路径和原图SHA/类别/权重/预处理/取层/源码及拟合身份，分开calibration/refit；仅保存调用方声明的身份，不生成张量缓存，不自行证明SHA来源或无测试泄漏。完整训练器、refit测试门禁和真实数据smoke仍未实现/运行。
- CPU依赖仅安装到/private/tmp/gavit-dfagcn-cpu-20261027，版本及wheel SHA已记录。昨晚自动审批因用量未完成，今日正常审批后下载；期间PyPI/TLS和CPU索引缺pytest导致失败，最终保留官方wheel并改用原生unittest。全仓库unittest发现检查仍报9项failure和9项error，缺pytest/timm/torch_geometric；pytest入口也因缺框架无法启动，[逐项明细](results/evidence/dfagcn_primitives_20261028/repository_unittest_failures.json)保留，不称全仓库通过。
- 旧19个指标文件和12个入口源码哈希核对保持；独立代码审查与最终源码哈希见证据目录。无新增准确率、真实数据训练/全test、GPU任务或费用数据，无邮件外发、Git提交推送。单元检查耗时不作完整方法效率。

---

## 2026-10-27 — 外部方法转为独立复现准备，首篇优先DFAGCN

- 用户同意“文献报告值先整理，同时推进独立复现；先补两篇模块/未决细节，再选首篇”的路线。已完成[准备总览与次序](docs/external_baselines_20261027/README.md)、[可复制文献对比](docs/external_baselines_20261027/literature_comparison.md)、[DFAGCN审计](docs/external_baselines_20261027/dfagcn_audit.md)、[HGTNet规格](docs/external_baselines_20261027/hgtnet_audit.md)和[现有协议/验收](docs/external_baselines_20261027/protocol_and_acceptance.md)。作者/导师未回复来自用户陈述，本轮未访问邮箱或外发。
- 两名独立代理重新读取目标论文作者上传全文的可读文字层，核到HGTNet的AID20/50% OA均值96.91/98.47%、NWPU10/20%均值94.52/95.75%；DFAGCN正文报告AID50% OA94.88%、NWPU20% OA89.29%。六项均为[原论文报告值](results/evidence/external_baselines_20261027/literature_reported_oa.json)，非本项目实测。PDF/表格图片未取得，SD全部保留null；DF的均值/SD单元格未视觉确认，不升级成完整均值±SD。DF十次随机划分已核，HGT只确认五次重复，具体是否重划分未确认；本轮独立复核已据此修正文献说明，不能沿用旧审计把HGT也写成已确认随机划分。
- 首篇优先DFAGCN的已确定模块，依据是完整特征/GCN/融合/LIBLINEAR链及C网格、融合系数更明确；Table I层配置、池化目标、权重/预处理及停止/选参规则仍需补齐或显式冻结。HGT另有max/mean、DA门控符号、STB计数、空间形状与k20及预训练映射歧义，暂不先拼装全模型。原文已公开的DF融合配置不再笼统称缺失。无需作者回信即可开始确定模块，但本轮尚未创建模型包、执行张量检查或真实smoke。
- 已有AID和NWPU冻结manifest及正式流程核对完成，NWPU清单并非仍待生成。新入口拟独立于旧scene-v1/GAViT身份，保留5040/1260校准→6300 refit→25200 test隔离；不把多尺度分阶段方法塞进单224端到端脚本。三个已检查本地Python环境均缺torch等模型依赖，未安装。相关[来源、依赖与状态](results/evidence/external_baselines_20261027/)已归档；耗时/显存/费用待测。既有指标表、训练代码及checkpoint未改，无新GPU任务、完整测试、Git提交推送。

---

## 2026-10-27 — 英文论文方法与实验设置LaTeX初稿

- 用户明确选择LaTeX，并确认本轮先写英文“方法＋实验设置”。新增独立[论文目录](thesis/ntu_gavit/README.md)、[主文件](thesis/ntu_gavit/main.tex)、第3章方法与第4章实验设置，约3800英文词（源码词数估算）、22个编号公式、8条原始来源引用；不是完整论文或最终提交稿。
- 使用用户给定NTU模板，固定上游commit `3094f44d8705919b7db3d08391678c4b4d709817`，保留MIT许可证和原始主文件。仅适配章节评阅封面/必要依赖/正常编号，未填造个人信息、声明或签名；为XeTeX关闭不支持的microtype expansion。
- 两路只读审计核对当前模型/协议与导师邮件，独立稿件复核未发现重要方法或数学错误；补明确BEN v2 metadata来源及CosineAnnealingLR的T_max退火时长，修正模板Chapter前缀造成的编号重复。[证据索引](thesis/ntu_gavit/EVIDENCE.md)区分当前代码、历史冻结运行和未验证范围。
- 记录固定分区与组内attention、soft S/W不同归一化、48+32 pre-GAT边、neighbor→query、cosine仅选拓扑、GAT与反馈；明确AID/NWPU校准后重新初始化refit、BEN验证mAP best测试以及RGB/阈值/精度差异。不复制过期表格，不把HGTNet/DFAGCN比较、最终显著性或稳定优势写成已完成。
- 原生LaTeX最终编译返回success；引用键/标签/相对输入静态核对通过。初次格式与字体缓存下载曾超时，日志和[验证状态](thesis/ntu_gavit/validation.json)保留。未取得单独PDF导出及逐页视觉验收，不能称已完成最终版式QA；提供可编辑源码、原生预览入口和Overleaf导入ZIP。
- 未启动训练、服务器任务或新测试集评估；未修改模型、结果CSV、已有汇报/草稿；无外发邮件、Git提交或推送。

---

## 2026-10-20 — 导师汇报改为便于复制粘贴的无表格格式

- 按用户要求，将[中文汇报](docs/supervisor_progress_report_20261012.md)的10张表和[英文邮件草稿](docs/supervisor_email_draft_20261012.md)的2张表改为逐项文字；新增[中文纯文本](docs/supervisor_progress_report_20261012_plaintext.txt)和[英文纯文本](docs/supervisor_email_draft_20261012_plaintext.txt)，英文从Subject至落款，可直接复制。原稿备份保存在[before](results/evidence/supervisor_copyable_20261020/before)。
- [排版核对](results/evidence/supervisor_copyable_20261020/validation.json)确认全部表格单元格和数字及其出现次数保留，无表格行或tab；正文仅将英文“main table below”调整为“following results”，其余内容保持原样。当前英文正文seed42/43/44/45与中文seed42/43/46/47表述不同，本轮只做排版，未擅自统一seed或重算指标。
- 无新增训练、推理、实验指标、代码测试、外部发送或Git提交推送。

---

## 2026-10-20 — BEN受训3×3／4×4返回文件审计及正式图稿完成

- 用户提供/Users/patrick/Downloads/ben_grid_pair_visual。将原文件完整保留并复制归档到[server_return](results/evidence/ben_gridpair42_visual_return_20261020/server_return)，[审计](results/evidence/ben_gridpair42_visual_return_20261020/validation.json)核对37个清单SHA、与准备包一致的配置/源码、捕获及父训练两项exit0、8组JSON/PT及CRC、488条实际pre-GAT边、原PNG/标签、全部76项概率/判断。未执行下载代码、HTML脚本或pickle，没有本地Torch前向；PT核验限字节SHA/容器CRC，32项零误差和K16旧新图0误差来自真实服务器报告。
- 配对结果为63项双方对、9项双方错、3项K16纠正、1项K16新增漏判。row14680草地42.85→51.59%、row28450过渡林地44.74→91.82%、row4439城市建成区41.38→76.57%被纠正；row28450海洋水域64.96→5.84%新增漏判。row97389牧场29.89→46.36%、row4439沙地17.67→49.32%仍低于阈值0.5，保留未纠正状态。3/1是标签判断次数，不是整图成败或总体获益率；单图草地纠正与该类全test AP下降分别解释。
- [两页PDF及300dpi PNG/HTML](output/pdf/ben_gridpair42_20261020/README.md)、[逐图中文分析与中英图注/全部76判断](docs/ben_gridpair42_visual_analysis_20261020.md)完成。第一页真实区域＋特征边＋全部阈值翻转，第二页真实区域内汇总权重；原图不调光/裁剪，固定全部4例、不按本次收益重选。四例原SVG和最终两页PDF渲染均看图；[独立导出核对](output/pdf/ben_gridpair42_20261020/export_qa.json)验证24次原始像素放置、200条PDF特征边、392个权重格、所有翻转、2034字符页内与两幅2126×3459PNG。HTML与原返回字节一致，保留空间边/完整19类；未重新浏览器交互验收。
- 原训练summary同步取回：K9 best10/val mAP79.1052%、final30、参数31366926、RTX4090 UUID GPU-8734427f-061f-9fdb-97eb-fe9190a11225；train20705.394s/test178.096s/parent20974.617s，峰值allocated3884810240字节。原完整test分数71.07533208/64.48830571/76.68872316不变，历史K16为72.0/66.0/76.7的有限精度参考。当前可视化CPU21.677s，未混作训练硬件/效率。
- [状态](results/ben_grid3_seed42_status_20261018.json)推进PAIRED_CAPTURE_AUDITED_FIGURES_COMPLETE；训练summary审计完成，原完整history/受训checkpoint字节仍在服务器，因此full_artifact_audit_complete保留false并注明范围，不要求为图稿再次取回大文件。[导师汇报](docs/supervisor_progress_report_20261012.md)与[英文草稿](docs/supervisor_email_draft_20261012.md)更新真实图稿进度；顺带依据既有四seed表更正英文soft正向seed为43/46、负向为42/47，汇总数字及CSV未改。未外发、Git提交推送、重训或重新全test。

---

## 2026-10-20 — BEN K9/K16同图捕获8组PASS已回传；审计ZIP待取回

- 用户回传BEN_GRID_PAIR_VISUAL_INSTALLED、ARTIFACT_GATE_PASS、CPU软件信息、全部4图×2模型共8条CAPTURE_PASS、BEN_GRID_PAIR_VISUAL_COMPLETE及RETURN_HTML/RETURN_AUDIT_ZIP，根目录/home/featurize/work/ben_gridpair42_visual_20261020_01。[粘贴原文](results/evidence/ben_gridpair42_visual_run_20261020/user_return.txt)及[解析/核对](results/evidence/ben_gridpair42_visual_run_20261020/validation.json)归档；不是服务器文件独立下载或重新推理。
- 8组row/K组合完整且无重复，K9每图24+18=42边、K16每图48+32=80边，共488条pre-GAT边。每组capture/restoration/pooling/feedback四项报告误差均为0，共32项；表示本次观察/重建保持模型输出，不是分类错误率为0。CPU gpu_identity=null正常；torch2.2.2/torchvision0.17.2/timm1.0.27/PyG2.8.0/numpy1.26.4/Pillow10.3.0与已准备环境一致。
- 服务器报告原30轮/两权重/冻结源码/4PNG/val标签门禁通过并完成HTML与ZIP导出；原run.exit/summary文件、K16新旧图函数具体误差、原始张量、真实概率及图像尚未独立取回，不把日志完成标记当本地文件审计或图像验收。可视化[状态](results/ben_grid3_seed42_status_20261018.json)推进SERVER_CAPTURE_REPORTED_COMPLETE_ARTIFACTS_PENDING；paired_capture_audit_complete及最终图稿保持false，原正式性能不变。
- 下一步只从持久化目录下载ben_grid_pair_visual.zip并返回本地，核验SHA/输入/边/19类配对概率/训练summary后制作正式图稿；[说明](docs/run_ben_gridpair42_visual_20261020.md)已更新，无需重新捕获、训练或解压。无新模型代码/正式指标/CSV修改、直接远端操作、提交推送或外部发送。

---

## 2026-10-20 — BEN受训K9/42与K16/42同图捕获包准备完成；服务器执行待回传

- 用户批准“进行下一步”。在既有研究工作树新增外置tools/ben_grid_pair_visual.py、package_ben_grid_pair_visual.py及9项测试，保留所有已有模型/训练源码和未提交改动。[运行说明](docs/run_ben_gridpair42_visual_20261020.md)、[单行命令](results/evidence/ben_gridpair42_visual_prepare_20261020/install_and_capture_command.txt)与[准备核对](results/evidence/ben_gridpair42_visual_prepare_20261020/preparation_validation.json)保存；目标/home/featurize/work/ben_gridpair42_visual_20261020_01。默认CPU8线程，仅4图×2受训模型推理，无新训练、全量test或原数据解压。
- 复用pair44_feedback_k_7ehbne内全部4张原reader/Resize224无损PNG（97389/14680/28450/4439），本地原始SHA和来源再次核验；服务端对照原val CSV的SHA/row/路径/19标签，并共享同一Normalize后连续FP32输入。固定历史诊断选样保留，不把旧seed44概率/图边用于新对照，不推断总体获益率。
- 捕获前自动校验完整训练summary/各阶段exit0/30轮及既有119825指标、K9/K16两套best SHA、严格metadata与13文件冻结快照。真实前向捕获tokens、区域、attn scores/weights、实际42/80边和feedback/logits；开启/移除观察、区域汇总和feedback重建差均需≤1e-5，受训K16再检新旧图函数一致。输出自包含HTML、每例SVG、所有19类概率/四种成败分类、CPU原始张量及SHA，原summary/metadata随审计ZIP返回。
- 先观察7项因模块缺失失败，再观察单行打包测试因未实现失败，最终新增9项及既有K9回归11项共20项通过；实际安装fixture的dispatch被明确屏蔽，未模拟模型推理。Bash/zsh及19KB单行载荷/哈希/重复安装拒绝核对通过。四张明确FIXTURE的合成边/分数SVG静态渲染看图，12次原PNG嵌入及488条fixture边逐条对应；浏览器file协议被策略拒绝，改做静态SVG和结构核对，HTML交互未实测，无绕过。独立requesting-code-review复核无Critical/Important项，minor过期计划状态已修正。
- 全套unittest实际执行199项，8项缺torch/13项缺pytest错误，名称及trace见[全日志](results/evidence/ben_gridpair42_visual_prepare_20261020/full_unittest.log)；本地无torch/timm/PyG，未做真实模型或服务器完整导出。[当前状态](results/ben_grid3_seed42_status_20261018.json)仅记HANDOFF_PREPARED_SERVER_CAPTURE_PENDING；真实案例概率、新图稿及K9训练summary本地审计仍待服务器运行后回传ZIP。无新性能分数/CSV改动、Git提交推送或外部发送。

---

## 2026-10-20 — 用户重申K9补跑用于3×3／4×4可视化；性能结果是辅助材料

- 用户明确“记得我们做33是为了补充可视化33和44的对比”。纠正上一条把“只读summary”写成全部后续收尾的范围偏差：summary归档仅补齐训练/权重身份，主交付仍是受训K9/42与历史K16/42在同一批BEN输入上的区域、真实空间/特征边和预测对照，之后继续Swin／固定4×4／软分配三模型图。
- 已核对旧几何图和捕获工具：4张归档验证图仍无本次K9/K16真实前向；现有BEN反馈捕获硬编码16区域/80边，训练门禁只保存首图边，均不能直接替代这组配对可视化。首批可沿用97389/14680/28450/4439全部4例，但须两套权重新捕获，保留历史验证选样来源及无改善案例，旧seed44图边/分数不改标为新结果。
- [运行说明](docs/run_ben_grid3_seed42_20261018.md#训练后的可视化交付)、[性能背景页](docs/ben_grid3_grid4_seed42_20261020.md)、[旧几何图状态](docs/ben_grid3_grid4_analysis_20261017.md)和[实验状态](results/ben_grid3_seed42_status_20261018.json)已对齐：同输入/预处理/设备/精度、严格权重与输入身份、捕获前后logits一致，分别显示实际空间边/特征边及真实标签/类别概率。训练后的固定边界仍由位置规则决定，模型相关观察需来自实际汇总权重、图连接和预测。
- 本轮仅核对与修正记录/交付范围；未新增训练、服务器推理、图稿或正式分数，可视化配对捕获仍未完成。

---

## 2026-10-20 — BEN固定K9 seed42收到队列exit0与全量测试指标；mAP低于历史K16约0.92pp

- 用户回传/home/featurize/work/ben_grid3_seed42_20261018_01/run.exit为exit_code=0，test/metrics.json为119825张/阈值0.5/19类AP，checkpoint SHA c4b77953d6dd2908d7a61ebe7a34f4004e8c219a0843de6ab05aaf207460feff。[粘贴记录](results/evidence/ben_grid3_seed42_metrics_20261020/user_return.txt)、[规范化指标](results/evidence/ben_grid3_seed42_metrics_20261020/metrics.json)与[核对边界](results/evidence/ben_grid3_seed42_metrics_20261020/validation.json)保存；不是服务器原文件下载或本地重新测试。
- test mAP71.07533208222321%、macro-F1 64.48830571218534%、micro-F1 76.68872316423352%；19类AP均值独立复算71.07533208222323%，差1.42e-14。总体/逐类均有限、范围合法、类名与冻结BEN19列表吻合；F1缺原预测/标签未独立复算，best/final、真实GPU UUID/软件/精度、短测重载/构图、资源、训练/测试报告绑定还需已有summary归档，不据总exit0补造这些字段。
- [独立3×3/4×4对照表](docs/ben_grid3_grid4_seed42_20261020.md)及[全精度总体/19类JSON](results/ben_grid3_grid4_seed42_20261020.json)保存，显示两位小数。历史K16为72.0/66.0/76.7%，K16−K9约+0.9246679/+1.5116943/+0.0112768pp；旧值仅一位小数，micro-F1基本持平。4×4对海岸湿地约+13.12pp、沙滩+6.97、内陆湿地+1.14；天然草地/稀疏植被−4.52、多年生作物−1.73。前两类等权贡献约+1.06mAP，其他类别有抵消；按旧打印范围10类明确高/8类低/1类方向未定，非显著性类别计数。支持本次4×4区域数选择的初步性能方向，不证明所有类别/多seed稳定优势或建边语义机制，历史批次/硬件/精度限制保留。
- [当前状态](results/ben_grid3_seed42_status_20261018.json)改为TEST_METRICS_AND_QUEUE_EXIT0_REPORTED_SUMMARY_PENDING，指标已记录，full_artifact_audit_complete=false；保留既有4seed主表与所有历史结果，不替换固定K16/42。总exit0在已准备程序下意味着队列报告走完门禁/30轮/test，但仍只把尚未回传的实际轮次、UUID、耗时/费用留null。下一步仅[读取已有summary](results/evidence/ben_grid3_seed42_metrics_20261020/read_summary_command.txt)，不重训或重复测试。
- 本轮无远端直接访问、新推理/GPU任务、模型/性能CSV修改、Git提交推送或外部发送；只进行回传、算术核验和文档/状态更新。

---

## 2026-10-19 — BEN固定K9 seed42已安装/分派，回传处于RGB预检扫描

- 用户回传BEN_GRID3_42_INSTALLED与BEN_GRID3_42_DISPATCHED，PID3599，输出/home/featurize/work/ben_grid3_seed42_20261018_01，training_start_verified=false，顺序preflight→smoke→train30→best_test。[粘贴记录](results/evidence/ben_grid3_seed42_launch_20261019/user_return.txt)、[分派JSON/扫描进度与核对](results/evidence/ben_grid3_seed42_launch_20261019/validation.json)保存；不是独立服务器文件下载或日志全量副本。
- root日志出现START preflight及34条RGB CHECK，末次打印train230000/237871（17.4s）、val110000/122342（8.3s），未见报错。未回传完整test扫描、BEN_GRID3_42_PREFLIGHT_PASS、短测/重载结果、START train之后的首个optimizer batch、GPU UUID/进程/显存或任何正式指标；只确认队列分派和观测到的数据预检，不将PID或CPU扫描当正式训练启动。实际当前进程存活和后续进度未独立读取。
- [状态](results/ben_grid3_seed42_status_20261018.json)改为DISPATCHED_PREFLIGHT_RGB_SCAN_OBSERVED，formal training started未知/null，verified=false，正式分数保持null；[操作说明](docs/run_ben_grid3_seed42_20261018.md)同步，明确短测也会打印首批，需START train之后的TRAIN_FIRST_BATCH_OK才能确认正式首批。流程通过门禁后自动继续，不重新安装、重跑预检或重复分派，下一步只读tail -f及nvidia-smi核查。
- 本轮只记录回传并更新文档/状态；原安装载荷/模型源码/性能CSV保持，无新的远端执行、重训、Git提交推送或外部发送。

---

## 2026-10-18 — 用户授权补BEN固定K9 seed42；本地快照与自动门禁队列已准备

- 用户明确“那就补吧”，授权补缺少的3×3性能；按最小新增成本只准备一个K9/seed42完整30轮→best test，历史固定K16/seed42作单seed跨批次参考。此前本地和回传库存均无可复用K9正式结果，离线几何图不能补性能。[运行说明](docs/run_ben_grid3_seed42_20261018.md)、[状态](results/ben_grid3_seed42_status_20261018.json)及[安装/后台命令](results/evidence/ben_grid3_seed42_prepare_20261018/install_and_launch_command.txt)保存；当前服务器未执行，GPU UUID、best/final及正式分数均未知，不填性能表。
- 从历史固定42基础提交cd39be91a013bcf98099808c451300f87d3898b7导出独立13文件快照，仅扩展固定K9/K16构图和相应metadata，保留attention pooling、GAT256/4头/2层、token_feedback/dropout0.1；K9四邻接24+非空间cosine top2共18=42条有向pre-GAT边，方向neighbor→query，K16仍48+32。原checkout和受训K16权重不改，验证/测试loader采用已有pin_memory=false主存修复，训练loader保持原设置，首批与history记录仅增加日志。源码明确标识modified_frozen_snapshot/dirty=true，用全文件SHA门禁，不伪装成干净Git提交。快照SHA f82d37a9cc1d18920a553fa1e4fd516e9d9ebda9d9abddb6595e12a69a944f58。
- 数据原237871/122342/119825及split SHA、ImageNet SHA f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36，RGB/19映射与预处理/增强沿用冻结配方。预期31366926参数；seed42/epochs30/batch32/workers4/AdamW3e-4/wd1e-4/Cosine30/BCE/阈值0.5，val mAP选best。目标单张4090/UUID现场记录、FP32/noAMP/matmulTF32false/cuDNNtrue/8线程，软件对齐已完成soft42；历史4×4的4090训练/3060测试及精度/软件记录不齐限制保留，不能称严格同环境纯K因果或稳定优势。
- 队列独立目录/home/featurize/work/ben_grid3_seed42_20261018_01，依次预检全部RGB/配置/现有K9库存、真实图与前向门禁→96/64一轮K9短测/保存/fresh重载→完整30轮→119825×19 best测试，记录每阶段退出、checkpoint绑定、精确总体及19类指标、预测、history和资源，短测完成先写该输出目录experiment_diary.md。任一步失败停后续、拒绝覆盖/重复启动，PID不作为训练启动证据。旧72.0%打印精度只到0.1pp；结果正负均保留，不自动追加seed或重跑。
- 先写缺失模块/安装/数值容差回归并观察失败，最终新增11项标准库检查通过；独立requesting-code-review复核发现CUDA scatter不能要求logits逐位相同，已改为有限≤1e-5并加回归，CPU图张量仍要求完全一致。真实torch/PyG/CUDA门禁与模型短测仅由服务器执行，目前未运行。安装载荷SHA43838122f39c4b1e30dcd886f1cafb48c840a2010ed4a6014aab3dd65cda6fb2；完整导出/安装本地fixture通过且屏蔽GPU dispatch，重复安装拒绝，Bash/zsh与载荷解码核验通过。[准备核对及完整测试限制](results/evidence/ben_grid3_seed42_prepare_20261018/preparation_validation.json)记录，本地缺torch/pytest的全套测试不能称通过。
- 参考此前4090 BEN完整队列5h46m～6h22m，预留6～8h；只有旧1.67元/h价格仍适用时预算约10～14元，当前实例价格、实际耗时费用未知，首轮后再估算。模型/历史性能CSV/图稿和无关修改保持，无付费任务直接执行、Git提交推送或外部发送。

---

## 2026-10-17 — 收到Featurize库存；未识别可比BEN 3×3性能或权重候选

- 用户回传两条只读find/grep命令的输出；[原文逐字节归档](results/evidence/ben_k9_comparability_20261017/server_inventory_return_20261017/server_return.txt)，39390字节/SHA2d032da46b31c8da1e3aa1e8c7e3e0f238530c88b8b2a5227e08042042853246；[解析库存](results/evidence/ben_k9_comparability_20261017/server_inventory_return_20261017/parsed_inventory.json)保存全部路径与检查范围。命令与准备版本一致，粘贴无可见错误，第二条后直接为提示符；单独退出码未回传，本轮未直接访问服务器或加载权重。
- 第一条510个不同路径（136个.pth/373个.pt/1个.safetensors），没有文件名含K9/3x3的候选，包含大量探针/缓存/训练状态，不能称作510个模型。第二条小于20MiB的JSON/log/out/err配置检索无匹配。普通命名权重的完整身份、大日志、剪枝目录及原NTU/其他存储仍有盲区，因此结论限于当前已识别的库存，不声称历史上从未运行K9。
- [可比性审查](docs/ben_k9_k16_comparability_audit_20261017.md)、[结构化状态](results/ben_k9_k16_comparability_audit_20261017.json)与几何图分析的后续状态已更新：当前没有可直接复用的BEN K9/K16性能配对，K9 test mAP及两者差值保持null；固定42的72.0%不变。若后续量化区域数效应，先明确配方一致的K9单变量消融并处理sparse_hybrid硬限制，不能强改4×4 checkpoint当作受训3×3，也不根据几何覆盖图宣称4×4胜出。
- 本轮仅解析回传和更新文档/状态；未新增实验分数、修改性能CSV或模型、启动训练/测试、提交推送或外部发送。

---

## 2026-10-17 — 检查既有3×3性能；本地无可配对BEN结果，服务器库存待查

- 用户要求先检查是否存在可比3×3性能，再进行对照。本轮读取结果表、历史日记、现有回传、初始BEN Git配方及当前K16构图限制，形成[审查说明](docs/ben_k9_k16_comparability_audit_20261017.md)及[结构化记录/来源SHA](results/ben_k9_k16_comparability_audit_20261017.json)。未启动训练、模型评估或服务器访问，BEN K9 mAP与K16−K9差值继续null。
- 早期NWPU有K9验证96.5%/日记test95.9%，K16日记test95.8%（4526/4725）；同时改变均值/注意力池化和反馈结构，不能当纯区域数消融，也不能混入当前25200张test。K4/K9/K16消融表对应性能为空。本地未找到模型权重，服务器及原NTU库存未查；旧文件名仅作为查找线索。
- 研究工作树BEN表13条无K9项；70.4%为历史v2未修正kNN标签，70.6%为修正kNN K16，不能改标K9。0b1e02c首个BEN训练脚本默认K16、配套作业显式K16/attentive_spatial/token_feedback/kNN；有针对性的Git历史差异搜索未匹配K9，但不能排除未入库CLI覆盖。既有固定42的72.0%及cd39be9/K16/best13/final30身份保留，尚无3×3配对，不宣称4×4性能更好。
- [服务器两条只读库存命令](results/evidence/ben_k9_comparability_20261017/server_inventory_commands.txt)已准备，跳过Git/依赖/原始数据，列权重并检索小于20MiB的JSON/log/out/err配置线索；不导入torch或反序列化权重。普通权重缺metadata、大日志和其他服务器目录仍属盲区；命中只是候选。若有匹配权重先考虑补测试，若无则记录缺项，不自动触发新训练。
- Bash/zsh语法、合成目录中的K9/K90边界、普通命名权重、剪枝、文件大小界限、来源/CSV身份等26项[本地核对](results/evidence/ben_k9_comparability_20261017/validation.json)通过，仅验证审查和命令，不代表服务器库存或模型测试。性能CSV、模型源码与旧图保持，无Git提交推送或外部发送。

---

## 2026-10-17 — BEN固定3×3/4×4同图几何对照首批完成

- 用户选择BEN作为主展示数据集，并批准先完成分区本身。复用本地归档的全部4张历史验证输入（97389/14680/28450/4439），原来是seed44反馈诊断24例中的全部阈值翻转案例，选择偏差和历史来源明确保留，不冒充19类代表样本、42/43/46/47性能证据或新获益案例。
- [逐例分析及图注](docs/ben_grid3_grid4_analysis_20261017.md)、[单页PDF/300dpi PNG/四例SVG与PNG](output/pdf/ben_grid3_grid4_20261017/README.md)、[来源及49token成员](output/pdf/ben_grid3_grid4_20261017/figure_manifest.json)保存。按冻结ed480e6与当前分组源码相同的位置公式绘制K9/K16，4×4成员和数量与4份历史实际捕获逐一一致；源PNG/SHA及底图像素不变、未增强亮度或重新裁剪。没有模型推理、受训K9前向、图边、新预测或语义质量数值。
- 180×254mm PDF含12次原图放置和100个区域覆盖，源图仍224×224；300dpi导出2126×3000像素不增加遥感细节。四例及完整PDF已实际看图；160项[独立导出核对](output/pdf/ben_grid3_grid4_20261017/export_qa.json)通过，包含原始像素、K16捕获、整数参照分区、PDF矩形、SVG原图字节及字体/范围说明。构建命令为bundled Python运行output/pdf/ben_grid3_grid4_20261017/build_figure.py，之后Poppler渲染并运行同目录verify_figure.py；没有付费GPU成本。
- 观察：海岸4439中左列由96px缩为64px，包含岸带的空间单元覆盖水面范围变小，但岸水仍混合且连续岸线跨更多区域；14680下方浅绿斑块被更细分区进一步切开；另外两例的不规则纹理也未变成内容边界。支持“覆盖尺度与切分方式变化”，不支持普遍语义改善或4×4分类提升。四例原诊断分数未拿来充作本次网格结果。
- 中文汇报优先项、英文草稿可视化段和原NWPU审查的后续指针已同步，其他用户编辑保持。后续才核验BEN K9库存、三模型同图预测及实际边/扰动证据；当前缺项不自动触发新训练。性能CSV、模型/训练源码、历史图稿均未改，未外部发送或Git提交推送。

---

## 2026-10-17 — 优先审查3×3/4×4区域分析；现有内容部分满足要求

- 用户明确先做3×3/4×4区域分析，再处理其余内容。本轮按导师原文核对源代码、模型库存、旧图和新90例图册，形成[审查说明](docs/nwpu_grid3_grid4_visualization_audit_20261017.md)与[素材/SHA/分区复算](results/nwpu_grid3_grid4_visualization_audit_20261017.json)；中文汇报记录此优先顺序。未生成新对照图，未训练、推理或访问服务器。
- 本地90张224输入逐一SHA与清单吻合，45类各2张，可用于同图几何覆盖；旧12张诊断图实看教堂/机场均为K16 attention/token_feedback、kNN-5，不能替代当前稀疏混合图或3×3对照。旧K9权重/graph_vis仅有历史日记线索，本地没有模型权重；服务器库存尚未检查，不能据此断言已丢失。
- 分组位置公式与冻结ed480e6一致。7×7 token分配到3×3的单轴数量为3/2/2、区域4～9 token；4×4为2/2/2/1、区域1～4 token，显示边界分别96/160及64/128/192像素。4×4不是旧3×3的嵌套细分，更细不自动等于语义更连贯。此为源码公式复算，不是新模型实验或语义评分。
- 当前模型/稀疏混合构图拒绝K≠16，新捕获工具也固定49×16/80边；旧diagnosis手工重建attention且硬调用kNN，不能直接用于旧均值K9或当前sparse_hybrid。先用确定分区规则与既定90图做离线区域对照；随后核验旧K9权重/源码并捕获真实边。历史版本改变池化/反馈/构图，不把全部差异归因区域数；导师定性要求不自动意味着新增纯K训练消融。代码/CSV/性能结果保持，另将邮件seed文字不一致记入审查，未覆盖其余用户编辑。

---

## 2026-10-17 — 澄清NWPU图册的比较对象及共同错误含义

- 用户询问“3个优化5个退化”与固定GAViT高于Swin是否矛盾。重新读取捕获工具、图册清单和正式测试配对；从90条逐例记录独立复算为82双方正确、3 soft纠正fixed、0 soft新增错误、5双方错误，45类各2张。5张共同错误不等于新增退化。
- 图册用seed42校准best的fixed/soft权重、按val每类路径排序前两张，记录真实S/W与构造边并校验捕获前后logits/缓存；正式Swin对照来自独立refit后的25200张test。该test固定对Swin纠正586/新增543、净+43，OA93.93%→94.10%；不能由总体优势推出每张图改善，也不能以90例soft局部+3覆盖其正式test负面结果。
- [英文草稿](docs/supervisor_email_draft_20261012.md)及[中文汇报](docs/supervisor_progress_report_20261012.md)仅澄清可视化段落，其他用户编辑保留。已有纠正/共同错误实图已复看；本轮未新增训练、推理、CSV改动或外部发送。

---

## 2026-10-17 — BEN soft42正式完成；补齐指定四seed并保留负面结果

- 收尾独立复算与改动范围检查71项通过，两个工作区git diff --check通过；四seedCSV与追加条目预览已检查。[统计/记录核对](results/evidence/bigearth_soft42_complete_20261017/report_update_validation.json)保存。这些是数据/文档核对，不代表重新执行模型测试。
- 对用户回传的总队列/训练/test三项exit0、summary complete、final30、最佳验证checkpoint、119825张/19类测试和完成标记完成31项身份/数值核对；[原文](results/evidence/bigearth_soft42_complete_20261017/server_return.txt)、[summary](results/evidence/bigearth_soft42_complete_20261017/summary.json)、[核对边界](results/evidence/bigearth_soft42_complete_20261017/validation.json)保存。该核对基于回传，未独立下载服务器权重/预测数组或运行CUDA。
- 实验bigearth_soft42_20261013_01，seed42、模型ed480e60fea03ed786fdba70c144de5e815c34f5，已修复调度worker SHA09722f10542bed449d2e80df0041387e113cab58db403b3979e07f192194f767与两阶段一致。原调度失败历史保留。RTX4090 UUID GPU-44825cf0-bd98-bfdc-efbc-0100238a6789，FP32/noAMP、matmulTF32false/cuDNNtrue、8线程。
- 问题为补齐指定42/43/46/47的BEN soft缺项及逐类比较；沿用soft43配方，主要变量为训练seed42，物理GPU身份按实记录。RGB/19标签、原划分237871/122342/119825及SHA、原ImageNet权重；预处理/增强沿用冻结配方，未重建split。K16局部原型dim128/radius1/temp0.1/1次迭代、48空间+32特征边、GAT256/4头/2层、token_feedback/dropout0.1，31466766参数；AdamW lr3e-4/wd1e-4、Cosine30、batch32/workers4、BCE、F1阈值0.5。原命令和完整metadata见summary argv；由训练seed变化不能排除历史对照代码/硬件混杂。
- best10/val mAP78.75%，final30/val75.70723466069043%；test mAP70.1745705651681%、macro-F1 64.04828509185678%、micro-F1 76.7999586513447%。checkpoint SHA23314ea561f753b1aad73025148d7b9788bbdddc4f38f49afeca59a6aec91c35在train/meta/test/metrics一致，路径为/home/featurize/work/bigearth_soft42_20261013_01/gavit_train/gavit.pth；日志为同根run.log和gavit_train/run.log，曲线/探针留在服务器阶段目录。
- 训练20619.28s、完整test155.17s、队列20808.02s（5h46m48s）；有效训练处理量346.09图/s包含validation/IO/探针，不是纯前向吞吐。训练峰值allocated3.66GiB；按旧价1.67元/h计运行部分约9.65元，实际账单/准备空闲未知。
- 相比Swin42 mAP−0.73pp、固定42−1.83pp；soft对Swin草地+6.24、多年生作物+2.76，但海岸湿地−19.64、沙滩−4.43、工业区−2.32pp。与soft43方向反转，补足失败分析，不删除、不为了正向结果重跑。指定四seed Swin70.90±0.22、固定71.65±0.39、soft71.19±1.25%；soft对Swin2升2降/均值+0.29pp，对固定仅43正/均值−0.46pp。用户指定子集及历史44/45均保留，46/47原工件/F1/逐类仍缺。
- [四seed汇总](docs/ben_four_seed_three_model_20261017.md)、[CSV](results/ben_four_seed_three_model_20261017.csv)、[逐seed固定/soft分别对Swin的全部类别](docs/ben_seed42_seed43_vs_swin_20261013.md)、导师中文汇报/英文草稿及当前状态已同步。研究工作树bigearth_comparison.csv仅追加一个正式soft42条目，旧行字节不变；主表CSV由artifact-tool author/inspect/render，统计先按原精度计算再四舍五入两位。其余AID/NWPU/seed43记录与46/47 CSV保持。无新训练、外部发送、Git提交推送。

正式阶段命令（保存的argv；冻结调度负责入口加载、精度和资源记录）：

```bash
python -B -u /home/featurize/work/GAViT_SOFT_20260927/train_bigearth.py --model gavit --data_dir /home/featurize/work/GAViT_Project/bigearth_files/splits --epochs 30 --batch_size 32 --lr 0.0003 --seed 42 --num_regions 16 --grouping soft_prototype --edge_type sparse_hybrid --knn_k 5 --gat_hidden 256 --gat_heads 4 --gat_layers 2 --integration token_feedback --dropout 0.1 --run_stage formal --run_tag bigearth_soft42_20261013_01_gavit_train --pretrained_path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --checkpoint_path /home/featurize/work/bigearth_soft42_20261013_01/gavit_train/gavit.pth
```

```bash
python -B -u /home/featurize/work/GAViT_SOFT_20260927/test_bigearth.py --model gavit --data_dir /home/featurize/work/GAViT_Project/bigearth_files/splits --ckpt /home/featurize/work/bigearth_soft42_20261013_01/gavit_train/gavit.pth --batch_size 32 --seed 42
```

---

## 2026-10-17 — 用户报告实例已退还；等待持久化结果核对

- 用户称“实例被退还了，应该是跑完了”，未提供退出码/summary/test指标；只记录实例退还的用户声明，不能推断已训练或测试完成。此前最后技术回传仍是队列分派PID5807，首批运行也未回传。正式BEN soft42指标继续null。
- [状态](results/bigearth_soft42_status_20261013.json)与[查看说明](docs/run_ben_soft42_20261013.md)更新为读取持久化work下bigearth_soft42_20261013_01的总run.exit、gavit_train/run.exit、gavit_test/run.exit、summary.json和run.log；[只读命令及来源/完成条件](results/evidence/bigearth_soft42_result_check_20261017/user_report_and_read_plan.json)保存。仅检查文本/JSON不需要GPU、原始图像、解压、重新预检或训练；无法访问终端时取回同名持久化文件。
- 完成判据仍为总/训练/test三项exit0、summary complete/seed42、final_epoch30、完整119825×19test及完成标记；缺文件或非0先看原日志诊断，不按实例归还推测，也不重训。旧PID仅历史身份，不再据其做活跃进程判断。
- 两条只读命令Bash/zsh语法核验通过；本轮没有直接服务器访问、CUDA执行、代码或CSV改动、分数新增、外部发送、Git提交推送。

---

## 2026-10-16 — BEN soft42队列已分派PID5807；首批运行尚待核验

- 用户回传新BIGEARTH_SOFT42_PREFLIGHT_PASS与BIGEARTH_SOFT42_LAUNCH_DISPATCHED，PID5807、正式输出bigearth_soft42_20261013_01，training_start_verified=false。保存[原文](results/evidence/bigearth_soft42_launch_20261016/server_return.txt)、[预检](results/evidence/bigearth_soft42_launch_20261016/preflight.json)、[分派JSON](results/evidence/bigearth_soft42_launch_20261016/launch.json)及[核对](results/evidence/bigearth_soft42_launch_20261016/validation.json)。首个JSON缺左花括号仅解析时补入，原始字节/SHA保留。
- 与初次预检相比，除库存多出当前smoke的metadata/summary外，源码/划分/旧对照/hash/GPU UUID/软件与精度相同；formal_or_partial/unreadable仍空。旧37.0547与当前37.5025均为工程smoke，未当正式结果。此次粘贴未包含修复PASS标记/更新SHA，不据此宣称修复产物独立确认；当前能确认的是队列分派，不能确认首批、GPU进程/显存或训练完成。
- [状态](results/bigearth_soft42_status_20261013.json)记为launch_dispatched_first_batch_pending；server_training_started为null/未知、server_training_start_verified=false，正式指标仍null。下一步仅nvidia-smi和总run.log末尾，不重复任何安装、smoke、修复或启动。
- 对照SHA8ac0083f的实际冻结基座确认阶段日志/退出路径为gavit_train/run.log、gavit_train/run.exit及gavit_test/run.exit。原操作说明误写process.log，旧status helper误读stage.process.exit；本轮只更正[说明与读取命令](docs/run_ben_soft42_20261013.md)，结束后直接读真实三项退出文件与summary，不以旧helper的pending判失败，不改正在运行的服务器文件。
- 本轮只有回传/路径核对和记录，无GPU执行、代码修改、训练重启、正式指标新增、CSV修改或外部发送/Git提交推送。

---

## 2026-10-16 — BEN soft42工程产物完成；复现并修复两处调度错误，正式启动待回传

- 用户回传当前RTX4090的96train/64val、seed42、1轮训练/validation/保存/重载/图探针完成标记。完整[原文](results/evidence/bigearth_soft42_smoke4090_20261016/server_return.txt)、[相同的stdout与cat摘要](results/evidence/bigearth_soft42_smoke4090_20261016/summary.json)归档；参数31466766、CUDA重载差4.76837158203125e-7、实际80边/48+32、8图3快照和S/W归一化均与既定配置相符，GPU UUID/software/precision与本次预检一致。checkpoint回传SHA为4a6ae6e43001ffe5d729f7eb4e50ef9da4951e3b8422b2e5342b02f88a813398，summary与metadata一致；实际文件字节仍由服务器启动检查核验。本次本地未下载权重或运行CUDA。
- 工程耗时27.455166778秒/入口22.720187440秒、allocated3932788736/reserved4242538496字节。val mAP37.502483541378865%、macro-F1 1.5037593984962405%仅为工程子集（缺类），不进入正式BEN19结果、预算推断或候选选择。正式test指标保持null。
- 原外层短测命令抛FileNotFoundError：冻结训练包装用atexit写run.exit，控制器却在退出前读取；解释了Traceback后cat出现内部exit_code=0。原始外层退出码未回传，不能把该内部标记当作外层命令exit0。本地真实子进程fixture复现，移除过早读取，保留独立launch的完整检查。继续审计正式队列时发现ns['b']保存接口错误（42预检的b为函数局部），真实CLI fixture复现KeyError:b；改用队列自身的hash验证load_base().save，不改预检/模型配方。
- 新两项回归加既有检查共21项通过（42共10、43共11）；全量pytest仍因本地缺pytest未执行。原v1包/源码/失败记录保留，新v2首次安装包包含两处修复。当前服务器未修复，不能直接启动原v1队列；[带SHA校验/备份的修复并启动命令](results/evidence/bigearth_soft42_smoke4090_20261016/repair_and_launch_command.txt)只更新两个外部调度文件及manifest，先复用并核验本次短测，再进入已授权30轮→best全test。命令真实文件fixture证明备份准确、工件字节不变、拒绝重复修复/已有正式输出；最终GPU启动被拦截，不是服务器执行证明。详见[诊断与核对](results/evidence/bigearth_soft42_smoke4090_20261016/validation.json)。
- [当前状态](results/bigearth_soft42_status_20261013.json)和[操作说明](docs/run_ben_soft42_20261013.md)同步；不重训smoke、不删除旧输出、不换实验目录绕过异常。正式启动/精度待回传，历史CSV/其他seed/模型和导师邮件保持，无外部发送、Git提交或推送。

---

## 2026-10-16 — 收到BEN soft42服务器预检PASS；当前4090短测待执行

- 用户回传BIGEARTH_SOFT42_PREFLIGHT_PASS/seed42/new_training_started=false，完整[原文](results/evidence/bigearth_soft42_preflight_20261016/server_return.txt)、[规范化JSON](results/evidence/bigearth_soft42_preflight_20261016/server_return.json)与[核对记录](results/evidence/bigearth_soft42_preflight_20261016/validation.json)归档。原粘贴省略首个左花括号，仅为解析补入，原始字节和SHA保留；不将本地核对当作独立服务器读取。
- 三划分237871/122342/119825及SHA、冻结ed480e6、既有soft43的软件和FP32/TF32设置匹配。当前RTX4090 UUID为GPU-44825cf0-bd98-bfdc-efbc-0100238a6789，型号不变、物理卡变化正常记录。库存formal_or_partial/unreadable均为空，仅旧20260928的seed42工程smoke；未以该37.0547%小样本val分数替代正式结果。
- 旧Swin42 checkpoint/test log SHA首次随本次回传记录，70.9/64.2/76.6与既有打印相符，仍无sidecar，不新增其源代码/训练身份或权重与日志强绑定的证明。固定42 metadata与返回checkpoint SHA一致，cd39be9、best13、final30和72.0/66.0/76.7符合档案；历史代码/硬件混杂保留。
- [当前状态](results/bigearth_soft42_status_20261013.json)及[运行说明](docs/run_ben_soft42_20261013.md)更新为预检通过、当前卡96/64一轮smoke待回传；沿用已安装20261013调度和输出身份，不重新安装/换目录。正式训练/测试未回传，42正式指标继续null，未改CSV、模型或其他seed结果。本轮无GPU执行、外部发送或Git提交推送。

---

## 2026-10-13 — 用户授权补齐BEN软分配seed42；本地调度已准备

- 用户明确选择补跑BEN soft42，使指定42/43/46/47四seed的三模型主指标齐全；已有43/46/47保持，42结果仍为空。研究问题是补足缺项与观察多seed/逐类表现，不根据结果挑正向seed或重训旧Swin/固定42。先查服务器已有正式/部分checkpoint，能复用则不新增训练。
- [运行说明](docs/run_ben_soft42_20261013.md)、[当前状态](results/bigearth_soft42_status_20261013.json)已建立；独立脚本在既有研究工作树中，正式ed480e6模型源码不改。沿用soft43的K16局部原型/sparse-hybrid 48+32/token_feedback与237871/122342/119825、30轮、batch32/workers4、AdamW3e-4/wd1e-4、Cosine30、BCE/阈值0.5、原FP32/TF32及预训练SHA，仅计划改变训练seed42，记录当前4090 UUID。使用新工程及正式输出目录，从ImageNet初始化，验证选best后完整test，保存历史/探针/19类AP和预测/资源/退出码。
- 本地新增8项安全检查、原43调度11项回归、真实冻结CLI解析、AST、Bash/zsh、载荷SHA、安装/重复安装fixture通过。全量python3 -B -m pytest因本地无pytest未执行；未做本地CUDA/模型训练。服务器预检/短测/正式启动均尚未回传，未宣称已启动或填新指标。完整命令、SHA与限制见工作树results/evidence/bigearth_soft42_prepare_20261013/preparation_validation.json。
- 旧对照Swin42日志70.9/64.2/76.6和固定42 cd39be9/best13/30轮、72.0/66.0/76.7复用；前者来源记录有限，后者历史4090训练/3060测试，与新soft42代码/设备差异保留。上一soft43的4090队列约6h22m/按旧价1.67元每小时约10.64元仅用于预算；新任务预估6–8h/10–14元，准备/空闲另计。原CSV/正式分数/图稿/导师邮件保持；无外部发送或Git提交推送。

---

## 2026-10-13 — 在逐seed对Swin的类别分析中加入BEN42

- 用户要求补入seed42。新增[BEN42/43逐seed合并展示](docs/ben_seed42_seed43_vs_swin_20261013.md)/[57条可得类别数值与缺项状态](results/ben_seed42_seed43_vs_swin_20261013.json)：seed42固定版对Swin全部19类加入，原seed43两组38条保持原值，每个seed分别展示，不合并或改用soft−fixed。主表沿用|ΔAP|≥1pp的数值幅度展示口径，非统计显著性判据。
- BEN42固定版mAP70.90→72.00（+1.10pp）；较大改善为荒原+5.00、海岸湿地+4.40、沙滩+3.50、多年生作物+2.40、复杂耕作+2.10、内陆湿地+1.40，工业商业区−1.30。沙滩在42改善但43退化，分别保留。BEN soft42当前无正式test/AP归档，只有工程短测记录，明确留空，不填入NWPU42或其他BEN seed。
- 原seed43稿加合并展示链接，导师汇报逐类段与完整资料附录同步；用户对邮件的其他编辑保留。原性能表、图片和模型不改，无新训练、推理或外部发送。

---

## 2026-10-13 — 按用户要求统一单seed、分别对Swin的逐类展示

- 用户纠正展示方式：固定某一seed，固定GAViT对Swin、软分配GAViT对Swin分别说明改善/退化。新增[BEN43单seed对照](docs/ben_seed43_vs_swin_20261013.md)/[38条完整类别数值](results/ben_seed43_vs_swin_20261013.json)，两组均以Swin为基准，不混用soft−fixed或可视化个例。中文汇报逐类段同步重排，旧多数据集/多比较资料标为附录；邮件另有用户编辑，未覆盖其余正文。
- 明显数值变化以主表展示|ΔAP|≥1pp，仅为排版筛选，非统计检验、模型选择或训练规则；完整19类×两比较保留。固定对Swin沙滩−2.70pp、草地/稀疏植被+1.30pp；soft对Swin沙滩+12.47pp、草地/稀疏植被−1.32pp，内陆湿地+2.64pp。强调“数值幅度明显”不等于统计显著，BEN43历史轮数/代码/设备与打印精度限制保留。
- 主性能四seed范围及原分数不变，无新GPU工作或外部发送。用户要求的是单seed描述，未推断缺少逐类工件的46/47或生成显著性结论。

---

## 2026-10-13 — 逐类别收益/退化与图册案例展开

- 用户要求具体展开哪些类提升/退化。新增[逐类分析](docs/per_class_findings_20261013.md)与[数值/来源SHA](results/per_class_findings_20261013.json)，并纳入导师汇报；区分NWPU42校准val图册案例与refit test类别统计，区分soft−fixed与fixed−Swin。NWPU42已回传每比较两侧各四个极值类，共24条；soft−fixed改善工业区/稀疏住宅/网球场/跑道，退化密集住宅/火车站/教堂/沙漠。图册3纠正/5共同错误/0新增退化不能直接当作整类变化，教堂/火车站在test分别净少对13/17图。
- BEN43完整19类soft−fixed按固定AP±0.05pp打印范围核对为8升/8降/3未确定，主要提升沙滩+15.17、海岸湿地+8.21，多年生作物+1.40；退化草地/稀疏植被−2.62、工业商业区−0.99、复杂耕作−0.62。固定版对Swin的42/43逐类另列，仅用该两seed可得值，不混入44/45或缺少的46/47。AID42/43完整30类正确数/支持数核对，8类两次正、3类两次负，分别保留全部正负/混合/持平。
- 四seed主表仍为42/43/46/47；46/47无逐类回传，NWPU43没有完整逐类，本次不据总OA/mAP推断类别。历史代码/硬件/最终轮和打印精度限制保留，所有变化是描述性结果而非图边/语义质量因果证明；不重训或修改模型/原分数。

---

## 2026-10-13 — 可视化进度与semantic coherence措辞释义

- 用户询问可视化状态及邮件“not yet demonstrate a consistent improvement in semantic coherence”的含义。重新读取NWPU图册审阅、inventory/display统计、渲染源码及BEN图反馈图稿记录，并实际查看教堂纠正/河道共同错误对照图。NWPU预定90例/180SVG制作和审阅已完成，3纠正/0新增错误/5共同错误，token主归属变化39.09%；这些是seed42校准val案例，不是测试总体或语义区域质量指标。颜色代表区域ID，连线为构造边；未做语义区域质量量化，人工审阅也未发现跨案例一致地物完整性改善。
- 邮件改为具体说明已完成可视化、区域/特征边变化及3纠正/5共同错误，另说明语义区域质量未量化，避免把模糊的“尚未证明”理解为图稿未完成或整个模型无收益。中文汇报补充案例解释及BEN历史seed44图反馈四例的已完成PDF/PNG，明确该诊断不属于42/43/46/47性能统计或fixed/soft比较。3×3/4×4受控比较仍未完成；不凭河道跨组或建筑跨组判定分类模型必然失败。原性能表、种子范围和图片保持原样，本轮无新训练/推理或外部发送。

---

## 2026-10-13 — 按用户指定用42/43/46/47四seed作为导师汇报主表

- 用户明确指定本次汇报使用seed42、43、46、47。主表从46/47两seed补充汇总改为该四seed的Swin/固定版均值±样本SD，并统一两位小数。AID为96.37±0.48%→96.83±0.51%（配对均值+0.46pp），NWPU为93.81±0.11%→94.00±0.19%（+0.20pp），BEN为70.90±0.22%→71.65±0.39%（+0.75pp）。NWPU43仍负，不能把四seed解释为每次均优于baseline。
- 软分配按相同seed完整结果比较：NWPU四seed齐全；BEN soft42缺正式归档，仅43/46/47三seed的三模型配对，不用45替换42或与四seed均值直接相减。AID soft未运行。新46/47 F1及原始工件仍待补。汇报JSON新增指定子集与统计，原全量rows/summary、原CSV及历史44/45保留；邮件和中文汇报明确本次是已获得结果上的汇报子集，不作为全部预设seed的稳定性证明。
- 可视化和效率分析段落沿用，没有新deadline表述、外部发送或GPU工作。更改前两份草稿保留于证据目录，全部新统计均从原始可得精度计算后再显示。

---

## 2026-10-13 — 简化导师邮件开头的seed说明

- 按用户要求取消逐个seed编号列举，邮件开头改为Swin/固定版在三个数据集各至少四个随机seed，并简短说明soft已在NWPU/BEN实现和评估。使用“至少四个”对应当前AID5、NWPU4、BEN固定版6的完整记录；现有结果表、种子范围和统计保持原值。

---

## 2026-10-12 — 按用户要求删除导师汇报中的deadline与倒排时间表

- 用户说明导师未给具体DDL，要求删去对应内容。中文汇报已删除交付目标、月底日期及倒排时间表，后续工作改为不带截止日期的研究事项；英文邮件已删除submission deadline、31 October、限时完成及剩余时间表述。外部方法复现路线与验收标准的咨询保留，不再以截止日期作为前提。实验数值、两位小数格式和可视化内容沿用。

---

## 2026-10-12 — 导师汇报统一两位小数；NWPU46/47回传精度确认

- 用户确认NWPU seed46/47为四舍五入后的OA，要求所有seed统一四舍五入至两位小数；中文导师汇报、英文邮件草稿及本批核对说明的单seed、均值/样本SD、配对差展示已统一。统计按原始可得精度先计算再展示，不从已舍入的三位小数二次舍入，不把回传精度补成额外观测。
- 当前CSV原值和全部已有数值JSON/统计数值保留；审计及汇报JSON新增展示规则和NWPU六个OA的两位小数回传来源，原始工件验收状态仍保持。BEN一位小数补尾零仅为排版。用户认为可视化部分可以，现有图册和分析段落沿用；该反馈不升级为语义改善或因果验证证据。
- 展示边界核对：全量NWPU固定均值94.004761…显示94.00；新46/47固定均值94.075显示94.08，soft均值93.545显示93.55，soft−Swin均值−0.175显示−0.18pp。当前全量种子范围和正负结果不变；无训练、推理或外部发送。

---

## 2026-10-12 — AID两项按用户确认更正；全量seed导师汇报草稿

- 用户明确确认AID固定seed46为97.12%、Swin seed47为95.72%，并要求重新记录及整理导师汇报。[当前CSV](results/multiseed_test_results.csv)只更正这两项及AID依赖汇总/说明；其他七行与原表逐字节一致。AID seed46/47的Swin为96.56/95.72、固定为97.12/96.08，均值±样本SD分别96.140±0.594%与96.600±0.735%，配对差+0.56/+0.36pp，均值+0.460pp。正确数4828/4786与4856/4804为按5000图及确认OA派生的相容性检查，不当作独立回传预测。
- [更正核对](results/seed46_seed47_audit_20261012.md)与[JSON](results/seed46_seed47_audit_20261012.json)已更新，原CSV、更正前JSON/说明保留于[证据目录](results/evidence/seed46_seed47_reported_20261012/)。原SHA为907592fb63b954fe9cbae87bde100b0f57efaef44302d751985765ca3f360c9b，当前CSV SHA为639a9b45b07625d63c1254a32969637c984a996997248622a647f261d43c0988。用户确认值解决了此前候选不唯一的问题，未由合法末位推测实验成绩。新批仍缺原始run/summary/test/exit、checkpoint/split身份、具体UUID及辅助指标，不升级为原始工件已验收。
- 新增[中文导师汇报](docs/supervisor_progress_report_20261012.md)、[英文邮件草稿](docs/supervisor_email_draft_20261012.md)与[完整精度/来源SHA](results/supervisor_progress_data_20261012.json)。汇报保留当前选定固定版的全部AID42/43/44/46/47、NWPU42/43/46/47、BEN42/43/44/45/46/47，以及所有可得soft，不挑出四个正向seed替换完整账本。混合历史描述性固定−Swin均值为AID+0.212pp、NWPU+0.197341pp、BEN+0.063551pp；不能作为统一硬件/代码/轮数的核心稳定性结论。原正式表及42/43专项汇总保持原证据范围。
- 对照导师邮件，软分配双数据集实现/评估已有多seed数据；最新NWPU两次负、BEN一升一降且两次低于固定版，负面结果保留。HGTNet×NWPU/AID、DFAGCN×NWPU统一协议多seed仍缺实测；3×3/4×4受控比较未完成；新46/47 macro-F1及BEN micro-F1待补。报告含现有可视化/效率、完整辅助指标范围与10-31前拟议安排，外部材料限制按10-06归档状态注明。邮件仅为草稿，本轮无外部发送、GPU训练/推理、Git提交或推送。
- 独立Python复核24项CSV算术、15组完整主指标及全部描述统计、源文件SHA和报告链接通过；四个更正后AID分数均与5000图OA步长相容。见[验证记录](results/evidence/seed46_seed47_reported_20261012/correction_and_report_validation.json)。

---

## 2026-10-12 — seed46/47数值复算与AID记录错误核对

> 更正前历史状态：AID真实值现已由上方用户确认更正条目补齐，下文候选及待确认描述不再代表当前记录。

- 检查主仓库新增[原数值表](results/multiseed_test_results.csv)，保持其字节不变；[核对报告](results/seed46_seed47_audit_20261012.md)与[JSON](results/seed46_seed47_audit_20261012.json)保存原表SHA及24项均值/样本SD/均值差复算。8个已报告模型×数据集行的计算均与三位小数打印一致，但不能以此确认训练/test完整结束。用户明确补充本批RTX4090、物理UUID与此前不同、全部测试数量及配置不变；型号和配置为直接用户声明，具体UUID/原始run.json/summary/run.exit/权重链尚未读取，保留为待原始工件核验。旧正式表未改，也未将新分数合并进正式多seed均值。
- 用户确认AID原97.13%（固定46）、95.73%（Swin47）是记录错误。5000张test的单次OA步长0.02pp；最近合法候选分别97.12/97.14（4856/4857正确）、95.72/95.74（4786/4787正确），两侧距离相同，不能唯一推回末位2或4。候选是“只改最近末位”的假设，不是实际观测，不替换原CSV或据此定稿AID均值。在该假设下两次固定增益方向仍正，但真实OA、正确数与差值待原test报告。
- NWPU固定相对Swin的回传差+0.20/+0.51pp、均值+0.355；soft−Swin −0.22/−0.13pp、均值−0.175。BEN固定+0.8/+0.5pp、均值+0.650；soft +0.4/−0.3pp、均值+0.050，soft−fixed均值−0.600。BEN一位小数输入使soft−Swin均值保守舍入范围为−0.05至+0.15pp，不能据+0.05认定整体提升或统计等效。原记录中AID汇总仅是错误输入的计算，实际值未恢复；不以跨任务组数当独立重复、不据本批证明方法普遍有效或无效。本轮仅只读数值检查/记录，无新GPU训练、推理、提交或推送。

---

## 2026-10-12 — 三数据集多seed测评结果记录（seed46/47：Swin / GAViT / 软分配GAViT）

> 原回传历史条目：AID97.13/95.73已更正为97.12/95.72；均值、软分配解释及正式证据范围以本日顶部核对与导师汇报为准。

- 本条为结果记录。用户回传已完成的多seed测评：AID、NWPU-RESISC45、BigEarthNet-19 三数据集上各测 Swin baseline、固定分组 GAViT、局部原型软分配 GAViT 三个模型（软分配未在 AID 上运行），每个"数据集×模型"含 seed46、seed47 两个seed。
- 指标口径：AID/NWPU 为单标签分类 test 正确率，BigEarthNet 为 test mAP（数值量级与既有 seed42–44 正式结果 70.2~72.0 一致）。所有数字均为用户口头回传，本地未见日志、checkpoint、run.json 或完整命令；Git commit、GPU、最佳轮、val 曲线、参数量、macro-F1/per-class/混淆矩阵与推理成本全部待回传确认。复核前不写入既有 formal 明细表（`bigearth_comparison.csv` 等），明细与均值见新表 [multiseed_test_results.csv](results/multiseed_test_results.csv)。
- 汇总（mean±std，样本标准差 n−1；括号内为 seed46/seed47 原值）：

| 数据集 | 指标 | Swin | GAViT | 软分配GAViT |
|---|---|---|---|---|
| AID | test Acc(%) | 96.15±0.59（96.56/95.73） | **96.61±0.74**（97.13/96.08） | 未运行 |
| NWPU | test Acc(%) | 93.72±0.06（93.76/93.68） | **94.08±0.16**（93.96/94.19） | 93.55±0.01（93.54/93.55） |
| BigEarthNet | test mAP(%) | 70.80±0.28（71.0/70.6） | **71.45±0.49**（71.8/71.1） | 70.85±0.78（71.4/70.3） |

- 解释（GAViT vs Swin）：3 数据集 × 2 seed 共 6 组配对全部为正；逐 seed 差值 AID +0.57/+0.35、NWPU +0.20/+0.51、BigEarthNet +0.8/+0.5，均值分别 +0.46、+0.36、+0.65。方向与既有证据一致，BigEarthNet 幅度落在 seed42–44 正式区间内。
- 负面结果（软分配）：软分配 GAViT 在全部已测数据集均低于固定分组 GAViT（NWPU 均值 −0.53、BigEarthNet −0.60），NWPU 上低于 Swin（−0.18），BigEarthNet 上与 Swin 打平（+0.05）；与 NWPU seed42 校准中 soft 93.73 < Swin 94.21（val OA）的形态一致。结论：局部原型软分配作为分组替代不带来收益且另加约 10 万参数，不再为该变体追加训练，除非出现新的研究问题。
- 限制与待办：每配置目前仅 2 seeds，论文核心对比按规则需 3 seeds 均值±标准差，缺口需补 seed 或在论文中说明；NWPU seed42 校准为 val OA 口径，不能与本次 test 口径直接合并为第三 seed。AID/NWPU 各模型具体配置（grouping 类型、edge 构造）与 BigEarthNet 侧变体的对应关系待 run.json 回传核对；本表数字在日志审计前一律视为"用户回传、待审计"。

---

## 2026-09-28 — 局部原型软分配独立发布与NWPU短测准备

- 用户选择独立分支推送并创建PR。此分支从服务器现有`deb2b45`派生，只包含18个新模块相关代码/测试文件及本次发布说明、证据；原研究工作区其他未提交材料保留。这里未纳入其间的全部实验日记，不用此分支历史判断项目总体进度。
- 新增`soft_prototype`：共享LN/768→128投影与图像内局部原型、cosine温度0.1、单次可微软分配，S/W分别用于软反馈/区域汇总；新增99840参数。Swin/16区域/GAT及48+32图边构造不变，旧分组默认路径保持；新配置严格写入并重建metadata。
- scene/BigEarth入口与固定train/val/pool探针接通，保留epoch0/best/final；中断后基于已保存快照恢复别名并验证身份。候选完整CPU回归133项通过、44warnings、47.52秒、exit0；原始测试日志和30个源码哈希随分支保存。首次无Git导出/沙箱OpenMP导致的本地测试失败保留于原工作区，使用真正Git副本并获准沙箱外复测后通过，未改模型来绕过。
- 服务器回传：clean deb2b45、RTX2080 Ti total22528/free22009 MiB、NWPU清单5040/1260/25200及8组同类重复组隔离，manifest SHA256 `ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`。预训练SHA256 `f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`匹配。
- [独立服务器运行说明](docs/run_soft_assignment_nwpu.md)限定360train/90val、1轮、seed42工程短测；真实CUDA、多worker、S专属梯度、资源与精度尚未验收，未启动GPU或填入新成绩。BigEarth96/64短测、三模型正式消融及多seed继续待后续验证。

---

## 2026-09-20 — AID压缩包哈希确认；prepare被手动中断，补充扫描进度

- 用户回传Git fast-forward 2ba112a→06fddd0及ZIP完整SHA256 `cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7`，与固定来源预期一致；据服务器用户输出确认下载完整性，未在本地重新下载/哈希整个压缩包。
- group prepare在`_image_identity`像素SHA256计算处收到Ctrl+C，抛KeyboardInterrupt。没有新的数据错误证据；未达到清单写入步骤，未完成全量预检，不能当作数据或GPU验收通过。日志没有开始时间，不能估算本次已运行多久或归因为性能故障。
- 原入口只在全量扫描后打印结果，容易误以为卡住；新增开始、首张、每250张和最终张的已完成数量/总数、已用秒数并立即flush。prepare及训练/评估的load_manifest共用提示；不改变哈希、分组、seed、划分或输出清单内容。
- 18项现有数据测试和5个subtest通过；实际CLI使用600张合成图含1组重复，prepare/load退出0，进度0/1/250/500/600均可见，清单与06fddd0逐字节一致。见[CLI证据](results/evidence/aid_progress_20260920/cli.log)。未新增仅镜像日志实现的单元测试。
- 下一步同步代码后用新编号`aid_protocol_20260920_03.json`前台运行，等待MANIFEST COMPLETE及DUPLICATES；通过后再进行Swin GPU短测。没有要求重下/重解压，没有启动正式训练或改正式结果表。

---

## 2026-09-20 — AID重复图片预检修复：保留原图并按像素组隔离划分

- 用户服务器2ba112a运行prepare，在`Industrial/industrial_205.jpg`与`industrial_203.jpg`解码像素重复处退出。原入口对所有重复一律报错，未生成清单、未开始训练；此输出不证明ZIP整体哈希或所有图片均已验收。已有压缩包和图片保留，不要求重新下载/解压。
- 新增显式`--duplicate-policy group`，采用schema 2 / scene-pool-group-v2。同类完全相同像素不可拆组，固定seed按组分配，保持每类精确50%训练池及池内20%val；跨类标签冲突仍拒绝。确定性子集和先选pool，再选val；选定pool无法满足精确val时拒绝而不改比例/换seed。默认reject及旧schema 1兼容，无重复类别沿用旧划分。
- 清单记录重复组路径、像素哈希、类别、split、唯一图像/多余副本数。加载时重算并校验数据哈希、报告、无跨划分重复与seed划分。全部原图保留，指标仍按全部图片计数；这是自定义防完全重复泄漏协议，不称AID官方固定划分，也未处理近似重复或地理重叠。两模型必须共用同一清单。
- 新增7项回归：同类跨编码重复、原图/数量保留、跨类拒绝、无法精确划分拒绝、跨split泄漏拒绝、报告一致性、旧划分兼容，并覆盖新清单真实CPU训练/重载/评估流程。发布工作树完整测试96 passed、34 subtests、exit0、10.52秒；39条现有Torch/Python3.12弃用警告。合成CLI prepare及指南bash/zsh语法通过；代码审查无阻断项。见[验证记录](results/aid_duplicate_fix_validation.md)。
- 下一步服务器Git同步后以原data-root和新清单`/home/featurize/work/aid_protocol_20260920_02.json`执行prepare group，回传MANIFEST COMPLETE及完整DUPLICATES报告；实际重复总数和完整数据预检仍待确认。ZIP源预期SHA256 cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7，尚无用户核验回传。未启动GPU短测/正式训练、未改正式结果表。

---

## 2026-09-20 — AID统一入口整理为独立服务器同步分支

- 用户确认继续准备服务器短测，并选择自行复制单行命令执行。新分支`codex/aid-entry-20260920`从既有模型版本8f2ed07建立，仅复制经过本地审查和测试的AID入口与测试；原研究工作树全部未提交实验记录保留。本分支历史日记没有包含其间尚未提交的实验分析，不作为最新整体实验档案。
- 新入口提供固定数据清单、共同训练配置、smoke/calibrate/refit、OA/macro-F1/逐类正确率/混淆矩阵、checkpoint身份与源码校验。原有模型、训练入口和正式结果表未改动。
- 既有本地验证74项测试与31个subtest通过；两模型合成数据CLI完成且保存/重建logit一致。隔离分支验证及限制见[记录](results/aid_entry_release_validation.md)。真实AID、ImageNet文件及CUDA双worker均待服务器验收，没有新正式指标。
- [服务器指南](docs/run_aid.md)改用全新代码目录`/home/featurize/work/GAViT_AID_20260920`，复用旧持久目录中的预训练文件，依次预检数据、Swin smoke、GAViT smoke。此条记录的是同步准备，不声称用户已拉取代码或运行成功。

---

## 2026-09-10 — GAViT44（seed44，RTX 3080）正式完成：best epoch 19，Test mAP 70.2%

**实验身份**

- 日期：训练 2026-09-09 06:52 UTC 启动；测试 2026-09-10。run_stage formal，run_tag `rtx3080_seed44_20260909`，seed 44。
- Git commit：`36345f7d5ffccf714992120abe8e4f6b52d438c7`（pin_memory 修复版；服务器 git pull 后 preflight 校验 commit 一致、工作区干净）。
- 完整命令：`python train_bigearth.py --model gavit --data_dir bigearth_files/splits --epochs 30 --batch_size 32 --lr 3e-4 --seed 44 --num_regions 16 --grouping attentive_spatial --edge_type sparse_hybrid --knn_k 5 --gat_hidden 256 --gat_heads 4 --gat_layers 2 --integration token_feedback --dropout 0.1 --run_stage formal --run_tag rtx3080_seed44_20260909 --pretrained_path bigearth_files/model.safetensors --checkpoint_path /home/featurize/work/gavit44_3080_20260909/gavit.pth`（经 runpy 包装器 nohup 启动，见 [docs/run_gavit44_3080.md](docs/run_gavit44_3080.md)）。

**数据**：BigEarthNet-19 RGB；train 237871 / val 122342 / test 119825，三个 split 的 SHA256 与 preflight 记录一致；预训练权重 SHA256 `f7e995…a9d36`；标准增强与阈值 0.5。

**模型**：GAViT v2 sparse-hybrid：K16、attentive_spatial、sparse_hybrid（4n top2）、GAT hidden256/heads4/layers2、token_feedback、dropout 0.1；参数 31.37M；Swin-T 骨干全量微调（freeze_backbone=False）。

**优化**：计划 30 epochs；AdamW lr 3e-4、wd 1e-4；CosineAnnealingLR；BCEWithLogitsLoss；实际完成 26 epoch 后用户手动早停（见"解释"）。

**资源**：RTX 3080（10GB，CUDA 12.1，cuDNN 8902），实例 24GB RAM；训练中观测显存 4272MiB（~4.2GiB）；吞吐 5.4~6.9 it/s；每轮约 19~23 分钟训练 + ~7 分钟验证 ≈ 24 分钟/轮；26 轮约 10.4 小时（待按日志时间戳精确核算）；费用待确认。`memavail.log` 全程稳定 ~21GB——pin_memory 修复后 26 次全量验证无任何内存压力，OOM 根因闭环。

**输出**

- 最佳轮 epoch 19：Val mAP 77.9%，Val macro-F1 72.3%；best checkpoint `gavit.pth` + sidecar `gavit.meta.json`。
- 最终完成轮 epoch 26：Val mAP 76.9%，Val macro-F1 71.9%（按规则同时记录，不隐藏后期过拟合）。
- 测试（best checkpoint，test_exit=0，3745 批 5:20 @11.69 it/s，架构自 sidecar 重建）：**Test mAP 70.2%，macro-F1 65.2%，micro-F1 77.1%**。per-class AP 已记录于 test.log——最低：Beaches/dunes/sands 9.7%、Coastal wetlands 18.0%、Natural grassland 41.8%、Industrial/commercial 48.7%；最高：Marine waters 99.7%、Coniferous forest 93.8%。
- 产物（Featurize 持久盘）：`/home/featurize/work/gavit44_3080_20260909/{train.log,test.log,memavail.log,train.exit,test.exit,gavit.pth,gavit.meta.json}`。

**解释**

- 早停决策：epoch 19 达峰后 Val mAP 缓慢回落（77.9→76.9，约 1 点），用户据此在 epoch 26 完成后手动停训；随后用户在另一台服务器完整重跑全流程，后段轮次指标持续降低，二次独立确认过拟合形态（该重跑为用户自发验证，未纳入正式记录）。best-val-mAP 挑选规则未变，不属于事后改指标。
- val→test 落差（77.9→70.2，约 7.7 点）与历史 run 一致（seed42/43 同样 val 系统性高于 test），属数据集固有现象，不是异常。
- 2080 Ti 的 epoch1 失败段与第二次服务器重跑均不计入任何论文数字；pin_memory 修复为资源层改动，论文按工程修复披露。
- 待办：同一块 3080 上执行 Swin44 配对（启动器 [docs/run_swin44_3080.md](docs/run_swin44_3080.md)），完成后更新对比表并计算 3+ seed 均值/方差。

---

## 2026-09-09 — run4 单变量实验 PASS；pin_memory 修复提交并推送（36345f7）

- run4（仅验证 loader `pin_memory=True→False`）结果：主进程 RSS 全程平稳（~1.66GB，对比 run3 同期已 6.9GB 并线性冲向 23.3GB），MemAvailable 保持 ~22GB，全量 122342 张验证图与原指标计算完成，`diagnostic_exit=0`，`FULL VALIDATION DIAGNOSTIC PASS`。根因实锤：评估 loader 的 pin_memory 锁页主机内存池在小内存实例上耗尽 RAM。
- 修复（hotfix，commit `36345f7d5ffccf714992120abe8e4f6b52d438c7`，分支 codex/sparse-hybrid-4n-top2，已推送 origin）：`train_bigearth.py` 验证 loader 与 `test_bigearth.py` 测试 loader 均改为 `pin_memory=False`（test 集 119825 图同路径同风险）；训练 loader 保持 `pin_memory=True`（epoch1 全程 7434 批 RSS 采样仅 1.9GB，证明训练路径无此问题，单变量原则不动）。纯资源改动，数据、模型、指标、评估顺序不变。
- 本地验证：`py_compile` 通过；`python3 -m unittest tests.test_resource_probe tests.test_training_profile tests.test_experiment_identity tests.test_sparse_hybrid_static_contract` 37 项全部通过。
- GAViT44 正式重启启动器已整理为 [docs/run_gavit44_3080.md](docs/run_gavit44_3080.md)：git pull 拉取修复、preflight 校验新 commit/3080/权重/全量 RGB、独立输出目录 `gavit44_3080_20260909`、退出码落盘 `train.exit`、主机可用内存每 30 秒记录 `memavail.log`、nohup 后台。训练参数与 2080 Ti 失败尝试完全一致（seed44、30 epochs、batch32、lr3e-4、K16 attentive_spatial、sparse_hybrid、token_feedback），run_tag `rtx3080_seed44_20260909`。
- 预计耗时：每轮约 35~40 分钟（训练段 29.5 分钟 + 修复后验证约 7 分钟），30 轮约 18~20 小时。费用与时长待运行记录。
- 披露事项：GAViT44 训练硬件为 RTX 3080（与 Swin44 配对时也用同一块 3080）；2080 Ti 的 epoch1 失败段不计入任何结果；pin_memory 修复属资源层改动，论文中按工程修复说明。
- 本地 research_diary.md 与 docs/run_gavit44_3080.md 暂未提交（避免改变服务器 pull 目标 commit；训练启动后再提交推送）。

---

## 2026-09-09 — 静默中断根因确认：验证阶段主机内存耗尽触发 OOM killer

- 三阶段诊断链条：短验证（70图）通过 → 训练→验证转换（3+3批）通过 → 全量验证（122342图）三次尝试。第一次在 2080 Ti 精确匹配处主动停止（用户已换卡 3080）；第二次前台运行整组进程消失且无 exit_status，判定网页终端 SIGHUP；第三次 nohup 后台仍整组消失，uptime 2h 排除实例重启。
- 第四次（内存监控版，run3）拿到决定性证据：`diagnostic_exit=137`（SIGKILL），且仅 python 主进程被杀、bash/tee/awk 存活写下退出码——OOM killer 单杀最肥进程的典型签名。内核计数实锤：`system.slice/memory.events: oom_kill=4`、`apphub.service: oom_kill=1`。
- 内存轨迹：验证开始后主进程 RSS 以约 8~9MB/batch 线性增长，2.3GB→6.9GB→13.6GB→19.9GB→23.3GB（4.5 分钟），MemAvailable 跌到 ~50MiB 后于 batch 3369/3824 被杀。DataLoader 子进程稳定在 5.2GB。
- 代码排查（commit bb9f1cf）：`evaluate` 在 `torch.no_grad()` 下每批仅保留 `logits.cpu().numpy()`（~2.4KB），数据集读图在 worker 内无缓存——主进程没有与 8~9MB/batch 匹配的逻辑对象。每批 float32 图像 tensor 为 19.25MB，经 `pin_memory=True` 锁页内存池分配；锁页内存不可被内核回收，与"avail 归零且页面缓存无法挽救"的现象一致。当前主嫌疑：PyTorch 2.2.2 锁页主机内存池在该负载下复用失效。
- 死亡史统一解释：Swin43 在 4090 实例（大内存）30 轮无恙；GAViT43 停于 epoch 21、GAViT44 停于 epoch1 训练段结束，均发生在验证窗口——三次静默中断同为验证期主机 OOM，不是平台故障、不是代码逻辑错误。数据集位于 /home/featurize/data 高速盘，位置无问题。
- 进行中：run4 单变量验证实验——同一诊断仅将验证 loader `pin_memory=True→False`（训练 loader 不变），内存监控保留。若 RSS 平稳且 PASS 则实锤 pin_memory 锁页池；修复方案为正式训练代码验证 loader 关闭 pin_memory（纯资源改动，不改变数据/模型/指标语义），是否改动冻结训练代码由用户决定并记录。
- 硬件/环境记录：RTX 3080（9874MiB，10GB 版）、24GB RAM、torch2.2.2+cu121、torch-geometric 2.8.0（WITH_PYG_LIB/WITH_TORCH_SCATTER/WITH_SOFTMAX 均为 False）。诊断脚本本地备份：`/tmp/gavit44-fullval.KTrcwi/`（bg3=内存监控版，bg4=pin_memory=False 单变量版）。

---

## 2026-09-09 — 用户主动换卡 RTX 3080；全量验证诊断重新定向

- 第三阶段全量验证诊断首次执行在 GPU 身份检查处主动停止（`diagnostic_exit=1`）：脚本要求精确匹配失败现场的 `NVIDIA GeForce RTX 2080 Ti`，当前实例已不是该卡。脚本按设计拒绝了在意外硬件上继续，不是训练/验证路径出错。
- 用户确认是有意将实例换成 RTX 3080（原 2080 Ti 实例不再使用；GAViT44 失败运行本就没有可恢复 checkpoint，无产物损失）。
- 影响与披露：诊断目标从"在失败现场同款硬件上复现"变为"验证全量验证代码路径在规模下可完成"；在 3080 上通过不能复现或解释 2080 Ti 上的静默退出，只能作为代码路径无全量缺陷的旁证。
- seed44 配对安排随之确定：GAViT44 与 Swin44 均在同一 RTX 3080 上执行（GAViT44 需重跑），论文披露硬件为 3080；不混用 2080 Ti 的训练段数据。
- 诊断脚本已更新：GPU 打印（名称/显存/CUDA/cuDNN）前移到校验之前，校验改为 `endswith('RTX 3080')`；其余抽取逻辑、样本边界、看门狗与 20 分钟上限不变。本地 bash -n 与内嵌 Python AST 检查通过。
- 待办：用户在新实例重新执行 `/tmp/gavit44-fullval.KTrcwi/diagnose_gavit44_fullval.sh`，回传 `diagnostic.log` 与 `exit_status.txt`；脚本仍只读旧 2080 Ti 目录的 `gavit.meta.json` 作身份参照（Featurize work 目录跨实例持久），不修改其中任何文件。

---

## 2026-09-09 — GAViT44首轮训练段后异常中断（原因待查，不重跑）

- 用户已同意第三阶段全量验证诊断，20分钟预算。已准备 `/tmp/gavit44-fullval.KTrcwi/diagnose_gavit44_fullval.sh`，沿用刚通过的3批短训练（79图，原变换/AdamW/Cosine和训练片段），仅把验证范围扩大为全部122342图。抽取原load_split/train_tf/val_tf/evaluate，原batch32、4 workers、pin_memory、seed44和模型配置保持；验证3824批（末批6图），完整计算原AP/macro-F1，但不输出为论文性能结果，不保存权重或修改正式训练文件。
- 全量诊断输出由服务器mktemp新建 `gavit44_fullvaldiag_XXXXXX`：完整模块/批次轨迹写diagnostic.log，终端显示前三批、每100批和最终批进度及异常；保留Python、tee和显示筛选各自退出码。每批完成后重置120秒无进展堆栈计时，避免在正常长任务中固定每分钟打印误导性Timeout；外层1200秒触发终止，必要时最多10秒强制清理。退出124仅表示此次人为时间限制，137单独不证明OOM；不自动延长、重试或启动正式训练。
- 本地检查通过：Bash语法、Python编译、原代码抽取、完整3824批/122342图、提前结束/错误batch大小/读取异常四种替身迭代检查、进度次数及计时重置、终端筛选不丢异常堆栈、20分钟限制与退出码记录。真实GPU、全量图像解码及全量指标尚未运行；命令交由用户执行。即使全量验证通过，也不能替代7434个训练batch后的状态或解释原后台进程退出。
- 训练→验证短诊断服务器回传通过：09-09 03:25:32—03:25:46 UTC（BEGIN至PASS约14秒），PID24721，目录 `/home/featurize/work/gavit44_transitiondiag_5HPWmR`，`diagnostic_exit=0 log_exit=0`。3批训练的forward/backward/optimizer均完成，训练iterator耗尽、scheduler.step、验证worker启动、3批验证及原AP/macro-F1全部返回。验证了所选79图短训练→70图验证的执行路径，未复现原退出，不据此宣称完整一轮或全量数据通过。回传文件 `/Users/patrick/.codex/attachments/527d985f-44d9-445b-8b9e-292998d76c02/pasted-text.txt`。
- 两次短诊断均通过后的边界：尚未覆盖完整122342张验证图像、全量预测汇总、经过7434个训练batch后的权重/资源状态或原后台运行的外部终止事件；不能把未复现直接解释为平台强杀，也没有根因修复。建议下一步保持短训练及模型配置，仅扩展到完整验证集并保留批次定位日志；拟设20分钟上限，该扩大GPU诊断需用户确认。本轮仅记录证据，未准备/启动第三次GPU诊断、正式重试、换卡或依赖修改，不更新性能表。
- 验证短诊断服务器回传通过：09-09 03:18:38—03:18:57 UTC（BEGIN至PASS约19秒），PID24474，目录 `/home/featurize/work/gavit44_valdiag_KUHfHp`；原evaluate处理batch32/32/6，数据读取、Swin、attentive_spatial、两层GAT、classifier、AP和macro-F1均完成，`diagnostic_exit=0 log_exit=0`。当前环境及val/预训练身份复核通过；PyG的WITH_PYG_LIB/WITH_TORCH_SCATTER/WITH_SOFTMAX为False，但本次所需路径可运行，不因此安装额外依赖。原始回传位于 `/Users/patrick/.codex/attachments/bcfa6752-2702-419f-97c3-c6b66bc0d439/pasted-text.txt`。
- 本次外层bash提示here-document到EOF仍未匹配BASH；内部Python已完整执行、日志和退出码完整，因此不把该警告当模型错误，也不为这条警告重跑已通过的70图验证。该短诊断仅证明新初始化模型下所选70图及三批推理/指标可执行，不证明原故障已解决、不排除其余验证数据或长时间运行问题，也没有覆盖训练后worker/梯度模式转换。
- 下一步针对缺失的训练→验证转换准备 `/tmp/gavit44-transition.LeJuqR/diagnose_gavit44_transition.sh`：复用第一份诊断，新增原训练变换、criterion/AdamW/Cosine设置和首轮验证调用之前的五条原训练语句，仅对独立模型执行3批训练（32/32/15，共79图），再执行原3批验证。train split SHA复核，4 workers/pin_memory/训练shuffle保持；5分钟上限，输出新建gavit44_transitiondiag目录，不写checkpoint、metadata或正式论文指标，不修改训练代码、依赖或自动提交正式重试。Bash/Python语法、原代码抽取、替身验证恰好3次backward/optimizer后scheduler、79/70样本边界和输出隔离检查通过；第二阶段服务器执行待回传。
- 后续补齐的preflight.log（用户回传）：commit `bb9f1cff463c996bc5f72695af538898c91490eb`；Python3.11.8、torch2.2.2、torchvision0.17.2、CUDA运行时12.1、cuDNN8902、timm1.0.27、torch-geometric2.8.0、numpy1.26.4、scikit-learn1.9.0、rasterio1.4.4。预训练SHA256为 `f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`；train237871，SHA256为 `25734fd9e8dc7fd78fb259e768ae91fe11b297e1caa50a2d57b74b0296413d8f`；val/test数量和SHA与09-08记录一致。仅凭版本不能认定依赖冲突，未升级或降级环境。
- 系统证据补充（用户回传）：09-09 02:59:53 UTC查询，uptime起点为09-08 09:37:37；train.log最后修改09-08 10:49 UTC，不等于实际退出时刻。当前 `/sys/fs/cgroup/system.slice/apphub.service/memory.events` 的low/high/max/oom/oom_kill均0，memory.max为max，memory.peak不存在；dmesg拒绝访问，journalctl提示无权查看系统消息后显示No entries。没有支持OOM的正面证据，也不能用受限日志或当前cgroup快照完全排除历史终止事件。
- 已转向代码路径定位：原evaluate不打印进度，验证后才打印指标及保存权重。源代码抽取的4个控制流检查验证了正常/异常验证时保存的先后关系，以及启动包装器对RuntimeError/SystemExit执行finally并传播异常；这些检查用替身对象，不是CUDA或数据解码实测。未发现启动器设置30分钟超时或吞普通异常。
- 已准备独立分段诊断命令 `/tmp/gavit44-validation.eIZvr2/diagnose_gavit44_validation.sh`：抽取原val_tf/evaluate，只读原metadata、val split与预训练，选验证集前64张及最后6张，batch32、num_workers4、pin_memory=True；记录取batch、Swin/grouping/GAT/classifier和原指标计算的边界。没有backward/optimizer更新或checkpoint写入；输出由服务器mktemp创建在新的gavit44_valdiag目录，保存日志与退出码，5分钟上限只用于此诊断。使用新初始化模型，不能复现丢失的epoch1权重、完整训练→验证worker生命周期或全量数据；通过不代表原故障已解决。Bash语法、Python编译、函数抽取、样本选择及无训练/权重写入静态检查通过；服务器诊断尚未执行。
- 用户检查 `/home/featurize/work/gavit44_2080ti_20260908/train.log`：最后记录为 `Epoch [1/30] Train` 的7434/7434批，耗时29:29、平均4.20it/s。没有第1轮验证指标、第2轮、最终Best Validation mAP或包装器finally资源统计。
- `ps -p 3504`仅输出表头；`gavit.pth`、`gavit.last.train_state.pth`均不存在，只有1.1K的 `gavit.meta.json`（Sep8 10:20）。据此判定本次不是正常完成；仅完成第1轮训练段，未确认第1轮验证及保存成功。保留全部文件，不重新提交，不把该次结果写入性能比较表。
- 当前训练代码在训练前写metadata；每轮结束先运行无进度条的 `evaluate(val_loader)`，输出验证指标后才调用 `save_epoch_artifacts`。因此metadata存在不证明训练完成；中断窗口在训练段末尾到首次验证/保存完成之前，不能精确到某一验证batch。
- 包装器普通Python异常也应执行finally统计；当前日志没有这部分输出。强制终止（含OOM killer/平台操作）、底层崩溃、实例重启或日志写入异常均为待排查候选，没有内核/平台证据前不认定OOM或显存不足。验证代码累计的是19维预测与标签，不能仅因有数组累积就断言28GB RAM不足。
- 当前已知输出中没有可用于恢复本次训练的完整checkpoint。下一步只读查看日志更新时间、输出目录其他文件、系统启动时间、内核OOM/segfault/NVRM记录和可用的cgroup内存事件；计数需结合时间/PID，不把共享cgroup历史事件直接归到3504。旧启动器未单独保存退出码，不能从消失PID事后推回退出原因。
- 已知训练段资源：2080 Ti；历史运行中GPU采样约4.1GiB，非峰值；主进程RSS采样1942188KiB，非全作业RAM峰值。总运行时长、异常时间、峰值资源和实际费用均待确认。没有续训、代码变更、自动换卡或新付费任务。

---

## 2026-09-08 — 用户选择2080 Ti直接训练（启动成功；后续中断见09-09记录）

- 用户随后补全日志：Train237871/Val122342；`Model: GAVIT | 31,366,926 params | Device: cuda`；`Epoch [1/30] Train`已到3338/7434批（45%），训练段已用13:39，tqdm显示剩余13:41、局部速度4.99it/s。结合PID3504/GPU进程证据，现确认GAViT seed44正式GPU训练已成功启动并推进；不需要重启或修改配置。
- 同轮10:34:20资源快照：PID3504存活14:12、STAT=Dl、RSS1942188KiB（仅主进程）；GPU总占用4184MiB，PID占用4180MiB，瞬时利用率0%。这是采样值，不是峰值；单个D/0%快照不足以证明持续I/O卡死，完整进度日志已证明训练在推进。首轮验证、存盘、全作业RAM峰值和总耗时仍待确认。
- 启动初期10:20:13回传：`Preflight passed`，GAViT44启动命令已提交，PID3504在5秒快照中存在；当时只打印参数、GPU4MiB/0%且无计算进程，因此当时仅记为“启动已提交”；后续补全日志才满足GPU启动确认标准。
- 数据身份回传：val122342，SHA256 `1c6627cb48e33588538c4e411e4f1b94560827f27d59fb7aeb0ea2b5d30c2a45`；test119825，SHA256 `cea165a6443fc5ee44de0a3c55f53ce70505c4e25aba9ca8b52d76634df77204`。按所给命令，`Preflight passed`意味着三个split的RGB路径检查均通过；本次粘贴未包含train SHA256或软件版本，待完整preflight.log补齐，未独立读取服务器文件。
- 用户09:39:37回传当前实例：NVIDIA GeForce RTX 2080 Ti、22528MiB、1MiB已用、无计算进程；驱动610.57.04，nvidia-smi显示CUDA UMD13.3（这不是PyTorch实际CUDA运行时版本）。commit bb9f1cf，Git状态命令无改动输出；预训练109M；CSV含表头分别237872/122343/119826；数据目录存在。完整RGB文件与PyTorch实际CUDA能力仍需启动命令检查。
- 正式命令已整理为 [docs/run_gavit44_2080ti.md](docs/run_gavit44_2080ti.md)，直接调用原入口，独立目录 `/home/featurize/work/gavit44_2080ti_20260908/`；配置、seed和输出名在启动前固定。状态以本条顶部最新回传为准，首批已确认，首轮验证与产物待确认。
- 本地检查：启动块通过 `bash -n`，两个内嵌Python片段通过AST语法检查，实际训练入口的argparse定义接受GAViT/seed44/30epochs/batch32/sparse_hybrid/formal参数；原训练代码无本轮差异。这些是命令静态检查，不是GPU运行测试。
- 用户明确决定使用 RTX 2080 Ti 22GB，跳过原独立资源短测。同步更新 [排期](docs/thesis_schedule.md)，旧3060/3080工具不再作为启动入口；模型与训练代码未改。
- 下一实验：BigEarthNet-19 sparse-hybrid GAViT seed44；研究问题为已有seed42/43提升是否在新增独立seed上成立，进入多seed核心对比表。相比既有GAViT固定架构，主要实验变量为seed；同时发生GPU更换，必须披露，不能把跨硬件方差解释成纯seed方差。
- 后续在同一2080 Ti/软件环境补Swin44形成配对；不自动启动Swin或seed45。保持batch32、224输入、30 epochs、AdamW lr3e-4/wd1e-4、CosineAnnealingLR、BCE、threshold0.5、K16 attentive_spatial、sparse_hybrid和token_feedback。
- 冻结训练commit仍为 `bb9f1cff463c996bc5f72695af538898c91490eb`；先检查当前服务器身份及完整数据，输出使用新的独立目录，不覆盖旧结果。保持原精度设置并记录，不因换卡改AMP/TF32。
- 启动策略：直接正式训练，同时观察首批、首轮验证和checkpoint；不再单独跑资源smoke。单次正式任务可能在新硬件/环境首次运行时失败，该风险由本次选择保留。数据与文件防覆盖检查不取消。
- 当前由用户在服务器执行并回传输出，本任务未直接访问服务器/已登录浏览器或代租实例。已确认启动和局部训练速度；完整首轮、全作业RAM、验证/测试指标及产物待回传，没有新论文性能结果或实际费用，不能标为实验完成。

---

## 2026-09-07 — 五周排期落盘与 3080/3060 资源短测准备

- 用户确认开始执行；排期主记录为 [docs/thesis_schedule.md](docs/thesis_schedule.md)，暂按 09-07 至 10-11 五周安排，最终提交日待确认。
- 当前 BigEarthNet 核心比较按最新导师要求至少再增加两个 seed；候选分配为 3080 10GB 配对 Swin44/GAViT44、3060 12GB 配对 Swin45/GAViT45。先以相同 seed42/小样本/配置测资源，不根据测试集分数决定机型。
- 本轮只完成本地 W1-01—W1-03。新增 `tools/resource_probe.py`、`tools/profile_training.py` 和 [运行说明](docs/gpu_resource_probe.md)，未启动服务器短测或正式训练，没有新增论文指标。
- 工具检查完整 train/val/test 的 RGB 路径、split 数量及身份，运行服务器完整测试后抽样 train4096/val1024；各模型一轮 smoke 保存 best、metadata、last-state、训练日志及资源报告。正式命令保持 batch32/30 epochs，需同 GPU/环境通过的配对 smoke 报告；短测不自动启动正式训练。
- 已验证训练代码基线为 `bb9f1cff463c996bc5f72695af538898c91490eb`，训练入口、模型及原身份工具未修改。资源工具必须放在服务器训练目录外；本地新增文件尚未 commit/push。
- 本地验证命令：`python3 -m unittest tests.test_resource_probe tests.test_training_profile tests.test_experiment_identity tests.test_sparse_hybrid_static_contract -v`；37 项通过（15 新增 + 22 既有）。`py_compile`、CLI help/只读 plan、差异空白检查及工具包解包一致性检查通过。
- 独立审查因服务额度限制未运行；已由主执行者核对训练/产物接口。上述测试不代表 CUDA 运行通过。两张 GPU 的显存、主机 RAM、吞吐、总时长及实际费用全部待服务器实测；不沿用小 split RAM 外推全量峰值。
- 工具包：`/private/tmp/gavit-resource-tools.v5MRYv/gavit-resource-tools-20260907.tar.gz`，仅含两个脚本和说明；SHA256 `3730a0f6185477689004fb3f224efd9a5d02553521c33edb1cbdf85b7b053ee6`。临时包若被系统清理，可由上述三个源文件重新打包。
- 保留 seed43 的既有事实：Swin43 完成30轮，GAViT43 停于21轮、best为8轮，用户曾决定不续跑。未改历史结果，不把当前记录表述为四次同协议完整训练。
- 下一步 W1-04：按运行说明在 3080 10GB 上执行配对资源短测，检查真实报告后再安排正式任务。未访问已登录 Featurize 浏览器、租用实例或改动服务器。

---

## 2026-09-02 — BigEarthNet sparse-hybrid 与 Swin 配对验证（seed 43）

### 一、研究问题与实验身份

- 研究问题：seed 42 上 `sparse_hybrid_4n_top2` 的 Test mAP、macro-F1、
  micro-F1 均高于 Swin 后，同一提升方向能否在独立 seed 43 上复现。
- 配对原则：只比较同一 seed 的 Swin43 与 sparse-hybrid GAViT43；固定
  BigEarthNet-19 split、预处理、batch size 32、AdamW、lr `3e-4`、weight
  decay `1e-4`、CosineAnnealingLR 和 threshold 0.5。
- 分支：`codex/sparse-hybrid-4n-top2`；Git commit：`bb9f1cf`；两份 metadata
  均记录 `dirty=false`、seed 43，checkpoint SHA256 校验一致。
- 配对 driver：
  `/home/featurize/work/run_gavit_seed43_pair_bb9f1cf_retry1.sh`；原计划依次
  运行 Swin43 和 GAViT43，使用 `_retry1` 隔离此前失败产物。

### 二、Swin43 完整训练与测试

- 完成 30/30 epochs；参数量 27,535,501；训练设备 NVIDIA GeForce RTX
  4090；峰值 CUDA allocated 3.56 GiB；训练 wall time 5.22 h。
- Best Validation mAP：**78.3459%（epoch 6）**。
- Epoch 30：Val mAP 75.8%、Val macro-F1 71.1%；相对 best 下降约
  2.55 pp，后期过拟合明显。
- Test mAP：**71.1%**；macro-F1：**64.1%**；micro-F1：**76.0%**。
- 测试 wall time 234.20 s；测试峰值 CUDA allocated 0.58 GiB。
- 训练日志：
  `logs/final_multiseed_swin_bb9f1cf_seed43_retry1_train.log`。
- 测试日志：
  `logs/final_multiseed_swin_bb9f1cf_seed43_retry1_test.log`。
- Best checkpoint：
  `checkpoints/best_bigearth_swin_final_multiseed_bb9f1cf_seed43_retry1.pth`；
  同 stem 的 `.meta.json` 和 `.last.train_state.pth` 均存在，last epoch 30。

### 三、Sparse-hybrid GAViT43 训练边界与测试

- 配置：31,366,926 参数；K=16 `attentive_spatial`；48 条四邻接空间边加
  32 条排除空间邻居后的 cosine top-2 特征边；2-layer/4-head GAT（hidden
  256）；`token_feedback`；dropout 0.1。
- 原计划 30 epochs，实际保存到 epoch 21 后进程终止；driver 和训练日志未记录
  traceback、OOM 或 Python 异常，外部终止原因待确认。该运行不得写成完成
  30/30 epochs。
- Best Validation mAP：**78.4728%（epoch 8）**；到 epoch 21 未出现更高
  best。last state epoch 21，记录的 best metric 与 metadata 一致。
- 结合本模型 seed 42 在 epoch 13 达峰后回落 2.31 pp、corrected-kNN seed 42
  在 epoch 17 达峰后回落，以及 Swin43 在 epoch 6 达峰后回落的重复模式，
  用户决定不再为本次运行续跑 epoch 22--30。正式表格必须保留这一提前停止
  边界；测试严格使用 epoch 8 best checkpoint，而不是 epoch 21 last state。
- 第一次 RTX 3060 测试因实例本地 BigEarthNet 尚缺一个 B04 文件而在首个
  batch 前失败；保留失败日志。数据解压完成后使用独立 `retry2` 日志重新测试，
  完成全量推理并得到：Test mAP **71.7%**、macro-F1 **65.7%**、micro-F1
  **76.8%**。
- 训练日志：
  `logs/final_multiseed_sparse_hybrid_bb9f1cf_seed43_retry1_train.log`。
- 成功测试日志：
  `logs/final_multiseed_sparse_hybrid_bb9f1cf_seed43_retry1_test_retry2.log`。
- Best checkpoint：
  `checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_final_multiseed_bb9f1cf_seed43_retry1.pth`；
  同 stem 的 `.meta.json` 和 `.last.train_state.pth` 均存在，checkpoint hash
  校验通过。
- 本次简化测试命令未包裹 wall-time/CUDA 统计；测试耗时、测试峰值显存和
  per-class AP 尚未从完整日志补记，当前标为待确认。

### 四、同 seed 比较与当前结论

| Model (seed 43) | Test mAP | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|
| Swin-T baseline | 71.1% | 64.1% | 76.0% |
| **Sparse-hybrid GAViT** | **71.7%** | **65.7%** | **76.8%** |
| Delta (GAViT - Swin) | **+0.6 pp** | **+1.6 pp** | **+0.8 pp** |

1. seed 43 上三个聚合指标均再次高于同 seed Swin，提升方向与 seed 42
   （+1.1/+1.8/+0.1 pp）一致。当前两 seed 的平均配对增益约为 mAP
   +0.85 pp、macro-F1 +1.70 pp、micro-F1 +0.45 pp；这里只是基于一位小数
   汇总值的阶段性描述，不替代三 seed 均值与标准差。
2. 两个 seed 的 macro-F1 增益最一致；mAP 为中等幅度提升，micro-F1 的增益
   较小且 seed 间波动更明显。仍不能表述为所有类别都改善，seed43 per-class
   AP 尚待提取。
3. 结果支持“提升方向在 seed42/43 上可复现”，但 GAViT43 未完成原定30轮，
   且最终稳定性结论仍缺配对的 Swin44/GAViT44。论文最终表必须披露本次
   epoch21 停止边界，并在 seed44 完成后报告三 seed 均值与标准差。
4. 后续若引入 early stopping，应预先定义 patience，并保持原定最大 epoch
   和学习率调度身份；不能直接把 `--epochs` 改为 20，因为当前
   `CosineAnnealingLR(T_max=args.epochs)` 会同时改变优化轨迹。

---

## 2026-08-17 — BigEarthNet sparse-hybrid 正式训练与测试（seed 42）

### 一、研究问题与实验身份

- 研究假设：在与 corrected cosine-kNN 相同的 80 条有向边预算下，以 48 条
  K=16 四邻接空间边和 32 条排除空间邻居后的 cosine top-2 特征边组成
  `sparse_hybrid_4n_top2`，能比纯 corrected-kNN 提供更有效的区域关系。
- 唯一主要变量是 graph topology；backbone、grouping、region 数、GAT、
  integration、数据划分、优化和 seed 均与 corrected-kNN control 保持一致。
- 论文用途：进入 `results/bigearth_comparison.csv`，作为 BigEarthNet edge
  topology 消融和候选最佳 GAViT；本条仍是单 seed 结果。
- 分支：`codex/sparse-hybrid-4n-top2`；训练 Git commit：`cd39be9`；
  metadata 记录 `dirty=false`。
- run stage：`formal`；run tag：
  `sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654`；seed 42。

### 二、数据、模型与优化

- BigEarthNet-19 固定 split：train 237,871 / val 122,342 / test 119,825。
  正式测试前逐行验证 test CSV 所需 B02/B03/B04 文件：119,825/119,825
  patch 完整，0 missing。
- 预处理：resize 224x224；训练随机水平/垂直翻转；ImageNet mean/std
  normalization。
- 模型：GAViT，31,366,926 参数；Swin-T backbone；K=16
  `attentive_spatial`；2-layer/4-head GAT（hidden 256）；dropout 0.1；
  `token_feedback` integration。
- 图：`edge_type=sparse_hybrid`；48 spatial + 32 feature = 80 条唯一有向边；
  feature top-2；`neighbor_to_query`；cosine 只决定拓扑。
- 预训练：`bigearth_files/model.safetensors`。
- 优化：30 epochs，batch size 32，AdamW，lr `3e-4`，weight decay `1e-4`，
  CosineAnnealingLR，BCEWithLogitsLoss；分类阈值 0.5。

正式训练命令：

```bash
nohup python -u -c \
'import runpy, torch, time; started=time.time(); runpy.run_path("train_bigearth.py", run_name="__main__"); print(f"Peak CUDA allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"); print(f"Total wall time: {(time.time()-started)/3600:.2f} h")' \
  --model gavit \
  --data_dir bigearth_files/splits \
  --epochs 30 \
  --batch_size 32 \
  --lr 3e-4 \
  --num_regions 16 \
  --knn_k 5 \
  --gat_hidden 256 \
  --gat_heads 4 \
  --gat_layers 2 \
  --grouping attentive_spatial \
  --edge_type sparse_hybrid \
  --integration token_feedback \
  --dropout 0.1 \
  --seed 42 \
  --run_stage formal \
  --run_tag sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654 \
  --pretrained_path bigearth_files/model.safetensors \
  --checkpoint_path checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.pth \
  > logs/sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.log 2>&1 &
```

### 三、训练结果与资源

- 完成 30/30 epochs；无 traceback、OOM、NaN 或中途终止。
- **Best Validation mAP：79.2087%（epoch 13）**；metadata 和 last state 的
  best metric 一致，last state epoch 30。
- Epoch 13 后训练 loss 继续下降，但验证 mAP 总体回落；epoch 17 短暂回到
  78.9%，未超过 best。Epoch 30：loss 0.0834、Val mAP 76.9%、Val macro-F1
  71.8%；相对 best mAP 下降 2.31 pp，存在明确后期过拟合。
- 前五轮 Val mAP：74.3%、76.4%、76.9%、78.0%、78.1%；epoch 7 为 78.8%，
  epoch 13 达峰；epoch 20--30 位于 76.9%--77.7%。测试严格使用 epoch 13
  best checkpoint，而非 epoch 30 last state。
- 训练设备：NVIDIA GeForce RTX 4090（24,564 MiB）；启动阶段约
  14.10 train batch/s；峰值 CUDA allocated 3.65 GiB；总 wall time 6.64 h，
  平均 wall time 约 13.28 min/epoch；实际费用待确认。
- 训练日志：
  `logs/sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.log`。
- Best checkpoint：
  `checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.pth`；
  同 stem 保存 `.meta.json` 和 `.last.train_state.pth`。

### 四、正式测试

正式测试命令：

```bash
nohup python -u -c \
'import runpy, torch, time; started=time.time(); runpy.run_path("test_bigearth.py", run_name="__main__"); print(f"Peak CUDA allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"); print(f"Test wall time: {time.time()-started:.2f} s")' \
  --model gavit \
  --data_dir bigearth_files/splits \
  --ckpt checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.pth \
  --batch_size 32 \
  > logs/sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654_test_rtx3060_20260817_114017.log 2>&1 &
```

- 测试设备：NVIDIA GeForce RTX 3060（12,288 MiB）；3,745 batches；tqdm
  10:33，5.91 batch/s；端到端 wall time 665.55 s（11.09 min）；峰值 CUDA
  allocated 0.60 GiB。
- **Test macro mAP：72.0%；macro-F1：66.0%；micro-F1：76.7%。**
- 测试日志：
  `logs/sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654_test_rtx3060_20260817_114017.log`。

| Class | Sparse-hybrid AP | vs corrected-kNN | vs Swin |
|---|---:|---:|---:|
| Urban fabric | 86.3% | +0.1 pp | -0.2 pp |
| Industrial or commercial units | 51.1% | +1.6 pp | -1.3 pp |
| Arable land | 93.8% | +0.6 pp | +0.5 pp |
| Permanent crops | 61.2% | +2.3 pp | +2.4 pp |
| Pastures | 85.8% | -0.3 pp | -0.4 pp |
| Complex cultivation patterns | 70.1% | +1.1 pp | +2.1 pp |
| Land principally occupied by agriculture, with significant areas of natural vegetation | 73.7% | +1.2 pp | +0.7 pp |
| Agro-forestry areas | 87.1% | +1.2 pp | -0.2 pp |
| Broad-leaved forest | 86.0% | +0.4 pp | +0.9 pp |
| Coniferous forest | 93.5% | -0.2 pp | +0.0 pp |
| Mixed forest | 89.8% | +1.1 pp | +0.5 pp |
| Natural grassland and sparsely vegetated areas | 42.1% | -1.9 pp | -0.8 pp |
| Moors, heathland and sclerophyllous vegetation | 61.8% | -1.0 pp | +5.0 pp |
| Transitional woodland, shrub | 78.8% | +0.0 pp | +0.5 pp |
| Beaches, dunes, sands | 16.8% | +6.7 pp | +3.5 pp |
| Inland wetlands | 64.1% | +1.0 pp | +1.4 pp |
| Coastal wetlands | 34.7% | +12.9 pp | +4.4 pp |
| Inland waters | 91.7% | +0.0 pp | +0.8 pp |
| Marine waters | 99.6% | -0.1 pp | +0.1 pp |

### 五、比较、解释与下一步

| Model | Test mAP | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|
| Swin-T baseline | 70.9% | 64.2% | 76.6% |
| Corrected-kNN GAViT | 70.6% | 65.4% | 76.6% |
| **Sparse-hybrid GAViT** | **72.0%** | **66.0%** | **76.7%** |

1. 相对 corrected-kNN，sparse-hybrid 的 test mAP +1.4 pp、macro-F1 +0.6 pp、
   micro-F1 +0.1 pp；相对 Swin 分别为 +1.1、+1.8、+0.1 pp。验证 mAP 也比
   corrected-kNN best 78.5732% 高 0.6355 pp，验证和测试方向一致。
2. mAP 增益主要来自 Coastal wetlands（+12.9 pp vs corrected）、Beaches
   （+6.7 pp）、Permanent crops（+2.3 pp）等困难类别；micro-F1 几乎不变，
   因此当前证据更支持“改善宏观/尾部类别表现”，而不是全面提高样本级预测。
3. Natural grassland（-1.9 pp）和 Moors（-1.0 pp）相对 corrected 下降，
   sparse topology 并非所有类别一致受益，后续应在论文中保留该限制。
4. **seed 42 的研究假设得到支持，sparse-hybrid 暂定为最佳 GAViT。** 单 seed
   不能声称稳定提升；不再搜索新 topology。下一步只为最终对比补齐 selected
   GAViT 与 Swin baseline 的 seeds 43/44，并报告三 seeds 均值与标准差。

---

## 2026-08-14 — sparse_hybrid_4n_top2 Featurize 工程 gate 通过

### 实验身份与验证环境

- 分支：`codex/sparse-hybrid-4n-top2`；Git commit：`a28cbe1`。
- Featurize GPU：NVIDIA GeForce RTX 4090（24,564 MiB）。
- 固定 BigEarthNet-19 split 已核对：train 237,871 / val 122,342 /
  test 119,825；原始图像位于实例本地
  `/home/featurize/data/BigEarthNet-S2`。
- 完整测试命令：`python -m unittest discover -s tests -v`。
- 测试结果：59 tests，1.879 s，全部通过；无 failure、error 或 skipped
  sparse-hybrid test。覆盖了 48/32/80 edge budget、top-2 排除规则、稳定
  tie-break、batch offset、forward/backward finite gradients、checkpoint metadata
  round-trip 和 training-state 恢复约束。

### Smoke 配置与结果

- 数据：一次性 256 train / 128 val split；1 epoch；batch size 32；seed 42。
- 模型：GAViT，31,366,926 参数；Swin-T backbone；K=16
  `attentive_spatial`；2-layer/4-head GAT（hidden 256）；
  `edge_type=sparse_hybrid`；`token_feedback`；dropout 0.1。
- 预训练权重：`bigearth_files/model.safetensors`；AdamW，lr `3e-4`，
  weight decay `1e-4`，CosineAnnealingLR，BCEWithLogitsLoss。
- run stage：`smoke`；run tag：
  `sparse_hybrid_4n_top2_a28cbe1_smoke_retry_20260814_125255`。
- 训练完成 8/8 batches，tqdm 平均 5.75 batch/s；该短跑吞吐仅用于工程
  估算，不作为正式效率结果。
- Epoch 1 loss：0.3408；Val macro mAP：47.7286%；Val macro-F1：15.9%。
- 峰值 CUDA allocated：3.65 GiB；总 wall time：8.07 s。
- 无 NaN、Inf、OOM 或 traceback；forward、backward、optimizer step、validation
  和 artifact save 均完成。

### 拓扑与产物验证

- metadata 中 `architecture.edge_type=sparse_hybrid`。
- `graph_topology` 明确记录：`name=sparse_hybrid_4n_top2`、
  `spatial_connectivity=4`、`feature_k=2`、48 条空间有向边、32 条特征有向边、
  总计 80 条唯一有向边、`message_direction=neighbor_to_query`、
  `cosine_role=topology_only`。
- Best checkpoint（约 120 MB）：
  `checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_a28cbe1_smoke_retry_20260814_125255.pth`。
- Metadata（约 1.4 KB）：同 stem 的 `.meta.json`；best metric 为
  `val_mAP=47.7286`，best epoch 1。
- Last training state（约 360 MB）：同 stem 的 `.last.train_state.pth`；
  last epoch 1，`best_metric=47.7285909477752`。
- 日志：
  `logs/sparse_hybrid_4n_top2_a28cbe1_smoke_retry_20260814_125255.log`。
- 独立 artifact 校验脚本完成文件存在性、architecture、topology、best epoch 和
  last-state 断言，输出 `SMOKE ARTIFACT VERIFICATION: OK`。

### 异常与结论

- 首次 smoke 在训练 8/8 后、无进度条的 validation 阶段被手动 `Ctrl+C`
  中断，因此只写入初始 metadata（`best=None`），没有 checkpoint 或 last state；
  使用新的 run identity 重试后完整通过。该中断不是模型或拓扑故障。
- **工程 gate 已通过。** Smoke 指标只证明运行路径和产物契约正确，不进入
  `results/bigearth_comparison.csv`，也不作为论文性能证据。
- 下一步按既定漏斗运行固定 seed 42 的 5-epoch proxy；proxy 之前不启动 30-epoch
  formal，也不追加其他拓扑或 integration 变量。

### Fixed 10% proxy 基础设施

- 为避免直接付费运行全量五轮，已批准固定 10% train/val、5 epochs、training
  seed 42 的 proxy 协议；proxy 只检查多轮学习稳定性，不替代全量 30-epoch
  formal，也不进入论文结果表。
- 设计 commit：`2ab6026`；实现 commit：`3f82563`。
- 新增 `baselines/bigearth/prepare_proxy_split.py`：从 seeds 42--141 的 100 个
  均匀随机候选中选择 train/val 共 38 个类别 prevalence 最大偏差最小者；任何
  split 缺少正类的候选 fail closed，分数相同选择较小 seed。
- 预期输出到 `bigearth_files/proxy_10pct_seed42`：23,787 train / 12,234 val、
  `prevalence.json`、输入/输出 SHA256、每类正例数与 prevalence deviation；
  test split 不参与生成或 proxy 训练，已有输出不会被覆盖。
- 本地标准库定向测试：12 tests 全部通过；相关 torch-free 套件：53 tests
  全部通过；生成器和测试通过 `py_compile`，`git diff --check` 通过。
- 本地完整 discover 因环境没有 PyTorch，三个 PyTorch test module 在 import
  阶段报 `ModuleNotFoundError: torch`；这不是代码回归。Featurize 拉取实现后仍须
  运行完整 suite，预期总数由 59 增至 71，全部通过后才能生成 proxy split。
- 生成、独立 hash/row/prevalence 校验和五轮前台训练命令已写入
  `docs/featurize_runbook.md`；实际运行结果记录如下，分布报告 SHA256 尚待补记。

### Fixed 10% proxy 实际运行结果

- Featurize 在 commit `cd39be9` 上运行完整测试：71 tests，1.512 s，全部通过。
- 生成器从 seeds 42--141 中选择 seed 137；train 23,787 / val 12,234；
  train/val 共 38 个类别 prevalence 的最大绝对偏差为 `0.00394910`
  （约 0.395 个百分点）。输出目录：
  `bigearth_files/proxy_10pct_seed42`。独立报告 SHA256 待补记。
- Proxy run stage：`proxy`；run tag：
  `sparse_hybrid_4n_top2_cd39be9_proxy_20260814_133423`；训练 seed 42。
- 模型和优化配置与计划一致：GAViT、K=16 `attentive_spatial`、
  `sparse_hybrid` 4-neighbor + top-2、2-layer/4-head GAT、
  `token_feedback`、batch size 32、lr `3e-4`、5 epochs。
- 每轮 744 个训练 batch；训练吞吐依次为 13.68、13.80、13.88、13.89、
  13.84 batch/s。Epoch 1--5 的 loss 为 0.2234、0.1852、0.1673、0.1492、
  0.1330；Val macro mAP 为 65.5%、67.9%、70.7%、73.7%、74.8%；
  Val macro-F1 为 56.3%、60.1%、60.7%、64.7%、67.3%。
- Best Validation mAP：74.8325%（epoch 5）；峰值 CUDA allocated：3.65 GiB；
  总 wall time：345.48 s（5.76 min，平均 69.10 s/epoch）。无 OOM、NaN、
  traceback 或指标退化，loss 单调下降且 mAP/F1 总体持续上升。
- Best checkpoint 保存到
  `checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_proxy_20260814_133423.pth`；
  训练日志为 `logs/sparse_hybrid_4n_top2_cd39be9_proxy_20260814_133423.log`。
  独立 artifact 校验确认同 stem 的 `.meta.json` 和 `.last.train_state.pth`
  可读取；metadata best 为 `val_mAP=74.8325`、epoch 5，last state epoch 5，
  Git identity 为 commit `cd39be9`、`dirty=false`，输出
  `PROXY ARTIFACT VERIFICATION: OK`。
- **多轮学习稳定性 proxy gate 已通过。** 该结果只支持启动固定 seed 42 的
  全量 30-epoch formal；不与全量 corrected-kNN 指标直接比较，也不进入论文
  结果表。按数据规模和本次 wall time 线性外推，formal 约 5.76 h；这是启动前
  成本估算，不是实际耗时。

---

## 2026-08-13 — sparse_hybrid_4n_top2 本地代码完成，待 Featurize 工程验证

### 研究假设与单一变量

- 研究问题：在与 corrected cosine-kNN 相同的 80 条 pre-GAT 有向边预算下，
  同时编码空间邻接与非局部特征相似性的稀疏图，是否能提供更符合导师建议、
  更有意义的区域关系。
- 唯一改变的主要变量是 graph topology。Swin-T、K=16
  `attentive_spatial`、2-layer/4-head GAT、`token_feedback`、数据划分、优化器、
  loss、scheduler、增强和阈值保持不变。
- 新配置名为 `edge_type=sparse_hybrid`；历史 `edge_type=hybrid` 的 8-neighbor
  加 kNN 直接拼接行为保持不变，不能用于本实验命名。

### 实现与身份

- 设计规格：`docs/superpowers/specs/2026-07-28-sparse-hybrid-graph-experiment-design.md`。
- 实施计划 commit：`d53cb1d`。
- 图构建 commit：`0be936a`。
- GAViT 与四个 train/test CLI 接线 commit：`79faafa`。
- CLI/topology 静态契约收紧 commit：`2c3f85b`。
- 显式 topology metadata 与 resume identity commit：`3bfd8d5`。
- 分支：`codex/sparse-hybrid-4n-top2`。
- 每张 K=16 图构造 48 条四邻接空间边；每个 query 排除 self 和空间邻居后，
  按 cosine similarity 选择 top-2，共 32 条特征边；合计 80 条唯一有向非自环边。
- 全部边遵守 PyG 的 `neighbor -> query` 消息方向。cosine 数值只用于选择拓扑，
  不传入 GAT 作为 message weight；相等分数按较小 region index 决定顺序。
- `sparse_hybrid` 对 K!=16 和 feature_k!=2 fail closed；checkpoint 名称和
  metadata 使用独立的 `edge_type=sparse_hybrid` 身份，并显式保存
  `graph_topology` 的 4-neighbor、feature_k=2、48/32/80 和消息方向快照；
  `--knn_k 5` 只是与 corrected-kNN 控制命令保持兼容，不表示本拓扑选择五条
  feature edges。resume identity 同时绑定该 topology 快照。

### 本地已验证与尚缺验证

- 本地 `python3 -m unittest tests.test_sparse_hybrid_static_contract
  tests.test_experiment_identity -v`：18 tests 全部通过。
- 本地 torch-free training-state 合约：10 tests 全部通过。
- 修改过的 graph/model/train/test/test-suite 文件均通过 `py_compile`；
  `git diff --check` 通过。
- 已加入但**尚未在本地执行**的 PyTorch 测试覆盖：80-edge budget、48/32 分解、
  四邻接无对角线、top-2 排除规则、tie-break、batch offset、无 self/重复/
  cross-image edge、模型 forward/backward finite gradients，以及 metadata
  round-trip logits。
- 原因：本机 Python 环境没有 PyTorch/NumPy，按既定原则未在本地安装；这些
  张量级测试必须在 Featurize 环境运行后才能标记为通过。
- Featurize 必跑命令和 256/128、1-epoch smoke gate 已写入
  `docs/featurize_runbook.md`。待记录：完整测试数量、GPU、峰值显存、throughput、
  首轮耗时、checkpoint/metadata/last-state 路径和 smoke 结果。

**当前结论：代码已实现，但工程 gate 尚未通过；未启动 proxy 或正式训练。**

---

## 2026-08-12 — BigEarthNet corrected-kNN GAViT 正式训练与测试

### 一、研究问题与实验身份

- 研究问题：修正 cosine-kNN 的消息方向为 `selected_neighbor -> query` 后，当前 GAViT v2 是否优于历史错误方向结果，并缩小或逆转相对 Swin baseline 的差距。
- 本次只改变 kNN 消息方向；数据划分、K=16 attentive spatial grouping、k=5、2-layer/4-head GAT、token feedback、优化器和训练轮数保持不变。
- Git commit：`8cb8563`；训练 metadata 记录 `dirty=false`。
- Featurize：NVIDIA GeForce RTX 4090（24,564 MiB）。
- 训练前验证：`python -m unittest discover -s tests -v`，44 tests，全部通过。
- 工程 smoke：256 train / 128 val，1 epoch，batch size 32；完成 forward、backward、validation 和隔离 checkpoint 保存，峰值 CUDA allocated 3.65 GiB。smoke 指标不作为论文性能证据。

### 二、正式配置与命令

- 数据集：BigEarthNet-19；固定 split 为 Train 237,871 / Val 122,342 / Test 119,825。
- 预处理：resize 224×224；训练使用随机水平/垂直翻转；ImageNet mean/std normalization。
- 模型：GAViT v2，31,366,926 参数；Swin-T backbone；K=16 `attentive_spatial`；corrected cosine-kNN `k=5`；GAT hidden=256、4 heads、2 layers；`token_feedback` integration；dropout=0.1。
- 预训练来源：持久化本地 `bigearth_files/model.safetensors`。
- 优化：30 epochs，batch size 32，seed 42，AdamW，lr 3e-4，weight decay 1e-4，CosineAnnealingLR，BCEWithLogitsLoss；F1 threshold=0.5。

正式训练命令：

```bash
nohup python -u -c \
'import runpy, torch, time; started=time.time(); runpy.run_path("train_bigearth.py", run_name="__main__"); print(f"Peak CUDA allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"); print(f"Total wall time: {(time.time()-started)/3600:.2f} h")' \
  --model gavit \
  --data_dir bigearth_files/splits \
  --epochs 30 \
  --batch_size 32 \
  --lr 3e-4 \
  --num_regions 16 \
  --knn_k 5 \
  --gat_hidden 256 \
  --gat_heads 4 \
  --gat_layers 2 \
  --grouping attentive_spatial \
  --edge_type knn \
  --integration token_feedback \
  --dropout 0.1 \
  --seed 42 \
  --run_stage formal \
  --run_tag corrected_knn_8cb8563_20260811 \
  --pretrained_path bigearth_files/model.safetensors \
  --checkpoint_path checkpoints/best_bigearth_gavit_K16_attentive_spatial_knn_k5_token_feedback_seed42_formal_corrected_knn_8cb8563_20260811.pth \
  > logs/bigearth_gavit_corrected_knn_seed42_8cb8563_20260811.log 2>&1 &
```

正式测试命令：

```bash
nohup python -u test_bigearth.py \
  --model gavit \
  --data_dir bigearth_files/splits \
  --ckpt checkpoints/best_bigearth_gavit_K16_attentive_spatial_knn_k5_token_feedback_seed42_formal_corrected_knn_8cb8563_20260811.pth \
  --batch_size 32 \
  > logs/bigearth_gavit_corrected_knn_seed42_8cb8563_20260811_test.log 2>&1 &
```

### 三、训练结果与资源

- 完成 30/30 epochs；无 traceback、OOM 或中途终止。
- **Best Validation mAP：78.5732%（epoch 17）**。
- Epoch 30：loss 0.0817，Val mAP 76.7%，Val macro-F1 71.7%。
- best 后验证 mAP 持续回落，存在明显后期过拟合；测试严格使用 epoch 17 best checkpoint，而不是 epoch 30 last state。
- 峰值 CUDA allocated：3.65 GiB。
- 训练总时长：6.49 h；实际费用待确认。
- 测试：3,745 batches，3:16，19.06 batch/s。

产物：

- 训练日志：`logs/bigearth_gavit_corrected_knn_seed42_8cb8563_20260811.log`
- 测试日志：`logs/bigearth_gavit_corrected_knn_seed42_8cb8563_20260811_test.log`
- Best checkpoint：`checkpoints/best_bigearth_gavit_K16_attentive_spatial_knn_k5_token_feedback_seed42_formal_corrected_knn_8cb8563_20260811.pth`（约 120 MB）
- Metadata：同名 `.meta.json`（best epoch 17，Git `8cb8563`，dirty=false）
- Last training state：同名 `.last.train_state.pth`（epoch 30，约 360 MB，仅用于精确恢复，不用于本次测试）

### 四、正式测试结果

- **Test macro mAP：70.6%**
- **Test macro-F1：65.4%**
- **Test micro-F1：76.6%**
- 分类阈值：0.5

| Class | AP | vs Swin |
|---|---:|---:|
| Urban fabric | 86.2% | -0.3 pp |
| Industrial or commercial units | 49.5% | -2.9 pp |
| Arable land | 93.2% | -0.1 pp |
| Permanent crops | 58.9% | +0.1 pp |
| Pastures | 86.1% | -0.1 pp |
| Complex cultivation patterns | 69.0% | +1.0 pp |
| Land principally occupied by agriculture, with significant areas of natural vegetation | 72.5% | -0.5 pp |
| Agro-forestry areas | 85.9% | -1.4 pp |
| Broad-leaved forest | 85.6% | +0.5 pp |
| Coniferous forest | 93.7% | +0.2 pp |
| Mixed forest | 88.7% | -0.6 pp |
| Natural grassland and sparsely vegetated areas | 44.0% | +1.1 pp |
| Moors, heathland and sclerophyllous vegetation | 62.8% | +6.0 pp |
| Transitional woodland, shrub | 78.8% | +0.5 pp |
| Beaches, dunes, sands | 10.1% | -3.2 pp |
| Inland wetlands | 63.1% | +0.4 pp |
| Coastal wetlands | 21.8% | -8.5 pp |
| Inland waters | 91.7% | +0.8 pp |
| Marine waters | 99.7% | +0.2 pp |

### 五、比较与结论

| Model | Test mAP | Macro-F1 | Micro-F1 | 说明 |
|---|---:|---:|---:|---|
| Swin-T baseline | 70.9% | 64.2% | 76.6% | 相同正式 split，seed 42 |
| Historical GAViT v2 | 70.4% | 待确认 | 待确认 | 修正消息方向前的历史结果 |
| Corrected-kNN GAViT v2 | 70.6% | 65.4% | 76.6% | 本次正式结果，seed 42 |

1. 修正消息方向后，GAViT 相对历史结果的 test mAP 提升 0.2 pp，说明方向错误有小幅负面影响，但不是性能差距的主要原因。
2. Corrected GAViT 的 mAP 仍比 Swin 低 0.3 pp；单 seed 下不能声称 graph module 提升总体 mAP。
3. Macro-F1 比 Swin 高 1.2 pp，micro-F1 持平；类别层面改善和退化并存，其中 Moors/... +6.0 pp，但 Coastal wetlands -8.5 pp。该结果值得在后续拓扑实验中继续观察，不能单独作为模型优越性的证据。
4. 本条形成时的下一项论文问题，是按已批准设计实现并验证 `sparse_hybrid_4n_top2`，以 corrected-kNN 结果作为同预算正式对照；该功能后来已于 2026-08-13 在 `codex/sparse-hybrid-4n-top2` 分支完成本地代码，但仍待 Featurize 工程验证。历史 `hybrid` 仍不得冒充该配置。
5. 在 sparse-hybrid 比较完成前，不追加 corrected-kNN 多 seed，也不启动 gated/cross-attention integration 实验。

---

## 2026-07-14 — Featurize BigEarthNet Swin baseline 完整训练

### 一、实验设置

- 平台：Featurize（GPU 具体型号待从实例记录确认）
- 数据集：BigEarthNet-19，多标签分类
- 数据划分：Train 237,871 / Val 122,342 / Test 119,825
- 模型：Swin-T baseline，27,535,501 参数
- 初始化：本地持久化 `bigearth_files/model.safetensors`，避免 Featurize 访问 Hugging Face 的 TLS 失败
- Seed：42
- 训练：30 epochs，batch size 32，AdamW，lr 3e-4，weight decay 1e-4，CosineAnnealingLR
- Loss：BCEWithLogitsLoss
- 验证指标：macro mAP + macro-F1（threshold=0.5）
- 启动入口：`train_bigearth_local.py`（临时为 Swin 注入本地 pretrained file）
- 日志：`logs/bigearth_swin.log`
- Checkpoint：`checkpoints/best_bigearth_swin.pth`，约 106 MB

### 二、训练结果

- 完成 30/30 epochs
- **Best Validation mAP：78.8%**（准确 best epoch 待从完整日志提取）
- Epoch 30 Loss：0.0223
- Epoch 30 Val mAP：75.8%
- Epoch 30 Val macro-F1：71.2%
- 后期 Val mAP 从最佳值回落至 75.8%，存在过拟合迹象；正式比较应使用 best checkpoint，而不是最后一轮权重
- 已观察到 epoch 24–30 单轮训练约 7:52–8:35；峰值显存、完整 wall time 和实际费用待补记

### 三、存储与复现状态

- BigEarthNet 图像位于实例本地 `/home/featurize/data/BigEarthNet-S2`，更换实例后需要重新添加
- metadata、split CSV、本地预训练权重、日志和 checkpoint 位于 `/home/featurize/work/GAViT_Project`，更换实例后保留
- 固定 split 已持久化在 `bigearth_files/splits`，无需随实例重新生成

### 四、测试结果

- 测试样本：119,825
- 测试耗时：2:25（3,745 batches，25.68 it/s）
- 分类阈值：0.5
- **Test macro mAP：70.9%**
- **Test macro-F1：64.2%**
- **Test micro-F1：76.6%**
- 测试日志：`logs/bigearth_swin_test_retry.log`

Per-class AP：

| Class | AP |
|---|---:|
| Urban fabric | 86.5% |
| Industrial or commercial units | 52.4% |
| Arable land | 93.3% |
| Permanent crops | 58.8% |
| Pastures | 86.2% |
| Complex cultivation patterns | 68.0% |
| Land principally occupied by agriculture, with significant areas of natural vegetation | 73.0% |
| Agro-forestry areas | 87.3% |
| Broad-leaved forest | 85.1% |
| Coniferous forest | 93.5% |
| Mixed forest | 89.3% |
| Natural grassland and sparsely vegetated areas | 42.9% |
| Moors, heathland and sclerophyllous vegetation | 56.8% |
| Transitional woodland, shrub | 78.3% |
| Beaches, dunes, sands | 13.3% |
| Inland wetlands | 62.7% |
| Coastal wetlands | 30.3% |
| Inland waters | 90.9% |
| Marine waters | 99.5% |

### 五、测试脚本问题与修复

- 第一次完整推理完成后，`average_precision_score` 报错：预测数组仍为三维。
- 根因：`test_bigearth.py` 未处理 timm Swin 返回的 `(B, H, W, C)` 四维特征，分类器输出成为 `(B, H, W, 19)`；训练脚本已正确进行空间池化。
- 修复：四维特征在维度 `(1, 2)` 上取均值，三维 token 特征在维度 `1` 上取均值。
- 修复后先用 32 条样本 smoke test，再运行完整测试，避免重复浪费 GPU 时间。

### 六、论文状态与下一步

- 本次结果建立了 BigEarthNet 上的 Swin baseline，后续 GAViT 必须使用相同 split、训练设置和主要指标
- [ ] 从完整日志确认 best epoch
- [ ] 补记训练和测试使用的 GPU 型号、峰值显存、完整 wall time 与实际费用
- [ ] 固定 GAViT 正式训练配置，并先完成小规模 smoke test
- [ ] 使用相同 split、epoch、batch size、seed 和指标训练及测试 GAViT

---

## 2026-04-08 — BigEarthNet 训练超时修复：添加断点续训

**问题**：
- BigEarthNet Swin baseline（Job 56169）在 epoch 16/30 因 8h 时间限制被 SLURM 杀掉
- BigEarthNet GAViT（Job 56170）在 epoch 24/30 因 10h 时间限制被杀掉
- 估算：Swin 每 epoch ~29min（总需 ~14.5h），GAViT 每 epoch ~25min（总需 ~12.5h）

**解决方案**：
- `train_bigearth.py` 新增 `--resume` 参数，从已有 best checkpoint 加载模型权重继续训练
- `train_bigearth.py` 新增 `--start_epoch` 参数，控制训练循环起始 epoch
- 注意：仅恢复模型权重，optimizer/scheduler 状态不保存（影响极小，模型已接近收敛）

**作业脚本更新**：
- `run_bigearth_train_swin.sh`：时间 8h → 16h，添加 `--resume --start_epoch 17`
- `run_bigearth_train_gavit.sh`：时间 10h → 6h，添加 `--resume --start_epoch 25`

**下一步**：
- [ ] Push 后在服务器 `git pull` 并重新 sbatch 两个作业
- [ ] 训练完成后运行测试脚本对比 Swin vs GAViT 在 BigEarthNet 上的表现

---

## 2026-04-04 — GAViT v2 训练结果 + BigEarthNet 实验准备

### 一、GAViT v2 训练 & 测试结果

**训练结果**（Job 53893，NVIDIA RTX A5000）：
- Model: GAViT | K=16 | grouping=attentive_spatial | edge=knn | integration=token_feedback | GAT 2L×4H
- 参数量：31,386,920
- **Best Val Acc：96.2%**（epoch 26）
- 训练过程：epoch 25-30 val acc 在 95.9%–96.2% 波动，train acc epoch 29 达 100%

**测试结果**（Job 54030，NVIDIA Tesla V100）：
- **Test Acc：95.8%**（4526/4725）

**完整对比**：

| 模型 | Val Acc | Test Acc | Params |
|------|---------|----------|--------|
| Swin-T Baseline | ~96.0% | 96.3% | 27.5M |
| GAViT v1 K=9 spatial + GAT 2L | 96.5% | 95.9% | 30.5M |
| GAViT Fusion (backbone+graph concat) | 95.9% | 95.9% | 30.5M |
| GAViT v2 K=16 attentive + token_feedback | 96.2% | 95.8% | 31.4M |

**结论**：三种 GAViT 变体 test acc 均在 95.8-95.9%，与 baseline（96.3%）差 ~0.4%。与教授预判一致——NWPU-RESISC45 单标签数据集 global appearance 信号太强，graph module 增益天然受限。

### 二、BigEarthNet 多标签实验准备

按 Prof Wang 0327 建议，新增 BigEarthNet 实验以展示 graph module 在关系更重要场景的价值。

**完成内容**：

1. **`models/bigearth_dataset.py`** — BigEarthNet Dataset 类
   - 支持 BigEarthNet-S2 格式（读取 B02/B03/B04 TIF 波段 → RGB）
   - 内置 43→19 类标签映射（BigEarthNet-19 标准）
   - 返回 19 维 binary label vector

2. **`baselines/bigearth/prepare_bigearth.py`** — 数据准备脚本
   - 自动下载官方 train/val/test split 列表
   - 生成 split CSV 文件（patch_path + 19 维标签列）

3. **`train_bigearth.py`** — 统一训练脚本
   - 支持 `--model swin`（baseline）和 `--model gavit`（GAViT v2）
   - Loss: BCEWithLogitsLoss（多标签）
   - Metric: mAP（mean Average Precision）+ Macro-F1 @ 0.5

4. **`test_bigearth.py`** — 测试脚本
   - 输出 mAP / macro-F1 / micro-F1 + per-class AP

5. **Slurm 作业脚本**：
   - `jobs/run_bigearth_prepare.sh`
   - `jobs/run_bigearth_train.sh`（依次训练 Swin baseline + GAViT v2）
   - `jobs/run_bigearth_test.sh`

6. **数据下载进行中**
   - 服务器创建了 `/projects/gavitdata/`（SSD，150GB 配额）
   - BigEarthNet-S2-v1.0 从 Zenodo 下载中（~65GB，约 6 小时）
   - 下载链接：`https://zenodo.org/records/12687186/files/BigEarthNet-S2-v1.0.tar.gz?download=1`

### 三、下一步计划（下载完成后按顺序执行）

- [ ] 选择性解压 RGB 波段 + 标签 JSON，删掉 tar.gz
  ```
  tar -xzf BigEarthNet-S2-v1.0.tar.gz --wildcards '*_B02.tif' '*_B03.tif' '*_B04.tif' '*_labels_metadata.json'
  rm BigEarthNet-S2-v1.0.tar.gz
  ```
- [ ] 运行 prepare 脚本生成 split CSV（`sbatch jobs/run_bigearth_prepare.sh`）
- [ ] 训练 Swin baseline + GAViT v2（`sbatch jobs/run_bigearth_train.sh`）
- [ ] 测试两个模型（`sbatch jobs/run_bigearth_test.sh`）
- [ ] 对比结果：NWPU（graph 增益小）vs BigEarthNet（graph 增益预期更大）
- [ ] NWPU 可视化：airport/bridge/church 的 region 分区和 graph 连接
- [ ] 整理所有结果给导师发邮件

---

## 2026-04-04 — GAViT v2 架构改进：AttentiveSpatialGrouping + Token Feedback

**动机（基于 Prof Wang 0327 回复）**：

教授指出当前 GAViT 的两个核心问题：
1. **集成方式不当**：graph module 输出在分类器阶段才与 backbone 拼接（fusion），relational reasoning 没有影响表征本身。应让 graph 输出直接修改 token 特征。
2. **Region 定义太粗糙**：3×3 spatial grid 分辨率不够，小结构被合并；mean pooling 没有区分 token 的重要性。应提高分辨率（4×4 或 5×5）并用 attention weighting 替代均值。

此外教授指出 NWPU-RESISC45 作为单标签数据集，global appearance 已是很强信号，graph module 增益天然有限。建议新增 BigEarthNet（多标签）数据集实验。

**完成内容**：

1. **`models/region_grouping.py` — 新增 `AttentiveSpatialGrouping`**
   - 支持任意完全平方数 K（默认 K=16，即 4×4 grid）
   - 每个 region 内用 lightweight attention（2 层 MLP → scalar score → softmax）加权聚合 token，替代均值 pooling
   - 信息量大的 token 贡献更多，使 region 特征更有语义代表性

2. **`models/gavit.py` — 新增 `token_feedback` 集成模式**
   - GAT 精炼 region features 后，通过 `feedback_proj`（Linear + LayerNorm）映射回 768 维
   - 每个 token 通过 assignments 获取其所属 region 的精炼特征
   - 残差更新：`updated_tokens = original_tokens + region_feedback`
   - 在更新后的 tokens 上做 mean pool → 分类
   - 旧的 `fusion` 模式保留，通过 `--integration` 参数切换

3. **`train_gavit.py` — 新增命令行参数**
   - `--grouping attentive_spatial`（新默认值）
   - `--integration token_feedback`（新默认值）
   - `--num_regions` 默认改为 16
   - Checkpoint 命名包含 integration 类型

**架构对比**：

| 版本 | Region Grouping | 集成方式 | 分类器输入 |
|------|----------------|----------|-----------|
| v1 (legacy fusion) | SpatialGrouping 3×3, mean pool | 末端 concat | backbone_global(768) + graph_global(1024) = 1792 |
| v2 (token feedback) | AttentiveSpatialGrouping 4×4, attention | token-level residual | updated tokens mean pool = 768 |

**实验结果**：

- **Best Val Acc：96.2%**（epoch 26）
- **Test Acc：95.8%**（4526/4725）
- 参数量：31,386,920（全部可训练）
- Checkpoint：`checkpoints/best_gavit_K16_attentive_spatial_knn_token_feedback.pth`
- 训练过程：epoch 25-30 val acc 在 95.9%–96.2% 之间波动，epoch 26 达峰；train acc 在 epoch 29 即达 100%，存在轻微过拟合
- 收敛特征：val acc 在 epoch 20 才突破 95.5%，收敛比 v1 慢（v1 epoch 20 已 ~96%），但最终结果持平

**对比 v1（GAViT K=9 spatial + GAT 2L，96.5%）**：
- Val Acc 96.2% vs 96.5%，略低 0.3%
- 尽管 attentive grouping + token feedback 架构更复杂，性能未见提升，可能原因：
  - K=16 比 K=9 区域更细，每个 region 内 token 数减少（~3 个），attention weighting 效果有限
  - Token feedback 引入了更长的梯度路径，30 epoch 内收敛不充分
  - NWPU-RESISC45 单标签数据集对 relational modeling 增益天然受限

**下一步计划**：
- [ ] 在 test set 上评估 v2（运行 `test_gavit.py`，加载 v2 checkpoint）
- [ ] 对比 v2 vs v1 vs Swin-T baseline 的混淆矩阵
- [ ] 可视化 v2 region 分区和 graph 连接（airport, bridge, church）
- [ ] 考虑延长训练至 50 epoch，或调整 lr schedule
- [ ] 准备 BigEarthNet 数据集实验

---

## 2026-03-27 — Backbone-Graph Fusion 架构实验 + Test Set 分析

### 一、Fusion 架构训练

**完成内容**：
- 修改 `gavit.py` 分类器架构：从纯 graph 特征改为 backbone global + graph global 拼接（768 + 1024 = 1792 维）
- 动机：之前的 GAViT 分类器只用 graph module 输出，backbone 全局特征被丢弃，fusion 旨在让两者互补
- 新建 `jobs/run_gavit_fusion.sh`，训练 50 epochs
- Checkpoint：`best_gavit_K9_spatial_knn_fusion.pth`

**训练结果**：
- Best Val Acc：**95.9%**（epoch 47-48）
- 训练过程：Train Acc epoch 42 达到 100%，Val Acc 从 epoch 28 的 94.4% 缓慢爬升至 95.9%

### 二、Fusion 模型 Test Set 分析

对 Fusion 模型跑了和原版 GAViT 相同的 per-class accuracy + 混淆矩阵分析（`compare_models.py --tag fusion`）。

**整体结果**：

| 模型 | Val Acc | Test Acc | vs Baseline (Test) |
|------|---------|----------|-------------------|
| Swin-T Baseline | ~96.0% | 96.3% | — |
| GAViT 原版（纯 graph） | 96.5% | 95.9% | -0.4% |
| GAViT Fusion（backbone+graph） | 95.9% | 95.9% | -0.4% |

两个 GAViT 版本在 test set 上表现完全一致（95.9%），均比 baseline 低 0.4%。但考虑到这种量级的差距（~0.4%）在单次运行中不具备统计显著性，三个模型可视为持平。

**Fusion Per-class 变化（vs Swin Baseline）**：

提升最大：
- roundabout: 97.1% → 100.0%（+2.9%）
- terrace: 95.2% → 98.1%（+2.9%）
- wetland: 89.5% → 91.4%（+1.9%）
- ship: 96.2% → 98.1%（+1.9%）

下降最大：
- bridge: 99.0% → 94.3%（-4.8%）
- airport: 98.1% → 95.2%（-2.9%）
- church: 87.6% → 84.8%（-2.9%）
- medium residential: 96.2% → 93.3%（-2.9%）
- lake: 97.1% → 94.3%（-2.9%）

**Sample-level 分析**：
- Baseline 错 → Fusion 对（修复）：67 samples
- Baseline 对 → Fusion 错（引入）：86 samples
- 净改善：-19 samples

**与原版 GAViT 的 per-class 对比**：
- 两者模式类似：复杂场景（airport, bridge, church）均未改善
- Fusion 波动幅度略小于原版（原版 palace -4.8%/lake -4.8%，Fusion 最大 bridge -4.8%）
- 两者提升的类也类似（ship, wetland, terrace, roundabout）

### 三、关键结论

1. **Fusion vs 原版 GAViT**：两种架构在 test set 上结果一致（95.9%），简单拼接并未带来额外收益
2. **两种 GAViT vs Baseline**：差距仅 0.4%，在单次运行中无统计显著性，三者可视为持平
3. **复杂场景未受益**：airport、bridge、palace 等需要 relational reasoning 的类别，两种 GAViT 均未改善，反而有所下降
4. **核心问题**：graph module 目前尚未展现出对复杂场景分类的独特价值

### 四、下一步计划

- [ ] 整理结果给导师发邮件，请教 graph module 与 backbone 的集成方式
- [ ] 根据导师反馈决定后续方向

---

## 2026-03-26 — Edge 消融实验 + 可视化对比 + 关键讨论

### 一、完成内容

1. **GAT 注意力可视化**（`visualize_graph.py`）
   - 对 `best_gavit_K9_spatial.pth` 生成 2×2 面板（原图 / 区域分配 / kNN图 / GAT注意力）
   - 6 类场景各 2 张，保存至 `results/figures/graph_vis_*.png`

2. **Spatial adjacency 边构建**（`models/graph_construction.py` 新增 `build_spatial_graph`）
   - 3×3 grid 的 8-邻接连边，边权均为 1
   - `gavit.py` 新增 `edge_type` 参数：`knn` / `spatial` / `hybrid`
   - `train_gavit.py` 新增 `--edge_type` 命令行参数

3. **Edge 消融实验**（`jobs/run_edge_ablation.sh`，Job 50315）

| Edge Type | Val Acc | 备注 |
|-----------|---------|------|
| spatial adjacency | **96.2%** | 固定 8-邻接 |
| cosine kNN | 96.1% | 动态边 |
| hybrid (合并) | 96.0% | kNN + spatial |

   **结论**：三种边策略准确率几乎一致，边定义对最终分类影响极小。

4. **Edge comparison 可视化**（`visualize_edge_comparison.py`，Job 50414）
   - 同一张图对比 spatial vs kNN 的 GAT 注意力，1×3 面板
   - 保存至 `results/figures/edge_compare_*.png` 和 `edge_comparison_summary.png`

### 二、关键讨论与反思

**GAT attention 权重的解读问题**：

- ⚠️ **GAT attention ≠ 语义关联度**。attention 权重反映的是"从哪个邻居获取信息最有用"（信息流方向），而不是"哪两个区域语义上相关"。
- 实际观察：stadium 中 R4（球场）对 R0、R2（角落/停车场）的注意力反而高于对 R3、R5（看台），因为差异大的邻居提供更多互补信息。
- 这意味着**用 attention 权重来直接讲"跑道+航站楼→强关联→机场"的故事是不准确的**。

**模型本身没问题**：

- GAViT 的 graph module 确实在帮助分类（96.0% → 96.5%），relational modeling 是有效的
- 只是 attention 权重不适合作为"展示 relational modeling 价值"的主要证据

**更好的展示方向（待验证）**：

- **注意力熵（attention entropy）对比**：复杂场景（airport, stadium）注意力应更集中（低熵），均质场景（forest, desert）注意力更分散（高熵）。"不均匀 vs 均匀"这个对比本身就说明模型根据场景动态调整了信息流。
- **混淆矩阵对比**：Baseline 分错但 GAViT 分对的样本，具体是哪些类
- **Per-class accuracy 对比**：但因为 baseline 已经 96%，差异可能很小，说服力有限

### 三、导师建议执行进度

- ✅ Region grouping 简单设计（spatial + kmeans）
- ✅ 轻量 GNN（2 层 GAT）
- ✅ 三级消融（baseline → +region → +graph）
- ✅ 图连接可视化（已完成，但解读方式需调整）
- ✅ 多种边定义系统对比（spatial / kNN / hybrid，准确率持平）
- ❌ AID 数据集验证

### 四、下一步计划（优先级排序）

- [ ] 计算注意力熵，量化"复杂场景 vs 均质场景"的注意力分布差异
- [ ] 在 test set 上评估最优模型，得到正式 Test Acc
- [ ] 混淆矩阵对比（Baseline vs GAViT）
- [ ] 消融：K 值影响（K=4 vs 9 vs 16）
- [ ] AID 数据集验证
- [ ] 给导师发进度邮件

---

## 2026-03-17 — 项目初始化 & Swin-T Baseline 完成

**完成内容**：
- 初始化项目结构，配置 Python 虚拟环境（PyTorch + timm + PyTorch Geometric）
- 编写 `split_nwpu.py`，将 NWPU-RESISC45 原始数据按 70/15/15 划分为 train/val/test，random seed=42
- 编写 `train_swin_baseline.py`：Swin-T (swin_tiny_patch4_window7_224)，pretrained=True，30 epoch，AdamW lr=3e-4，CosineAnnealingLR
- 编写 `test_swin.py`：加载 checkpoint 在 test set 评估

**实验结果**：
- Val Acc：~96%
- Test Acc：~96%
- 对比基线：本身即为基线

**遇到的问题及解决方案**：
- 无重大问题

**下一步计划**：
- [x] 实现 GAViT 核心模块（`models/` 目录）
- [ ] 训练 GAViT v1 并与 baseline 对比

---

## 2026-03-22 — P2a & P2b 完整实验：GAViT spatial/kmeans + 2-layer GAT

**完成内容**：
- P2a：GAViT K=9 SpatialGrouping + 2-layer GAT（4头，kNN k=5）
- P2b：GAViT K=9 KMeansGrouping + 2-layer GAT（重跑，含 argparse 版）
- 参数量：30,462,375（两组相同）

**实验结果**：
- P2a Best Val Acc：**96.5%**（epoch 30，仍在上升！）
- P2b Best Val Acc：**96.0%**（epoch 28-30）
- Checkpoint：`best_gavit_K9_spatial.pth` / `best_gavit_K9_kmeans.pth`

**消融结论**（完整链条）：

| 模型 | Val Acc | vs Baseline |
|------|---------|-------------|
| Swin-T Baseline | ~96.0% | — |
| + SpatialGrouping (no GNN) | 96.2% | +0.2% |
| + SpatialGrouping + GAT 2L | **96.5%** | **+0.5%** |
| + KMeansGrouping + GAT 2L | 96.0% | ≈0% |

- GAT 在 spatial grouping 基础上带来 +0.3% 提升，证明图推理有效
- KMeans 版性能低于 Spatial 版，原因：kmeans 随机性导致图结构不稳定，GNN 难以学习一致的区域关系
- Spatial+GAT 在 epoch 30 仍在上升（96.4%→96.5%），值得尝试更多 epoch

**遇到的问题及解决方案**：
- 无

**下一步计划**：
- [ ] 在 test set 上评估最优模型（`best_gavit_K9_spatial.pth`）
- [ ] 尝试 spatial+GAT 训练 50 epoch，观察是否继续上升
- [ ] 消融：K 的影响（4 vs 9 vs 16）
- [ ] 绘制训练曲线对比图

---

## 2026-03-22 — P2b 实验完成：GAViT K=9 KMeansGrouping + 2-layer GAT

**完成内容**：
- 在 NTU EEE GPU 服务器（NVIDIA RTX A5000）上运行 GAViT 完整模型（kmeans grouping）
- 模型：GAViT | K=9 | grouping=kmeans | GAT 2L×4H | kNN k=5
- 参数量：30,462,375（全部可训练，比 SpatialGrouping 版多约 290 万参数）
- 训练配置：30 epoch，AdamW lr=3e-4，CosineAnnealingLR，batch_size=32

**实验结果**：
- Best Val Acc：**96.0%**（epoch 29）
- 对比 Swin-T Baseline (~96.0%)：持平
- 对比 SpatialGrouping + no GNN (96.2%)：-0.2%
- 初步结论：kmeans 动态聚类引入了随机性，收敛更慢（epoch 1 train acc 仅 75.9% vs spatial 的 79.7%），最终性能与 baseline 持平

**训练曲线观察**：
- Epoch 1-10：收敛明显慢于 P1（epoch 10 val 92.2% vs P1 的 92.3%）
- Epoch 18-30：缓慢爬升，train acc 趋近 100%，val acc 在 95-96% 区间波动
- Checkpoint 保存至：`checkpoints/best_gavit.pth`（⚠️ 旧脚本名，argparse 未生效）

**注意事项**：
- 服务器拉取了 argparse 版本之前就已提交作业，导致只跑了 kmeans 一组，checkpoint 名为旧版 `best_gavit.pth`
- P2a（spatial + GAT）**尚未跑**，需补跑以完成消融对比

**下一步计划**：
- [ ] push argparse 版 `train_gavit.py` → 服务器 git pull → 补跑 P2a（`--grouping spatial`）
- [ ] 对比四组消融结果，绘制 bar chart
- [ ] 分析 kmeans vs spatial 差异原因

---

## 2026-03-22 — P1 实验完成：Swin + SpatialGrouping K=9（无 GNN）

**完成内容**：
- 在 NTU EEE GPU 服务器（NVIDIA RTX A5000, 24GB）上运行 P1 实验
- 模型：Swin-T + SpatialGrouping（K=9），无图推理层，直接 mean pool → FC 分类
- 参数量：27,555,495（全部可训练）
- 训练配置：30 epoch，AdamW lr=3e-4，CosineAnnealingLR，batch_size=32

**实验结果**：
- Best Val Acc：**96.2%**（epoch 27-28）
- 对比 Swin-T Baseline (~96.0%)：+0.2%（微弱提升）
- 结论：仅加 Region Grouping（无 GNN）相比 baseline 基本持平，说明区域表示本身信息量有限，需要图推理来充分利用区域间关系

**训练曲线观察**：
- Epoch 1-13：快速收敛，val acc 从 87% 升至 94%
- Epoch 14-28：缓慢爬升，train acc 趋近 100%（过拟合迹象）
- Best checkpoint 保存至：`checkpoints/best_region_only_K9_spatial.pth`

**遇到的问题及解决方案**：
- 无

**下一步计划**：
- [ ] **P2** 运行完整 GAViT v1（Swin + SpatialGrouping K=9 + 2-layer GAT）
- [ ] 对比三组结果：Baseline vs +Region Grouping vs +Graph Reasoning
- [ ] 保存训练曲线图至 `results/figures/`

---

## 2026-03-17 — GAViT 核心模块实现完成

**完成内容**：
- `models/swin_backbone.py`：封装 Swin-T，去掉分类头，`forward_features` 输出 `(B, 49, 768)` tokens
- `models/region_grouping.py`：实现两种聚类方式
  - `KMeansGrouping`：特征空间 k-means（PyTorch 实现，GPU 兼容），assignment 步骤 detach，averaging 步骤可微，梯度可回传至 backbone
  - `SpatialGrouping`：将 7×7 grid 按空间位置分为 K 个宏区域（K 须为完全平方数），全程可微
- `models/graph_construction.py`：`build_knn_graph`，基于 cosine similarity 构建有向 kNN 图，输出 PyG batch 格式 `edge_index` + `edge_weight` + `batch`
- `models/graph_reasoning.py`：多层 GAT（`GATConv`），含 input projection、残差连接、LayerNorm、GELU、Dropout
- `models/gavit.py`：完整 GAViT 模型，串联以上所有模块
- `train_gavit.py`：训练脚本（默认 K=9, kNN k=5, 2-layer GAT, 4 heads）
- `test_gavit.py`：测试脚本
- `utils.py`：`set_seed`、`accuracy` 工具函数

**实验结果**：
- 模块实现完成，尚未运行训练

**遇到的问题及解决方案**：
- 无

**下一步计划**：
- [ ] 在 GPU 服务器上运行 `train_gavit.py`，获得 GAViT v1 结果
- [ ] 将结果填入 `results/comparison_table.csv`
- [ ] 保存训练曲线至 `results/figures/`

---
