# HGTNet AID seed42正式包

沿用用户持续授权的AID外部方法推进流程。真实3060已完成240/60一轮训练/验证/保存/重载；新增batch16容量检查也已complete/exit0/来源与产物检查true。实际batch16 allocated4731960320B（4.406981GiB）、reserved5213519872B（4.855469GiB），三次同步计算中位0.469230027秒、compute34.0984图/秒、总13.4862秒。这些是固定16图4次更新，不是全数据loader ETA或正式性能。下一阶段交付可复核的正式运行包，GPU运行仍由用户在Featurize执行。

## 研究问题与方案

研究问题：在原AID冻结数据协议上，为独立实现HGTNet补充一个seed42全量结果，供既有AID场景分类对比与后续统一协议讨论。结果包括Accuracy、macro-F1、30类per-class accuracy/F1与混淆矩阵，保存有序5000条预测及checkpoint/配置/来源/资源记录。这不是单独图模块消融，也不替代外部方法多seed或公平统一训练的结论。

采用新增AID formal siblings复用成熟NWPU算法；旧formal模块绑定45类/NWPU/3090不能直接套配置。另选项是把旧模块泛化，会改变已冻结历史源；继续只做smoke不能回答全量问题。工程选择：新增protocol/data/checkpoints/engine/entry和独立builder，不改原文件。使用既有Python/PyTorch，当前用户codex分支保留所有未提交研究产物，无Git提交/暂存/推送、新依赖或租卡。

## 冻结输入与配方

- 原manifest `/home/featurize/work/aid_protocol_20260920_03.json`，SHA `30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22`；30类、train4000/val1000/pool5000/test5000。保留原不等类数量和Industrial同像素两图同train；不去重或重新划分。
- 原图 `/home/featurize/data/AID/AID`；安全no-follow打开并SHA核对同一份字节再解码，要求RGB600×600。preflight与校准/refit只读pool，test元数据可提前绑定，但test图片字节与Dataset创建仅在refit及独立重载通过后授权。
- 模型配置canonical SHA `f313abb893b29d674ace281d8d2905670ee65db131f4240640079123e783589b`；总/可训练74612163参数、451参数张量。Swin-B候选351654207B/SHA `68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893`，从原候选fresh初始化，不载入smoke或NWPU训练后的checkpoint。
- 独立非官方配方继承已披露NWPU工程选择：seed42、calibration100 epochs、batch/eval16、Adam lr1e-5/WD5e-5/betas[.9,.999]/eps1e-8，CE/label smoothing0，无scheduler/warmup/AMP/TF32/实际梯度裁剪，workers0/pinfalse/dropfalse。epoch seed42+epoch。train/pool RGB→Resize256×256 bilinear→RandomCrop224→horizontalflip0.5→ImageNetnormalize；val/test CenterCrop224，不增强。
- 每轮验证以correct count严格大于才选新最佳，相同选最早；不根据test选轮。保存最佳/最终calibration。fresh pool refit与calibration初态state SHA必须相等，refit到selected epoch，保存final_refit并在独立新模型严格重载，数值误差≤1e-6，再最终test一次。
- 正式binding schema `hgtnet-aid-formal-binding-v1`，input_scope real_aid；显式cpu_fixture仅允许≤3epochs、batch/eval≤16、小计数、30类且devicecpu，不能标正式GPU证据。新recipe schema `hgtnet-aid-independent-formal-recipe-v1`；评估packet `hgtnet-aid-formal-evaluation-v1`，engine result `hgtnet-aid-formal-result-v1`，入口 `hgtnet-aid-formal-v1`。

## 接口

`aid_formal_protocol`：`RECIPE/REAL_COUNTS/IDENTITIES/recipe_sha256(recipe)/load_recipe(path)/validate_binding(value)`，接现有formal结构但30类/real_aid及新schema。新配置文件`configs/hgtnet_aid_formal_v1.json`与RECIPE内容相同。

`aid_formal_data`：`MANIFEST/MANIFEST_SHA/DATA_ROOT/COUNTS`；`prepare_inputs(manifest_path,*,expected_sha256=MANIFEST_SHA)`严格读原manifest，返回class_names/counts/selected(train,val,pool,test)/manifest_sha256，按path排序的独立拷贝；`_records(records,*,allow_test=False)`、`validate_selected(root,records,*,allow_test=False)`、`make_dataset(root,records,training=False,*,allow_test=False)`。默认拒绝test记录，training永远拒test，即使allow_test=True。validate返回checked_count/verified_count/test_image_bytes_read/verified_bytes；Dataset暴露paths/labels/records副本。懒导入ML/PIL，仅原数据本地目录，无自动ZIP恢复。

`aid_formal_checkpoints`：复制现有安全有限model与双slot完整epoch恢复API，导入新protocol，并用AID独立model/snapshot schema；保留CPU finite dense/strict keys-shape-dtype、weights_only、模型/Adam/完整绑定和generation校验、SHA与原子发布。

`aid_formal_engine.run_formal(model_factory,loader_factory,output,*,binding,device,on_progress=None,resume=False)`沿用原formal返回结构，30类、新schema、real_aid；参考同名旧模块复用训练/恢复/fresh初态算法。首批要求五组精确键和正有限norm/代表更新；CPU夹具张量数量可小，真实完整模型由入口再核对451/74612163。实际optimizer返回后即保留已发生步数；失败callback不得把已发生更新清零。每完整epoch保存交替snapshot并记录train/val/IO-inclusive wall与当前phase剩余估计，保留早期/最终轮及最佳验证packet。

## 恢复与test一次

run拒绝旧输出；resume必须同binding/source/recipe/output并有SHA有效完整epochsnapshot，不默默从头训练。每attempt独立保存preflight/command/runtime/log及原summary/progress/exit快照，保留被回滚的部分epoch已观察更新证据；最终optimizer_steps指当前接受的完整轨迹，旧attempt实际丢弃更新不混入该计数。未完成首epoch无snapshot时明确拒绝自动resume。

完成refit+reload后，先落盘`checkpoint_reload.json`，再执行test事务：`test_transaction.json`须含schema `hgtnet-aid-formal-test-transaction-v1`、binding_sha256、selected_epoch、refit_checkpoint_sha256及state started/complete；开始前原子写started，完整有序test写入test.json后记录test_sha256并原子变complete。resume若已有一致完整test packet/receipt应验证复用，不再调用test loader/forward；若started且没有完整有效packet，明确拒绝自动重复test。若test.json完整而complete事务提交前中断，可以核验packet后补齐事务，不重测。不能在resume时重写已有final_refit容器导致SHA变动；复用snapshot引用的final_refit并严格重载。

## 运行入口、容量证明和完成状态

`aid_formal_entry`自建22SOURCE_FILES与来源检查，复用aid_entry的安全读取、实际3060选择/runtime，不调用旧smoke/formal的source/inspect/status。显式inspect/run/resume/status；新schema/defaultoutput `/home/featurize/work/hgtnet_aid_formal_3060_20261105_v1/runs/aid_seed42_01`。自动选唯一空闲RTX3060 NVML12288MiB，Torch导入前绑定当前UUID；实际软件/8threads/FP32/cudnn等沿用已验证严格检查。run和resume均持有独占output sibling锁；拒symlink/foreign/覆盖和并行写入。

容量证明复用已通过探针：`--capacity-bundle`默认`/home/featurize/work/hgtnet_aid_batch16_probe_3060_20261105.pyz`，完整SHA必须 `fd54b4c1a21db1731d1addb0808c49104c11113b610883b79b2c6af456fc80f5`。用当前Python子进程只执行该可信旧包status（--output默认旧batch16_seed42_01），不训练/导入ML；要求complete/exit0/来源与产物检查true、原config/manifest/weights、16/64/4/1+3及实际UUID等于本次selected GPU。保留报告到preflight/attempt。改变卡/不再有有效证明应明确阻止长训练。probe paths/UUID属于attempt资源证据，不加入可恢复数值binding的固定身份。

preflight核对全pool5000图及权重/模型配置/recipe/source，test图片不读；output实际可用空间至少6GiB，Linux/proc/meminfo可用hostRAM至少6GiB，用于完整模型/Adam双slot与临时复制，不使用smoke2GiB预算。无scratch或ZIP恢复新参数。所有GPU门禁和输入检查在训练前执行；run实际重复必要检查以防路径变化。

callback的before_test授权test数据访问（包括复用已有packet的结束字节核对），每attempt首步/更新进度与root first_calibration_step.json、first_refit_step.json保存；first_step.json可保留初始calibration首步兼容。结束再次核对源/config/manifest/recipe/初值与pool/test字节；完成receipt才打印`HGTNET_AID_FORMAL_COMPLETE`，写summary/run.exit0/completion.marker、completeprogress。

status纯stdlib、无checkpoint反序列化或ML导入；核对full binding/source/runtime、100+selected条history/earliest最佳/步数25000+313E、两相firststep五组451更新、fresh初态相等、bestval及5000test有序身份/30类预测与指标重算、三主checkpoint和reload SHA/误差≤1e-6、完整test事务/末尾输入检查/exit0/marker/progress及一致summary。仅summary.completed或marker不能判完成。可复用纯Pythonmetrics helper写在entry内，以同一CM公式重算perclass和macroF1；status不重新执行GPU或读原图片。

## 封装与验收

新builder `tools/package_hgtnet_aid_formal.py`，source-only安全安装逻辑沿用现有AID builder，新delivery schema `hgtnet-aid-formal-delivery-v1`，env `HGTNET_AID_FORMAL_DELIVERY_ENTRY`。22source/config=旧AID17减aid_engine/旧builder，加5formal模块/新builder/recipe；加provenance/manifest/bootstrap共25ZIP成员。显式四动作，无默认训练，不安装依赖或覆盖旧目录。正式包`output/hgtnet_aid_formal_3060_20261105_v1.pyz`，用户上传小包保留原名。

TDD覆盖原manifest/不等类/同pixel组/安全路径/test gate/transform；30类fresh初态/calibration tie/refit/reload/中断恢复与test复用或拒绝；真实bundle/importguard/严格status/来源容量证明/输出锁与错误计数。CPU小模型只证明数值链，现有真实GPU小规模证据不冒充正式完成；最终独立审查、包/source/旧文件哈希与git diff --check通过再交付。原始正式命令/源identity/seed/输出/checkpoint固定在启动前；本地不执行实际GPU或完整训练。完整时长由首个真实epoch/当前phase平均估计，不把0.469秒或probe13.486秒换成正式账单。
