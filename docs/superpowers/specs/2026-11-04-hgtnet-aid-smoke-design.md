# HGTNet AID小样本包

用户已完成原ZIP SHA校验及图片恢复，回传原manifest/初值与240train+60val全部300图字节校验通过。沿用已授权的下一阶段，在当前RTX3060上验证AID独立实现的完整训练/验证/保存/重载链，不做正式训练或test评估。

## 冻结身份

- AID原manifest SHA `30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22`；30类/4000train/1000val/5000test。每类按路径排序取8train、2val，240/60、test0。
- 初始候选SHA `68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893`，独立部分Swin-B初始化；不用已训练NWPU权重。
- 新30类配置与旧HGTNet候选除num_classes外一致。工程smoke seed42/1epoch/batch2/Adam lr1e-4/CE，无增强/AMP/TF32，RGB600原图resize224 bilinear后ImageNet归一化，顺序验证。
- 当前NVML设备NVIDIA GeForce RTX3060，12288MiB，UUID `GPU-7acce72a-93cd-e56f-35d8-5d33ecf6655e`。实际run必须在导入Torch前设置CUDA_VISIBLE_DEVICES，检查实际Torch设备和CPU/CUDA软件。CUDA可见总内存可比NVML少不超过1GiB，不要求两个口径完全相等。
- Python3.11、Torch2.2.2+cu121、torchvision0.17.2+cu121、NumPy1.26.4、Pillow10.3.0、CUDA12.1/cuDNN8902；8 CPU线程，FP32、cudnn deterministic、benchmark/双TF32关闭。

## 实现接口

新增`external_baselines/hgtnet/aid_data.py`：标准库阶段不导入ML；`prepare_inputs(manifest_path, *, expected_sha256=MANIFEST_SHA)`验证原字节及结构，返回`class_names/counts/selected/manifest_sha256`。`counts={train:240,val:60,test:0}`；`selected`包含train/val/pool（300条）三组深拷贝记录。`validate_selected(root,records)`只读取给定安全普通文件并核对SHA，拒绝test行；`make_dataset(root,records)`懒导入Torch/PIL，读取同一份已哈希字节解码，验证RGB600×600→224，`records`属性保存有序身份。

新增`aid_engine.py`：`PROTOCOL`及`run_epoch_smoke(model,train_loader,val_loader,output,*,binding,device,model_factory,on_progress=None)`接口与旧smoke一致，但仅接受`schema=hgtnet-aid-smoke-binding-v1`、30类、`input_scope=real_aid`的240/60/cuda:0或明确`cpu_fixture`。binding含config/config_sha256/class_names/counts/runtime/val_paths/val_labels，另可包含manifest、初始权重及完整源码身份。强制train shuffle可复现、val sequential，首步五组梯度有限/非零且每组代表参数实际更新；120步遍历全部train，保存60条val预测、30类指标/CM、history和独立smoke checkpoint。新模型严格载入及数值误差≤1e-6。CPU夹具不能标成真实AID/GPU证据。

新增`aid_entry.py`和独立builder：stdlib inspect/status；run才导入ML。真实输入绑定、来源文件及30类配置、初值及所选300图在开始/结束检查；独立输出目录拒绝覆盖，进程锁、失败/退出状态有证据。来源身份包含AID模块、复用模型/预训练/梯度/指标/安全读取代码和配置，不仅六个模型文件。打包先以安全成员/hash规则装入独立真实目录后运行，不能直接zipimport依赖Path(__file__)的模型。

## 交付与边界

交付`hgtnet_aid_smoke_3060_20261104.pyz`及中文单行说明；inspect生成实际run_command，默认指向原manifest/data/初值、新output，不复用旧NWPU目录。只有完成全部阶段、末尾输入来源复核、summary、exit0后输出COMPLETE。失败后如有实际首步或optimizer更新要如实保留，不能把训练开始状态清零。

不改冻结NWPU源/配置/包，不覆盖历史结果，不安装依赖/租卡/本地或远端正式训练/提交推送。新增代码的夹具和可获得的本地30类数值检查只证明工程行为，目标3060的实际结果待用户执行。只测1epoch的val指标不进入正式比较表。
