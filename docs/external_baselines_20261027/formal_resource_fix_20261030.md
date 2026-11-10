# 正式入口资源门禁修复

实际服务器inspect已证实文件上传、manifest、完整VGG和本机LIBLINEAR均通过。四个blockers归为三个根因：发行包metadata省略CUDA后缀；实例物理UUID变化；work空间15.86GiB而旧单卷容量门禁40GiB。df显示work为50G NFS、剩16G；实例根盘362G、剩349G，data目录尚未创建。

修复不改变模型/配方/seed或任何核心DFAGCN算法文件：

1. metadata允许2.2.2或2.2.2+cu121、0.17.2或0.17.2+cu121；已知其他构建如cu118和其他版本仍拒绝。run实际导入后仍核验CUDA12.1/cuDNN8902、真实设备、FP32及全部软件条件。metadata通过不等于CUDA已验证。
2. inspect未指定UUID时仅在恰好一张空闲10GiB RTX3080时选中并生成显式UUID命令；多候选必须显式选卡。run始终要求--gpu-uuid，实际runtime必须等于本次绑定值，不硬锁上一实例UUID。保存本次新GPU-0d9901ca-d2e0-d398-b9b4-2550e6db2fed。
3. 增加独立新--scratch-root，默认实例本地/tmp/包名_nwpu_seed42_01；将四份缓存和恢复图片写scratch，模型、历史、日志、线性模型/文本及最终指标继续work。缓存index/身份另存work/cache_manifests，记录scratch定位和易失性；不删除历史文件，不复用旧450缓存。
4. 分卷检查真实最近存在祖先的st_dev/剩余空间。scratch保留40GiB门禁；work按94500行×163331B=15434779500B的SVM文本上界，加1GiB artifact/安全余量，要求16508521324B（约15.375GiB）。固定配方图参数2957447，四个FP32权重约45MiB；十个5633×45线性文本模型上界约61MiB，1GiB余量覆盖这些产物/源/指标/日志。两路径同st_dev则合并两个预算，不分别重复计数。当前回传work17029922816B满足该分卷预算，但后续实际余量仍重查。
5. output/scratch任一已存在则拒绝，scratch不能覆盖output或反向包含；所有新增目录本轮独占。test恢复检查目标卷，不按ZIP所在work卷误判。没有自动删除或epoch恢复。

TDD覆盖实际回传版本/新UUID选择及多卡拒绝、run实际CUDA拒绝、分卷容量（同卷合并）、scratch路径与缓存位置、旧结果保护。新包使用fix1独立文件名/目录，旧正式包和两个smoke/probe包字节保持，不在服务器手改已交付源码。目标正式训练仍未启动；不为这个修复重跑GPU探针。
