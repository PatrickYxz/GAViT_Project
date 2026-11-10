# 实现自审与验证边界

- 范围：独立标准库 inspector，CLI 仅 `inspect`，训练相关两个状态恒为 false；未修改 NWPU 源或包、未安装/网络/训练/启动 GPU kernel。
- 协议身份：CLI 固定原 SHA，内部 SHA 参数只用于 synthetic fixture；schema2、排序 AID taxonomy30、固定 group-v2/seed42与4242/比例、10000 和 4000/1000/5000 均核验。原 counts 只有 train/val/test，pool5000 在输出中明确派生。每类依据其实际支持数检查 floor 比例，不要求类间均匀。
- 输入防护：拒绝绝对/穿越/非规范/反斜杠/重复路径、非法标签/split/hash、重复 JSON key、畸形 JSON、跨类/跨 split 像素组和不一致 duplicate report。逐组件 O_NOFOLLOW 的目录描述符遍历拒绝 manifest、权重、所选图片及父目录符号链接；最终打开仅接受 regular file。
- 字节边界：只有合法且原 SHA 匹配的 manifest 才能触发所选300图读取；按每类 path 排序取8/2，不读取其余 train/val 或任何 test 图。像素摘要只作 manifest 组一致性检查，不解码、不声称重算像素或重新生成 split。文件 SHA 不符/缺失/不可读分别披露。
- 盘点：权重完整流式 SHA；GPU 为固定参数列表、10秒超时的 nvidia-smi；版本经 importlib.metadata；磁盘明确测量目录。GPU不可用或缺资产不捏造 UUID/版本/ZIP/容量/训练 readiness。
- 测试：13项 unittest，含不等类200/300/500真实形状的10000条synthetic manifest、成功300图、缺失300图、变更字节/符号链接/危险元数据、同类同split像素重复组、畸形JSON、GPU实际UUID和超时、固定CLI SHA、禁ML/禁写。首轮11项 RED 为功能文件缺失；2项额外边界测试验证现有实现。首轮 GREEN 的 macOS TemporaryDirectory 使用 `/var` 符号链接而被正确拒绝，夹具改为 `/private/tmp` 实体路径，失败与修复日志均保留。
- 交付：两成员zipapp仅为源与source_identity.json；源码字节及SHA相等，独占模式构建，二次构建 FileExistsError拒绝覆盖。实际zipapp在全标准库导入与文件创建/修改/删除/改名 guard下运行，缺资产盘点exit0/JSON并保持状态不通过；该次guard首稿把stdlib相对导入误判，修正guard解析package后通过，原失败日志保留。默认服务器路径在本机 `/home` 的macOS符号链接祖先下，因此诚实报告invalid；独立实体缺资产路径报告missing，均非服务器实测。
- 限制：本地无原 manifest 和AID真实图片，不能确认真实资产可用；13项标准库测试和15项交付检查不替代既有ML工程suite或未来目标GPU smoke。没有运行旧工程suite、回收未知原件或新增训练配置。

冻结源SHA：`4e3b66b91f115d99d01e39fe168e8cf5dd18503ec8593e45c39dbccfcd0eb7b4`。
冻结包SHA：`1cd2ab967cdbc386b294b397c6804bf06c5796c01332c709e44646f98d03f036`。
原NWPU包SHA：`b2bc9077cf37e75c83fbb72e7c4047d2fb9ede04db8fb4e4a56344500ff572e1`，保持一致。
