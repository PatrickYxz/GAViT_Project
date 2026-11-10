# HGTNet AID输入盘点回传（2026-11-04）

- `reported_output.txt`逐字节保留用户附件31361B，SHA256 `7f751431f1c976b5d9c3a42b22b5caae08328bb7bbc1c360843cbbc4e7729bc8`；`inventory_reported.json`仅规范化JSON，不执行终端命令。
- `audit_return.py`及`validation.json`核对23项：原manifest引用SHA/结构/counts/每类比例，240/60选择、300全部missing、原Industrial重复组、初始权重引用SHA、当前3060清单/磁盘、无训练及无test图读取等。
- [当前摘要](../../hgtnet_aid_inputs_20261104.json)状态为SERVER_REPORTED_AID_MANIFEST_WEIGHTS_PASS_IMAGES_MISSING。身份通过是目标机检查的用户回传，原manifest/权重/图片原字节未下载到本地。distribution metadata不等于实际CUDA/软件导入验证；nvidia-smi不等于模型已在3060运行。
- 归因：默认原图目录不存在；原清单和初始化不缺失，不重新prepare或重下权重。没有证据证明图片不存在于整台实例的其他位置。
- 找回旧持久化ZIP路径/大小/SHA，历史完整SHA由9月20日用户回传核对；当前文件存在性与SHA仍待新命令。此后发现的新来源事实修正先前准备文档“ZIP身份未知”，不重写旧inspect程序或原输出。
- [恢复命令](../../../output/hgtnet_aid_restore_20261104.md)仅在当前完整SHA匹配时解压到实例本地data，并重跑原inspect。未在本地或远端实际执行恢复、下载、模型或GPU训练。
