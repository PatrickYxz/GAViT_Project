# BigEarthNet软分配短测：先检查160张原图

2026-09-28。NWPU的360/90训练、保存重载、快照一致性和真实CUDA分配专属梯度已回传通过。下一步按既定方案验证BigEarthNet的19类多标签BCE入口，96train/64val、seed42、1轮；不重复NWPU，也不启动完整训练。

先运行下列只读命令。核对原train/val CSV冻结哈希，各自使用独立Random(42)按行号无放回抽96/64张，以原CSV顺序保留；检查19列二元标签、160个唯一绝对路径，以及480个原始RGB波段文件存在。后续小CSV使用同一抽样规则，不能因某些图缺失而改抽样。

```bash
python -c 'import csv,json,random,hashlib; from pathlib import Path; p=Path("/home/featurize/work/GAViT_Project/bigearth_files/splits"); sizes={"train":237871,"val":122342}; take={"train":96,"val":64}; hashes={s:hashlib.sha256((p/(s+".csv")).read_bytes()).hexdigest() for s in sizes}; assert hashes=={"train":"25734fd9e8dc7fd78fb259e768ae91fe11b297e1caa50a2d57b74b0296413d8f","val":"1c6627cb48e33588538c4e411e4f1b94560827f27d59fb7aeb0ea2b5d30c2a45"},"Split changed"; ids={s:set(random.Random(42).sample(range(n),take[s])) for s,n in sizes.items()}; rows={s:[r for i,r in enumerate(csv.DictReader((p/(s+".csv")).open())) if i in ids[s]] for s in sizes}; assert all(len(rows[s])==take[s] for s in sizes); assert all(float(r["label_"+str(i)]) in (0,1) for v in rows.values() for r in v for i in range(19)); patches=[Path(r["patch_path"]) for v in rows.values() for r in v]; assert len(set(patches))==160 and all(q.is_absolute() for q in patches); bands=[q/(q.name+"_"+b+".tif") for q in patches for b in ("B02","B03","B04")]; missing=[str(f) for f in bands if not f.is_file()]; print(json.dumps({"selected":take,"required_bands":len(bands),"missing_bands":len(missing),"missing_examples":missing[:3]},indent=2)); print("BIGEARTH_FILES_READY" if not missing else "BIGEARTH_FILES_MISSING")'
```

`BIGEARTH_FILES_READY`仅表示所选波段文件存在，尚未验证解码/前后向/显存。后续在生成独立小CSV及1轮短测前完成解码与预处理优先级检查、正负支持数和小CSV哈希记录。若`BIGEARTH_FILES_MISSING`，先定位原始ZIP并只恢复所需RGB波段，避免无条件重新全量解压；不要用JPEG画廊或随机输入替代原图。

本命令不创建输出、不修改正式划分、不读取test、不使用GPU。服务器数据当前状态和下一轮短测仍待返回；正式训练只有在工程检查通过后按既定研究问题安排。

## 09-28返回缺图：先定位压缩包

用户返回160张图共480个波段全部在预期路径下缺失，冻结CSV/标签/抽样断言已通过。仅确认抽样路径缺图，不能推断压缩包已删除。训练尚未启动。下一步只查data/work的两层目录，不遍历数百万TIF，也不解压或改文件：

```bash
ls -lah /home/featurize/data
find /home/featurize/data /home/featurize/work -maxdepth 2 -type f \( -iname '*.zip' -o -iname '*.tar' -o -iname '*.tar.gz' -o -iname '*.tgz' -o -iname '*.7z' -o -iname '*.rar' \) -printf '%s bytes  %p\n'
```

返回真实压缩包路径和格式后再制定选择性恢复命令；不猜ZIP文件名或内部目录。没有输出只表示这两个根目录的两层内未找到这些格式，不能断言服务器全局没有压缩包。优先保持相同160图，只恢复所需RGB原始波段；是否需要平台重新添加数据取决于定位结果。

## 09-28定位返回：data不存在，搜索范围内没有BigEarthNet原包

用户返回data目录不存在，work两层内只有AID、NWPU和两个小诊断工具ZIP，没有找到BigEarthNet原始压缩包。不能对这些工具包执行原波段恢复，也不将此有限搜索扩大为全服务器没有归档。现有项目记录未提供另一个已核实的原始包路径。

下一步沿用项目已记录的Featurize添加数据集流程，找到此前使用的BigEarthNet-S2原始数据。若界面支持只下载，先获取原包并回传实际路径，再检查成员并选择性提取480个所需波段；若平台只提供自动下载解压，完成该恢复流程后重新运行本页固定抽样检查。具体界面能力未现场核验，不承诺一定能跳过平台解压。保留原CSV、预训练和既有权重，不重新prepare或换抽样。

恢复后可将本次160张原图的RGB波段按原相对路径保存为work中的小归档，并记录文件哈希，用于后续短测恢复；该归档尚未创建，不替代正式训练所需完整数据。

## 恢复后继续

09-28用户报告恢复完成；下一步执行[96/64准备和单轮短测](run_bigearth_soft_smoke_20260928.md)。该准备命令包含原图复核，不再要求单独重复本页存在性检查。恢复后真实GPU及产物验收仍待回传。
