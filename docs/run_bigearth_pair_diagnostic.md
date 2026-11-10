# Swin44 / GAViT44 配对诊断

目的：在同一批验证图片上比较两个现有模型的预测，判断海岸湿地、沙滩/沙丘、自然草地的错误是漏判还是误报，并留下可进一步分析的具体样本。本次只做预测和成败样本展示；区域划分、实际图边、反馈幅度是拿到本次结果后的后续诊断。

## 服务器运行

09-16已确认服务器用现有工具完成了4096张抽样诊断。继续完整验证集评估时使用下节命令，无须再次上传、解包或拉取代码。下面的初次安装说明仅用于尚未部署工具的环境。

将本地交付包 `gavit-pair-diagnostic-20260915.zip` 上传到服务器 `/home/featurize/work/`，逐行运行：

```bash
git -C /home/featurize/work/GAViT_Project status --short --branch
unzip -n /home/featurize/work/gavit-pair-diagnostic-20260915.zip -d /home/featurize/work/
bash /home/featurize/work/gavit-pair-diagnostic-20260915/jobs/run_bigearth_pair_seed44.sh
```

工具在项目目录外运行，读取服务器现有项目代码。无须为这次诊断提交或拉取代码，也不会写入权重或修改split。输入已经按本任务路径配置：

- 项目：`/home/featurize/work/GAViT_Project`
- 划分：项目中的 `bigearth_files/splits/{train,val,test}.csv`
- Swin：`/home/featurize/work/swin44_3080_20260910/swin.pth`
- GAViT：`/home/featurize/work/gavit44_3080_20260909/gavit.pth`
- 两个权重旁边的 `.meta.json` 用于恢复各自架构。

启动后会打印PID、独立输出目录，以及已经填好路径的 `tail -f` / `cat` 命令，直接复制它打印的命令。只看到PID还不代表模型运行正常；日志应依次出现设备信息、两个模型加载记录、`Smoke: 8/8`、`SMOKE PASS`和后续进度。退出日志查看可按Ctrl+C，后台任务继续运行。

## 09-16后续：现有工具评估完整验证集

用户已同意此步骤。目的：覆盖val全部426张沙滩和610张海岸湿地正例及全部负例，缓存配对分数，供后续按海洋/内陆水体等标签分组比较。保持现有seed44权重、split、预处理和阈值0.5。主要变化只有评估范围4096→122342，不训练、不调参；统计结论用于失败案例与背景相关性讨论。

服务器工具路径来自成功运行的`run.json`。继续使用当前已解压的数据；脚本会在模型加载前检查本次清单的全部RGB文件。命令创建独立目录，输出路径中的`diagnosis`必须尚不存在。

09-17粘贴兼容更新：用户终端停在`then dquote>`，说明当前if块仍在等待闭合双引号，尚未执行。先按Ctrl+C回到正常提示符，再按顺序执行下面各行。启动命令保持一个物理行，去掉引号、if块和反斜杠续行；`${pair_full:?}`在目录变量为空时会阻止命令继续。路径由固定无空格前缀的mktemp生成，因此此处有意采用无引号展开。

```bash
pair_full=$(mktemp -d /home/featurize/work/pair44_fullval_XXXXXX)
echo ${pair_full:?}
nohup python -u /home/featurize/work/gavit-pair-diagnostic-20260915/tools/diagnose_bigearth_pair.py --repo /home/featurize/work/GAViT_Project --data-dir /home/featurize/work/GAViT_Project/bigearth_files/splits --swin-ckpt /home/featurize/work/swin44_3080_20260910/swin.pth --gavit-ckpt /home/featurize/work/gavit44_3080_20260909/gavit.pth --out ${pair_full:?}/diagnosis --samples 122342 --seed 42 --min-focus-support 0 --focus-classes 11,14,16 --examples-per-bucket 2 --batch-size 32 --workers 2 --device cuda --expected-counts 237871,122342,119825 >${pair_full:?}/diagnostic.log 2>&1 </dev/null &
tail -f ${pair_full:?}/diagnostic.log
```

开始后应看到`Manifest: 122342 images, including 122342 uniform`、`Selected image files: PASS`、设备/模型加载、`SMOKE PASS`及`Validation diagnostic`进度。刚打印PID不算启动验证成功。全量路径检查和末尾指标/CSV/压缩可能暂时不打印进度；不能把前向循环结束当作整个任务完成。

本命令直接后台调用Python，**不生成旧启动器的`diagnostic.exit`**。成功以日志`DIAGNOSTIC COMPLETE`及`diagnosis/run.json`中`status=complete`为准；捕获异常写入`diagnosis/failure.json`。按Ctrl+C退出`tail`不会停止nohup任务；保存echo打印的输出路径，重连后不要依赖Shell变量仍在。

同一终端完成后先查看：

```bash
cat ${pair_full:?}/diagnosis/report.md
tail -n 8 ${pair_full:?}/diagnosis/run.json
```

先回传这两段文本即可，保留`predictions.csv`供服务器端离线分组，无须先下载全部结果。旧脚本的报告标题/`scope`仍写validation_subset，但本次完整范围以`uniform_n=122342`、`supplement_n=0`和manifest为准；它始终是val评估，不是新的test分数。当前脚本不会自动生成按海洋/内陆水体标签分组的指标，该分析在缓存产生后进行。

本地已核对CLI参数、全量样本选择逻辑及Bash/zsh语法；完整122342张CUDA运行仍待用户在服务器执行。正式test表和三seed汇总不改。不给全量任务套用小样本的17秒总耗时，时间以该次进度、指标生成和归档实测为准。

## 脚本执行顺序

1. 读取原train/val/test CSV，验证19维二值标签及数量237871/122342/119825，统计类别正例数和三类共现。不加载训练或测试图片。
2. 从val固定抽4096张，抽样seed42。三类中每个正/负标签若不足64个，按标签补充样本；不足64时最多取现有样本。补充组单独标记，绝不合入随机子集的AP/PR。保存行号、路径、完整标签及清单。
3. 检查所选图片文件，读取现有metadata并严格加载权重，记录当前代码/权重摘要；使用原有RGB读取、224预处理及0.5分类阈值。
4. 同一批8张依次送入两个模型，检查输出形状与有限值；通过后自动继续完整清单。默认batch32、workers2、pin_memory=False，不开启AMP，不进行梯度更新。输出的吞吐和显存属于两个模型的配对诊断，不作为正式效率对比。
5. 对随机验证子集计算逐类AP、precision、recall、F1和TP/FP/FN/TN，AP同时与服务器sklearn实现比对。无正例或无负例时AP为N/A；分类指标分母为0时按0记录，解读时查看支持数。
6. 三类分别保留双方各自正确、共同正确、共同错误，并按真实正/负标签分组，每组最多2张，最多48个案例。每组按固定哈希排序选样，不按最大分差挑图；空组明确记录0。

## 看什么结果

日志最终应有 `DIAGNOSTIC COMPLETE`，输出目录中的 `diagnostic.exit` 应为 `diagnostic_exit=0`。

| 产物（输出目录/diagnosis/） | 用途 |
|---|---|
| `report.md` | 先看三类的支持数及两模型AP差值 |
| `per_class.csv` | 完整19类指标；TP/FP/FN/TN区分漏判和误报 |
| `cases.html` + `images/` | 下载整个结果包后，用浏览器打开成败样本页 |
| `cases.csv`、`case_counts.csv` | 案例索引、全部真实标签、每组可用与选中数量 |
| `predictions.csv` | 每张图的19维真实标签与两个模型的预测分数 |
| `pr_curves.csv` | 随机子集PR数据，threshold降序；不用于事后调整正式测试阈值 |
| `manifest.json/csv` | 固定样本清单，源CSV数据行从0计数 |
| `label_support.csv`、`label_statistics.json` | 三个原始split的正例数及目标类共现 |
| `swin_scores.json`、`gavit_scores.json` | 可供本地再次分析的原始预测缓存 |
| `preflight.json`、`loaded_models.json`、`run.json` | 执行配置、来源、真实设备、完成状态与资源 |
| `results.zip` | 上述诊断结果的打包文件，不含模型权重 |

**运行完回传 `results.zip` 即可继续分析。** 如果运行失败，先回传 `diagnostic.log` 末尾及 `diagnostic.exit`；Python捕获的异常还会保存为 `diagnosis/failure.json`。不要把 `prepared_only` 或 `smoke_passed_only` 当作全部诊断完成。进程被SIGKILL或实例消失时可能无法写退出标记。

这些是验证子集诊断结果，不替代既有完整测试集分数。病例关联也不能单独证明图模块造成了错误。

## 本地验证与可选模式

本地无PyTorch/NumPy运行环境；数据抽样、指标、输出、HTML转义和失败退出路径使用标准库测试。真实权重前向、图片解码和GPU资源仍需服务器内置的8张检查。完整运行会额外核对AP与sklearn结果。

```bash
python3 -m unittest tests.test_bigearth_pair_diagnostic -v
python3 tools/diagnose_bigearth_pair.py --help
bash -n jobs/run_bigearth_pair_seed44.sh
```

Python入口支持 `--prepare-only`（只生成标签统计及清单）和 `--smoke-only`（只运行8张）；两者都需要独立的新 `--out`。日常运行直接用上面的启动器即可。新输出统一使用 `pair44_diagnostic_XXXXXX`，失败后保留原目录，再次运行启动器会生成新目录。
