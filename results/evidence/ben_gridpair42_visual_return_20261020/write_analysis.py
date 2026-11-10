"""Write the case report and current status from audited returned data."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=json.loads((HERE/'server_return/captures.json').read_text())
ANALYSIS=json.loads((HERE/'case_analysis.json').read_text())
AUDIT=json.loads((HERE/'validation.json').read_text())

def main():
    lines=['# BEN seed42：受训3×3与4×4的同图可视化分析','',
      '2026-10-20。已取得并归档实际捕获文件，核对37个清单文件SHA、8组配对记录、488条pre-GAT边及全部76项标签判断。捕获与父训练run.exit均为0；本次CPU推理耗时21.68秒，没有新增训练或全量测试。', '',
      '**本次结论：4×4在历史seed42总体test mAP上约高0.92pp；这4张预定验证图里纠正3个标签判断，同时新增1个漏判。改善、代价及未解决的错误都已展示。** 3和1是标签判断次数，不是整张图片分类结果；这些图也不估计BEN总体获益率。','',
      '## 图稿与数据','',
      '- [两页PDF：实际特征边与配对预测／区域内汇总权重](../output/pdf/ben_gridpair42_20261020/ben_gridpair42_cases.pdf)。',
      '- [主图300dpi PNG](../output/pdf/ben_gridpair42_20261020/graphs_predictions_300dpi.png)、[汇总权重300dpi PNG](../output/pdf/ben_gridpair42_20261020/pooling_300dpi.png)。',
      '- [可切换区域、空间边、特征边及权重的单文件HTML](../output/pdf/ben_gridpair42_20261020/comparison.html)。HTML完整保留19类概率，PDF显示所有阈值翻转及部分未解决错误。',
      '- [文件核验与范围](../results/evidence/ben_gridpair42_visual_return_20261020/validation.json)、[完整配对数据](../results/evidence/ben_gridpair42_visual_return_20261020/server_return/captures.json)。','',
      '## 阈值发生变化的全部标签','',
      '| 验证row | 类别（真实标签均为1） | 3×3概率 | 4×4概率 | 阈值0.5下的变化 |',
      '|---|---|---:|---:|---|']
    zh={0:'城市建成区',4:'牧场',11:'天然草地／稀疏植被',13:'过渡林地／灌丛',14:'沙滩／沙丘／沙地',18:'海洋水域'}
    for case in ANALYSIS['cases']:
        for r in case['threshold_flips']:
            verdict='纠正漏判' if r['outcome']=='k16_only_correct' else '新增漏判'
            lines.append(f'| {case["row_index"]} | {zh[r["class_index"]]} | {r["p9"]*100:.2f}% | {r["p16"]*100:.2f}% | {verdict} |')
    lines += ['', '## 逐图解释', '',
      '### a. row97389：牧场分数提高，但仍未检出','',
      '画面有深浅绿色斑块和细长浅色纹理。4×4使用更小的区域汇总单元，区域内最大汇总权重≥0.90的非单token区域为12/15，3×3为6/9。区域特征的取样位置和实际特征邻接随两套受训权重改变。', '',
      '真实标签包括牧场和针叶林。牧场概率29.89%→46.36%，方向朝真实标签移动，但仍低于0.5，不能列为纠正；本图没有任何标签跨过阈值。针叶林两模型均检出。本图保留为“分数变化、原错误仍在”的案例。', '',
      '### b. row14680：天然草地得到纠正，牧场误报仍存在','',
      '图像上方以连续深绿色纹理为主，底部有较浅绿斑块。4×4把底部组织拆成更多汇总单元；实际特征边中，区域8接收区域13、14的输入。这里的编号只标识图中位置，不能据颜色或连线直接断定节点属于草地。', '',
      '天然草地／稀疏植被是真实正标签，概率42.85%→51.59%，跨过0.5，纠正一次漏判；与此同时，真实为0的牧场64.38%→69.97%，两者均误报。这说明该图存在具体收益，但没有使所有类别判断都正确。', '',
      '### c. row28450：同图中同时出现明确改善与退化','',
      '深浅交错斑块和线状纹理跨越两种规则网格，4×4没有变成内容驱动的分区。真实捕获的关系也改变，例如左上区域0的特征来源从K9的4、6，变为K16的8、6；不同K的区域ID没有一一语义对应关系。', '',
      '过渡林地／灌丛44.74%→91.82%，纠正漏判；海洋水域64.96%→5.84%，新增漏判。两项都按原划分的正标签计分，不根据肉眼看到的地物改写标签。本图不能写成整图“改善”或“退化”，它展示多标签任务中不同地物判断的取舍。', '',
      '### d. row4439：建成区得到纠正，沙地仍差一步','',
      '左侧可见狭窄浅色岸带和陆地区域，右侧大部分为水面。按真实token成员，左列汇总宽度由3×3的96像素减为4×4的64像素，有机会减少岸带所在区域混入的宽水面。实际图仍含跨列联系：K16区域0接收5和12，不是只连接岸线上的节点。', '',
      '城市建成区41.38%→76.57%，纠正漏判。沙滩／沙丘／沙地17.67%→49.32%，虽提高31.66pp，仍未越过0.5，继续计为漏判。主图同时保留这两项，不把接近阈值写成识别成功。', '',
      '## 区域与图边可以解释到哪里','',
      '固定分区由7×7 token位置映射确定，3×3单轴token宽度为3/2/2，4×4为2/2/2/1。训练改变区域内汇总权重、节点特征、特征邻接和分类器；固定边界本身不沿地物边界学习。两种模型的图边均来自真实前向：K9为24条空间边＋18条特征边，K16为48＋32；PDF主图为清晰起见只画特征边，全部空间边在HTML中可切换。', '',
      '区域内权重常高度集中于少数token，权重页据真实attn scores重建，并与实际区域输出核对。非单token区域中，最大权重≥0.90的数量如下；分母不同，不把它当作统一的语义质量评分。K16右下角只有一个token，权重必为1，该区域已从计数分母排除。','',
      '| row | 3×3：高集中区域／非单token区域 | 4×4：高集中区域／非单token区域 |',
      '|---|---:|---:|']
    for case in ANALYSIS['cases']:
        a,b=case['pooling_concentration']['9'],case['pooling_concentration']['16']
        lines.append(f'| {case["row_index"]} | {a["non_singleton_max_ge_0_9"]}/{a["non_singleton_regions"]} | {b["non_singleton_max_ge_0_9"]}/{b["non_singleton_regions"]} |')
    lines += ['', '可以表述为：**更细的固定汇总单元与不同的受训区域特征、特征连接相伴，在部分标签上带来可复查的纠正，同时存在未解决错误和新增漏判。** 单图的天然草地纠正与整个test该类AP下降并不矛盾：一例概率跨阈值与整类样本的排序AP是不同统计对象。', '',
      '现有证据支持本次实验采用4×4的性能方向，但没有单独控制图边来归因某几条连接的贡献，也没有地物区域标注来量化语义连贯性。不同K受训权重整体不同；不能仅由图更密或分区更多推导语义联系更好。', '',
      '## 与正式性能、训练身份的对应','',
      '完整119825张test中，K9 mAP71.07533208%、macro-F1 64.48830571%、micro-F1 76.68872316%；历史K16打印为72.0/66.0/76.7%。约差+0.92/+1.51/+0.01pp，最后一项在旧打印精度内基本持平，见[性能对照](ben_grid3_grid4_seed42_20261020.md)。', '',
      '本次取回的训练summary补齐K9 best epoch10、val mAP79.1052%、final30、参数31,366,926、RTX4090 UUID GPU-8734427f-061f-9fdb-97eb-fe9190a11225；train20705.39秒、test178.10秒、父队列20974.62秒。训练峰值allocated约3.62GiB，原始字段保留在审计JSON。历史K16为best13/final30；两次训练不是同一硬件/软件批次，保留单seed历史参考范围。', '',
      '本地逐文件核对返回数据，未执行下载目录中的Python、HTML脚本或pickle；PT张量已核对SHA及容器CRC，未在本地反序列化或重跑神经网络。32项前向观察/重建误差0来自原服务器真实运行报告，受训K16在核对输入上新旧图函数误差也为0。', '',
      '## 可用图注','',
      '中文：BigEarthNet-19四张预定历史验证输入上的受训固定3×3与4×4 GAViT对照，seed42。第一页展示原始RGB输入、实际区域覆盖及特征邻接，箭头为邻居到查询节点；空间边在配套HTML保留。所列概率包含所有阈值0.5翻转和部分未解决错误。第二页显示区域内加权汇总相对均匀权重的变化，红色高于区域内均匀值，蓝色低于；其颜色不是语义标签或类别归因。全部四例保留，不能从案例数估计总体改善。', '',
      'English: Trained fixed 3 x 3 and 4 x 4 GAViT models on four predetermined archived BigEarthNet-19 validation inputs, seed42. Page 1 shows the original RGB inputs, fixed region coverage and captured feature edges directed from neighbor to query; spatial edges are retained in the accompanying HTML. Displayed probabilities include every decision change at threshold 0.5 and selected unresolved errors. Page 2 shows within-region pooling mass relative to uniform pooling, with red above and blue below the uniform value. Colors are not semantic labels or class attribution. All four cases are retained; case counts do not estimate population gains.', '',
      '## 附录：全部76项判断','',
      '以下概率显示到两位小数，原始浮点值保留在captures.json。标签0/1按原CSV；“双方错”不等于4×4新增退化。','']
    verdict={'both_correct':'双方对','both_wrong':'双方错','k16_only_correct':'4×4纠正','k9_only_correct':'4×4新增错误'}
    for case in ANALYSIS['cases']:
        lines += [f'### row{case["row_index"]}','','| 类别 | 标签 | 3×3 % | 4×4 % | 判定 |','|---|---:|---:|---:|---|']
        lines += [f'| {r["name"]} | {r["label"]} | {r["p9"]*100:.2f} | {r["p16"]*100:.2f} | {verdict[r["outcome"]]} |' for r in case['all_class_rows']]
        lines += ['']
    (ROOT/'docs/ben_gridpair42_visual_analysis_20261020.md').write_text('\n'.join(lines))
    print('Analysis written: all four cases, all threshold changes and all 76 label decisions')


if __name__=='__main__': main()
