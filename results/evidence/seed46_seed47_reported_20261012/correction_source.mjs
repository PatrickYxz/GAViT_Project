import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {Workbook} from '@oai/artifact-tool';

const root='/Users/patrick/Projects/GAViT_Project';
const wt=path.join(root,'.worktrees/sparse-hybrid-4n-top2');
const evidence=path.join(root,'results/evidence/seed46_seed47_reported_20261012');
const source=path.join(root,'results/multiseed_test_results.csv');
const auditPath=path.join(root,'results/seed46_seed47_audit_20261012.json');
const auditMdPath=path.join(root,'results/seed46_seed47_audit_20261012.md');
const sha=v=>crypto.createHash('sha256').update(v).digest('hex');
const mean=v=>v.reduce((a,b)=>a+b,0)/v.length;
const sd=v=>Math.sqrt(v.reduce((a,b)=>a+(b-mean(v))**2,0)/(v.length-1));
const num=(v,n=3)=>v.toFixed(n);
const signed=(v,n=3)=>(v>=0?'+':'')+v.toFixed(n);
const halfUnit=v=>.5*10**(-(v.split('.')[1]?.length??0));
const loadJson=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const snapshot=async (p,value)=>{
  try {await fs.writeFile(p,value,{flag:'wx'});}
  catch(e){if(e.code!=='EEXIST')throw e;assert.deepEqual(await fs.readFile(p),Buffer.from(value));}
};

const raw=await fs.readFile(source);
const original=await fs.readFile(path.join(evidence,'reported_results.csv'));
assert.equal(sha(original),'907592fb63b954fe9cbae87bde100b0f57efaef44302d751985765ca3f360c9b');
assert.deepEqual(raw,original);
const prior=await loadJson(auditPath);
await snapshot(path.join(evidence,'audit_before_exact_AID_correction.json'),await fs.readFile(auditPath));
await snapshot(path.join(evidence,'audit_before_exact_AID_correction.md'),await fs.readFile(auditMdPath));

const workbook=await Workbook.fromCSV(raw.toString('utf8'),{sheetName:'Reported'});
const sheet=workbook.worksheets.getItem('Reported');
const header=sheet.getRange('A1:J1').values[0];
assert.deepEqual(header,['Dataset','Metric','Model','Test_seed46','Test_seed47','Mean','Std_sample_n-1','Delta_vs_Swin_mean','Source','Notes']);
assert.equal(Number(sheet.getRange('D3').values[0][0]),97.13);
assert.equal(Number(sheet.getRange('E2').values[0][0]),95.73);
sheet.getRange('D2:E3').values=[[96.56,95.72],[97.12,96.08]];
sheet.getRange('F2:H3').formulas=[
  ['=AVERAGE(D2:E2)','=SQRT((D2-E2)^2/2)','=F2-F2'],
  ['=AVERAGE(D3:E3)','=SQRT((D3-E3)^2/2)','=F3-F2']
];
sheet.getRange('I2:J3').values=[
  ['user-confirmed correction 2026-10-12','Swin47 corrected 95.73 to 95.72 on explicit user confirmation; original artifacts and secondary metrics pending'],
  ['user-confirmed correction 2026-10-12','Fixed46 corrected 97.13 to 97.12 on explicit user confirmation; per-seed delta vs Swin +0.56/+0.36; original configuration artifacts pending']
];
workbook.recalculate();
const numeric=sheet.getRange('D2:H3').values;
assert(Math.abs(numeric[0][2]-96.14)<1e-10);
assert(Math.abs(numeric[1][2]-96.6)<1e-10);
assert(Math.abs(numeric[0][3]-sd([96.56,95.72]))<1e-10);
assert(Math.abs(numeric[1][3]-sd([97.12,96.08]))<1e-10);

const csvField=v=>/[",\r\n]/.test(v)?'"'+v.replaceAll('"','""')+'"':v;
const lines=raw.toString('utf8').split('\n');
for(let i=0;i<2;i++){
  const row=sheet.getRange(`A${i+2}:J${i+2}`).values[0].map(v=>String(v??''));
  for(const col of [3,4])row[col]=Number(row[col]).toFixed(2);
  for(const col of [5,6])row[col]=Number(row[col]).toFixed(3);
  row[7]=i===0?'0.000':signed(Number(row[7]));
  lines[i+1]=row.map(csvField).join(',');
}
const corrected=lines.join('\n');
assert.deepEqual(lines.slice(3),raw.toString('utf8').split('\n').slice(3));

const checked=await Workbook.fromCSV(corrected,{sheetName:'Corrected'});
const matrix=checked.worksheets.getItem('Corrected').getRange('A1:J10').values;
const rows=matrix.slice(1).map(v=>Object.fromEntries(header.map((k,i)=>[k,String(v[i]??'')])));
const normalized=[],arithmetic=[];
for(const row of rows){
  const values=['Test_seed46','Test_seed47'].map(k=>row[k]===''?null:Number(row[k]));
  if(values[0]===null){assert.deepEqual(values,[null,null]);normalized.push({...row,values,status:'not_run_user_reported'});continue;}
  const baseline=rows.find(r=>r.Dataset===row.Dataset&&r.Model==='Swin-T Baseline');
  const baseValues=[Number(baseline.Test_seed46),Number(baseline.Test_seed47)];
  const deltas=values.map((v,i)=>v-baseValues[i]);
  const recomputed={mean:mean(values),sample_sd:sd(values),mean_delta_vs_swin_pp:mean(deltas)};
  for(const [field,key] of [['Mean','mean'],['Std_sample_n-1','sample_sd'],['Delta_vs_Swin_mean','mean_delta_vs_swin_pp']]){
    const agrees=Math.abs(Number(row[field])-recomputed[key])<=halfUnit(row[field])+1e-10;
    assert(agrees,`${row.Dataset}/${row.Model}/${field}`);
    arithmetic.push({dataset:row.Dataset,model:row.Model,field,reported:Number(row[field]),recomputed:recomputed[key],agrees_with_print_precision:agrees});
  }
  const uncertainty=mean(['Test_seed46','Test_seed47'].map(k=>halfUnit(row[k])+halfUnit(baseline[k])));
  normalized.push({...row,values,status:row.Dataset==='AID'?'user_confirmed_corrected_scores_original_artifacts_pending':'user_reported_artifact_audit_pending',recomputed,deltas_vs_swin_pp:deltas,sample_sd_of_deltas_pp:sd(deltas),mean_delta_rounding_interval_pp:row.Model==='Swin-T Baseline'?[0,0]:[mean(deltas)-uncertainty,mean(deltas)+uncertainty]});
}
assert.equal(arithmetic.length,24);
const comparisons=[];
for(const dataset of ['AID','NWPU-RESISC45','BigEarthNet-19'])for(const [i,seed]of [46,47].entries()){
  const get=model=>normalized.find(r=>r.Dataset===dataset&&r.Model===model).values[i];
  const swin=get('Swin-T Baseline'),fixed=get('GAViT (fixed grouping)'),soft=get('GAViT soft_prototype');
  comparisons.push({dataset,seed,metric:rows.find(r=>r.Dataset===dataset).Metric,swin,fixed,soft,fixed_minus_swin_pp:fixed-swin,soft_minus_swin_pp:soft===null?null:soft-swin,soft_minus_fixed_pp:soft===null?null:soft-fixed,score_correction_pending:false});
}
assert(comparisons.every(v=>v.fixed_minus_swin_pp>0));
const countChecks=[];
for(const r of normalized.filter(v=>v.Dataset==='AID'&&v.values[0]!==null))for(const [i,seed]of [46,47].entries()){
  const correct=r.values[i]*5000/100;
  assert(Math.abs(correct-Math.round(correct))<1e-8);
  countChecks.push({seed,model:r.Model,reported_percent:r.values[i],test_samples_user_confirmed:5000,derived_integer_correct:Math.round(correct),consistent_with_single_pass_oa:true,count_is_derived_not_independently_returned:true});
}
const corrections=[{seed:46,model:'GAViT (fixed grouping)',before_percent:97.13,confirmed_percent:97.12},{seed:47,model:'Swin-T Baseline',before_percent:95.73,confirmed_percent:95.72}];
const audit={
  status:'user_confirmed_AID_corrections_recorded_arithmetic_pass_original_artifacts_pending',
  source,source_sha256:sha(corrected),original_reported_source:path.join(evidence,'reported_results.csv'),original_reported_sha256:sha(original),seeds:[46,47],
  hardware:prior.hardware,user_clarification:{...prior.user_clarification,exact_AID_corrections_user_confirmed:true,confirmation_text:'应该是97.12以及95.72。你重新记录，然后我们可以开始整理内容汇报给导师了。'},
  spreadsheet_read_only:false,source_corrected_on_explicit_user_instruction:true,
  corrections,arithmetic_checks:arithmetic,rows:normalized,per_seed_comparisons:comparisons,aid_count_compatibility_checks:countChecks,
  old_aid_test_scope_source:prior.old_aid_test_scope_source,formal_artifact_verification_complete:false,
  evidence_before_correction:{json:'evidence/seed46_seed47_reported_20261012/audit_before_exact_AID_correction.json',markdown:'evidence/seed46_seed47_reported_20261012/audit_before_exact_AID_correction.md'},
  pending:['original test logs/run.json or summary.json/run.exit to verify full test completion and exact metrics','code commit, architecture, training/selection configuration','split/pretrained/checkpoint SHA and train/test checkpoint match','per-stage GPU UUID/software/precision and physical-card pairing','macro-F1/micro-F1/per-class metrics and resources'],
  interpretation_limits:['exact AID values are user-confirmed, not uniquely inferred from accuracy granularity','arithmetic checks validate reported values, not GPU training or test completion','correct counts are OA-derived compatibility checks, not returned predictions','two new seeds do not establish general stability; historical positive and negative results retained','BEN mean soft-minus-Swin +.05pp has a conservative rounding interval[-.05,+.15]pp','pooled history has hardware/code/epoch and evidence differences; descriptive means do not establish significance or pure seed variance']
};

const datasetShort=d=>d==='BigEarthNet-19'?'BEN':d==='NWPU-RESISC45'?'NWPU':d;
const perSeedTable=comparisons.map(v=>`| ${datasetShort(v.dataset)} | ${v.seed} | ${num(v.swin,v.dataset==='BigEarthNet-19'?1:2)} | ${num(v.fixed,v.dataset==='BigEarthNet-19'?1:2)} | ${v.soft===null?'未运行':num(v.soft,v.dataset==='BigEarthNet-19'?1:2)} | ${signed(v.fixed_minus_swin_pp,2)} | ${v.soft===null?'—':signed(v.soft_minus_swin_pp,2)} |`).join('\n');
const batchRows=['AID','NWPU-RESISC45','BigEarthNet-19'].map(d=>{
  const rs=normalized.filter(v=>v.Dataset===d);
  const stat=model=>{const r=rs.find(v=>v.Model===model);return r.recomputed?`${num(r.recomputed.mean)}±${num(r.recomputed.sample_sd)}`:'未运行';};
  return `| ${datasetShort(d)} | ${stat('Swin-T Baseline')} | ${stat('GAViT (fixed grouping)')} | ${stat('GAViT soft_prototype')} | ${signed(rs[1].recomputed.mean_delta_vs_swin_pp)} | ${rs[2].recomputed?signed(rs[2].recomputed.mean_delta_vs_swin_pp):'—'} |`;
}).join('\n');
const auditMd=`# seed46/47回传结果核对（2026-10-12，更正后）

按用户明确确认，AID固定seed46由97.13%更正为**97.12%**，Swin seed47由95.73%更正为**95.72%**。其余14个指标值不变；[当前数值表](multiseed_test_results.csv)已更新两项分数、对应均值/样本SD及AID配对差说明。24项算术核对通过。更正来源是用户确认，不是从OA步长猜出的成绩。

[原回传CSV](evidence/seed46_seed47_reported_20261012/reported_results.csv)、[更正前JSON](evidence/seed46_seed47_reported_20261012/audit_before_exact_AID_correction.json)和[更正前说明](evidence/seed46_seed47_reported_20261012/audit_before_exact_AID_correction.md)保留。原CSV SHA256：\`${sha(original)}\`；当前CSV SHA256：\`${sha(corrected)}\`。

用户确认本批使用RTX4090、配置与测试数量不变、Featurize物理GPU UUID不同于此前；AID/NWPU/BEN test数量为5000/25200/119825。具体UUID、原始summary/run/test/exit及权重身份尚未回传，当前属于“用户确认数值，原始实验工件待核验”。AID软分配未运行。

## 各seed分数

单位为%；AID/NWPU为test OA，BEN为test mAP，差值单位pp。

| 数据集 | seed | Swin | 固定GAViT | soft | 固定−Swin pp | soft−Swin pp |
|---|---:|---:|---:|---:|---:|---:|
${perSeedTable}

## 两seed汇总

样本SD使用n−1。额外小数只用于复算，不增加BEN原打印精度。

| 数据集 | Swin均值±SD % | 固定均值±SD % | soft均值±SD % | 固定−Swin均值 pp | soft−Swin均值 pp |
|---|---:|---:|---:|---:|---:|
${batchRows}

固定版六组配对主指标均高于Swin。soft在NWPU两次低于Swin，在BEN一次升一次降；两数据集上均低于固定版，soft−fixed两seed均值分别−0.530和−0.600pp。BEN soft−Swin均值+0.050pp的保守舍入区间为[−0.05,+0.15]pp，方向不确定，不能写成已证实提升或等效。跨数据集的六组不是同一任务的六次独立重复。

## AID更正与计数相容性

5000图的单次OA步长为0.02pp；更正后四个AID分数均与该步长相容。按确认的OA计算，Swin46/47对应4828/4786正确，固定46/47对应4856/4804正确。这些整数是OA派生值，原始预测/混淆矩阵尚未独立回传。固定相对Swin差为+0.56/+0.36pp，均值+0.460pp。此前最近合法候选仅作为历史推断保留，当前以用户明确更正为准。

## 原始记录尚缺的字段

- summary.json/run.json、run.exit与test日志：完整测试退出、实际数量、精确主指标及辅助指标。
- commit、grouping/区域数/图边/反馈配置、校准/refit或best-test流程、最佳/最终轮。
- split/预训练/checkpoint SHA及训练与测试权重一致性。
- 各阶段4090 UUID、软件版本、AMP/TF32与同seed物理卡配对。
- macro-F1、BEN micro-F1/阈值、逐类数据及资源。

历史正式表保持其证据范围；新分数纳入[导师进度汇报](../docs/supervisor_progress_report_20261012.md)的“回传待核验”部分，全部正负seed保留。未追加训练。完整复算与更正来源见[JSON](seed46_seed47_audit_20261012.json)。
`;
await fs.writeFile(source,corrected);
await fs.writeFile(auditPath,JSON.stringify(audit,null,2)+'\n');
await fs.writeFile(auditMdPath,auditMd);
const preview=await checked.render({sheetName:'Corrected',range:'D2:H3',format:'png',scale:2});
await fs.writeFile('/private/tmp/gavit-seed46-47-audit/after_correction.png',new Uint8Array(await preview.arrayBuffer()));
await snapshot(path.join(evidence,'correction_source.mjs'),await fs.readFile(new URL(import.meta.url)));
console.log(JSON.stringify({status:audit.status,source_sha256:audit.source_sha256,arithmetic_checks:arithmetic.length,AID:normalized.filter(v=>v.Dataset==='AID').map(v=>({model:v.Model,values:v.values,stats:v.recomputed})),unchanged_non_AID_rows:true},null,2));
