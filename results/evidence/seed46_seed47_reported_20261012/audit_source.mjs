import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { Workbook } from '@oai/artifact-tool';

const source = '/Users/patrick/Projects/GAViT_Project/results/multiseed_test_results.csv';
const raw = await fs.readFile(source);
const sourceSha = crypto.createHash('sha256').update(raw).digest('hex');
const workbook = await Workbook.fromCSV(raw.toString('utf8'), { sheetName: 'Reported' });
const matrix = workbook.worksheets.getItem('Reported').getRange('A1:J10').values;
const [header, ...cells] = matrix;
assert.deepEqual(header, ['Dataset','Metric','Model','Test_seed46','Test_seed47','Mean','Std_sample_n-1','Delta_vs_Swin_mean','Source','Notes']);
const rows = cells.map(values => Object.fromEntries(header.map((key,i) => [key,String(values[i] ?? '')])));
assert.equal(rows.length, 9);
assert.equal(new Set(rows.map(row => `${row.Dataset}/${row.Model}`)).size,9);
const mean = values => values.reduce((a,b) => a+b,0)/values.length;
const sampleSd = values => Math.sqrt(values.reduce((sum,v) => sum+(v-mean(values))**2,0)/(values.length-1));
const halfUnit = printed => .5*10**(-(printed.split('.')[1]?.length ?? 0));
const normalized = [];
const arithmetic = [];
for (const row of rows) {
  const pair = ['Test_seed46','Test_seed47'].map(key => row[key] === '' ? null : Number(row[key]));
  if (pair[0] === null || pair[1] === null) {
    assert.deepEqual(pair,[null,null]);
    assert.equal(row.Dataset,'AID');
    assert.equal(row.Model,'GAViT soft_prototype');
    normalized.push({ ...row, values:pair, status:'not_run_user_reported' });
    continue;
  }
  assert(pair.every(v => Number.isFinite(v) && v >=0 && v <=100));
  const baseline = rows.find(v => v.Dataset===row.Dataset && v.Model==='Swin-T Baseline');
  assert(baseline);
  const baselinePair = [Number(baseline.Test_seed46),Number(baseline.Test_seed47)];
  const deltas = pair.map((v,i) => v-baselinePair[i]);
  const recomputed = {mean:mean(pair),sample_sd:sampleSd(pair),mean_delta_vs_swin_pp:mean(deltas)};
  for (const [field,key] of [['Mean','mean'],['Std_sample_n-1','sample_sd'],['Delta_vs_Swin_mean','mean_delta_vs_swin_pp']]) {
    const agrees = Math.abs(Number(row[field])-recomputed[key]) <= halfUnit(row[field])+1e-10;
    assert(agrees,`${row.Dataset}/${row.Model}/${field}`);
    arithmetic.push({dataset:row.Dataset,model:row.Model,field,reported:Number(row[field]),recomputed:recomputed[key],agrees_with_print_precision:agrees});
  }
  const uncertainty = mean(['Test_seed46','Test_seed47'].map(key => halfUnit(row[key])+halfUnit(baseline[key])));
  normalized.push({...row,values:pair,status:'user_reported_artifact_audit_pending',recomputed,
    deltas_vs_swin_pp:deltas,sample_sd_of_deltas_pp:sampleSd(deltas),
    mean_delta_rounding_interval_pp:row.Model==='Swin-T Baseline'?[0,0]:[mean(deltas)-uncertainty,mean(deltas)+uncertainty]});
}
const oldAIDPath='/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2/results/aid_seed42_test_summary.json';
const oldAID=JSON.parse(await fs.readFile(oldAIDPath,'utf8'));
assert.equal(oldAID.test_samples,5000);
const conditionalCountChecks=[];
for(const row of rows.filter(v=>v.Dataset==='AID' && v.Model!=='GAViT soft_prototype')) {
  for(const seed of [46,47]) {
    const printed=row[`Test_seed${seed}`]; const p=Number(printed); const h=halfUnit(printed); const count=oldAID.test_samples;
    const lo=Math.ceil((p-h)*count/100-1e-10); const hi=Math.floor((p+h)*count/100-1e-10);
    conditionalCountChecks.push({seed,model:row.Model,reported_percent:p,assumed_test_samples:count,
      assumed_scope:'historical single-pass OA over5000 test; new run count/metric not confirmed',
      possible_integer_correct_counts:lo<=hi?[lo,hi]:[],consistent_if_historical_scope:lo<=hi});
  }
}
const pairs=[];
for(const dataset of ['AID','NWPU-RESISC45','BigEarthNet-19']) {
  for(const seed of [46,47]) {
    const i=seed===46?0:1;
    const models=Object.fromEntries(normalized.filter(v=>v.Dataset===dataset).map(v=>[v.Model,v]));
    pairs.push({dataset,seed,metric:models['Swin-T Baseline'].Metric,
      swin:models['Swin-T Baseline'].values[i],fixed:models['GAViT (fixed grouping)'].values[i],soft:models['GAViT soft_prototype'].values[i],
      fixed_minus_swin_pp:models['GAViT (fixed grouping)'].deltas_vs_swin_pp[i],
      soft_minus_swin_pp:models['GAViT soft_prototype'].deltas_vs_swin_pp?.[i]??null,
      soft_minus_fixed_pp:models['GAViT soft_prototype'].values[i]===null?null:models['GAViT soft_prototype'].values[i]-models['GAViT (fixed grouping)'].values[i]});
  }
}
assert(pairs.every(v=>v.fixed_minus_swin_pp>0));
assert.equal(conditionalCountChecks.filter(v=>!v.consistent_if_historical_scope).length,2);
assert.equal(crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex'),sourceSha);
const result={
  status:'reported_numbers_recomputed_original_artifacts_pending',
  source,source_sha256:sourceSha,seeds:[46,47],
  hardware:{model_user_declared:'NVIDIA GeForce RTX4090',provenance:'direct user message in this chat',
    raw_hardware_record_seen:false,gpu_uuid:null,physical_gpu_pairing_verified:false},
  spreadsheet_read_only:true,arithmetic_checks:arithmetic,rows:normalized,per_seed_comparisons:pairs,
  conditional_aid_count_checks:conditionalCountChecks,old_aid_test_scope_source:oldAIDPath,
  formal_artifact_verification_complete:false,
  pending:['original test logs/run.json or summary.json/run.exit','AID actual test count and whether metric is single-pass OA',
    'code commit, model/grouping/edges and training/selection recipe','split/pretrained/checkpoint SHA and train/test checkpoint match',
    'per-stage GPU UUID/software/precision and hardware pairing','macro-F1/micro-F1/per-class metrics and resources'],
  interpretation_limits:['arithmetic checks validate numbers in the CSV, not GPU training or test completion',
    'two new seeds do not alone establish stable superiority; historical results retained',
    'BEN mean soft-minus-Swin+.05pp has a conservative rounding interval[-.05,+.15]pp and is not evidence of equivalence',
    '5000-test OA incompatibility is conditional; different sample counts or metrics may explain it',
    'do not combine with historical seeds until configuration and protocol identity are checked']
};
await fs.writeFile('/private/tmp/gavit-seed46-47-audit/audit.json',JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({status:result.status,arithmetic_checks:arithmetic.length,
  pairs,conditional_aid_count_issues:conditionalCountChecks.filter(v=>!v.consistent_if_historical_scope),
  model_statistics:normalized.map(v=>({dataset:v.Dataset,model:v.Model,...v.recomputed,mean_delta_rounding_interval_pp:v.mean_delta_rounding_interval_pp}))},null,2));
