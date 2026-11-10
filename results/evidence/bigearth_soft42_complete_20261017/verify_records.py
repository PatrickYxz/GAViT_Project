"""Independent arithmetic, preservation and current-document checks for closeout."""
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import re
import statistics

ROOT=Path('/Users/patrick/Projects/GAViT_Project')
WT=ROOT/'.worktrees/sparse-hybrid-4n-top2'
E=Path(__file__).resolve().parent
checks=[]
def read(p):
    return json.loads(p.read_text())
def check(condition, name):
    assert condition, name
    checks.append(name)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

v=read(E/'validation.json')
source=Path(v['source_attachment'])
check(sha(source)==sha(E/'server_return.txt')==v['source_sha256'],'Exact returned source bytes/SHA preserved')
check(len(v['checks'])==31 and v['exit_codes']==dict(queue=0,train=0,test=0),'31 identity/metric checks and three exit0 receipts archived')
s=read(E/'summary.json'); m=s['stages']['gavit_test']['metrics']
check(len(m['per_class_ap_percent'])==19 and math.isclose(statistics.mean(m['per_class_ap_percent'].values()),m['mAP_percent'],abs_tol=1e-8),'Exact19AP mean matches test mAP')
check(s['stages']['gavit_train']['final_epoch']==30 and m['samples']==119825,'Full30 and119825 test')

old=read(E/'before/supervisor_progress_data_20261012.json')
new=read(ROOT/'results/supervisor_progress_data_20261012.json')
for coll in ('rows','selected_four_seed_rows'):
    check(len(old[coll])==len(new[coll]),coll+' length unchanged')
    for a,b in zip(old[coll],new[coll]):
        if a['dataset']=='BEN' and a['seed']==42:
            check(all(b[k]==a[k] for k in a if k not in ('soft','soft_macro_f1','soft_micro_f1')),
                  coll+' BEN42 controls and other old fields preserved')
            check(b['soft']==m['mAP_percent'] and b['soft_macro_f1']==m['macro_f1_percent'] and b['soft_micro_f1']==m['micro_f1_percent'],coll+' BEN42 exact new metrics')
        else:
            check(a==b,coll+' unchanged '+a['dataset']+str(a['seed']))
for key in ('summaries','selected_four_seed_summaries'):
    check(old[key]==new[key],key+' unchanged baseline/fixed aggregates')
for key in ('selected_soft_same_seed_summaries','soft_three_model_same_seed_summaries'):
    check([x for x in old[key] if x['dataset']!='BEN']==[x for x in new[key] if x['dataset']!='BEN'],key+' nonBEN unchanged')

four=read(ROOT/'results/ben_four_seed_three_model_20261017.json')
selected=four['seed_rows']
check([r['seed'] for r in selected]==[42,43,46,47],'Requested cohort exactly42/43/46/47')
for model in ('swin','fixed','soft'):
    values=[r[model] for r in selected]
    check(math.isclose(statistics.mean(values),four['summary'][model]['mean'],abs_tol=1e-12),model+' four-seed mean independently checked')
    check(math.isclose(statistics.stdev(values),four['summary'][model]['sample_sd_ddof1'],abs_tol=1e-12),model+' sampleSD independently checked')
for ref in ('swin','fixed'):
    deltas=[r['soft']-r[ref] for r in selected]
    check(math.isclose(statistics.mean(deltas),four['summary']['paired_soft_minus_'+ref+'_pp']['mean'],abs_tol=1e-12),'Paired soft-'+ref+' independent mean')
check(sum(r['soft']>r['swin'] for r in selected)==2 and sum(r['soft']>r['fixed'] for r in selected)==1,'Soft2positive vsSwin/1positive vsfixed')
check(four['full_common_history_summary']['seeds']==[42,43,45,46,47],'Full available soft history retains45')

before_per=read(E/'before/ben_seed42_seed43_vs_swin_20261013.json')
per=read(ROOT/'results/ben_seed42_seed43_vs_swin_20261013.json')
check(per['per_seed']['43']==before_per['per_seed']['43'],'All seed43 per-class records unchanged')
check(per['per_seed']['42']['fixed']==before_per['per_seed']['42']['fixed'],'All fixed42 per-class records unchanged')
soft=per['per_seed']['42']['soft']['all19_rows']
check(len(soft)==19,'Soft42 exactly19 new class rows')
check(sum(len(block['all19_rows']) for seed in per['per_seed'].values() for block in seed.values() if isinstance(block,dict) and 'all19_rows' in block)==76,'76 total independent class-comparison rows')

job=read(E/'csv_job.json')
original=(E/'before/bigearth_comparison.csv').read_bytes()
ledger=Path(job['official_path']).read_bytes()
check(ledger.startswith(original),'Entire historical official CSV byte prefix unchanged')
prior_rows=list(csv.reader(io.StringIO(original.decode())))
new_rows=list(csv.reader(io.StringIO(ledger.decode())))
check(len(new_rows)==len(prior_rows)+1 and new_rows[:-1]==prior_rows,'Exactlyone official CSV append')
check(new_rows[-1][3]=='42' and math.isclose(float(new_rows[-1][7]),m['mAP_percent'],abs_tol=1e-12),'Official soft42 precise metric row')
display=list(csv.reader(open(job['new_path'])))
check(display==[job['new_headers']]+job['display_rows'],'Selected CSV labels/two-decimal display match source job')
check(sha(ROOT/'results/multiseed_test_results.csv')=='639a9b45b07625d63c1254a32969637c984a996997248622a647f261d43c0988','Original46/47 CSV unchanged SHA')
check(read(E/'spreadsheet_authoring.json')['engine']=='@oai/artifact-tool','Artifact-tool authoring/inspection recorded')

state=read(ROOT/'results/bigearth_soft42_status_20261013.json')
check(state['completion_verified'] and state['formal_test_metrics']==m and state['missing_formal_soft_seed'] is None,'Current state reconciles completed result')
check(state['scheduler_repair_required'] is False and state['scheduler_repair_applied_on_server'] is True,'Worker repair identity confirmed and history retained')
email=(ROOT/'docs/supervisor_email_draft_20261012.md').read_text()
check('Dear Professor Wang,' in email and email.endswith('Xingze\n'),'User email salutation/signoff preserved')
check('seeds 42, 43, 46, and 47' in email and '71.19 ± 1.25' in email,'Advisor seed labels/softfour summary current')
check('soft seed 42 is unavailable' not in email and '43, 44, and 45' not in email,'No stale email seed/missing-result statement')
for name in ('supervisor_email_draft_20261012.md','supervisor_progress_report_20261012.md'):
    text=(ROOT/'docs'/name).read_text()
    check(not re.search(r'deadline|\bddl\b|月底|截止',text,re.I),'No deadline in '+name)
for name in ('ben_four_seed_three_model_20261017.md','ben_seed42_seed43_vs_swin_20261013.md'):
    p=ROOT/'docs'/name
    links=re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text())
    check(all((p.parent/link.split('#')[0]).exists() for link in links if not re.match(r'https?://',link)),'All artifact links resolve '+name)
for diary in (ROOT/'research_diary.md',WT/'research_diary.md'):
    text=diary.read_text()
    check(text.index('BEN soft42正式完成')<text.index('用户报告实例已退还'),'Newest completed entry precedes historical pending entry '+str(diary.parent))

out={'status':'pass','checks':checks,'check_count':len(checks),'scope':'Data/doc reconciliation only; no new GPU execution or independently downloaded server model/prediction bytes','visual_QA':'Four-seed CSV preview checked; ledger value preview regenerated at readable width'}
(E/'report_update_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'pass','checks':len(checks)},ensure_ascii=False))
