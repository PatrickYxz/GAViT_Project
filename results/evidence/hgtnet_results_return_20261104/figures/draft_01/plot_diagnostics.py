"""Reproducible vector/raster diagnostics for a single independent HGTNet run.

Run with Python3.12 and PYTHONPATH=/private/tmp/dfagcn-plotdeps-20261031.
All figure data derive from the returned raw predictions/history, without GPU.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import os
import platform

os.environ.setdefault('MPLCONFIGDIR','/private/tmp/hgtnet-mpl-20261104')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

OUT=Path(__file__).resolve().parent
FILES=OUT.parent/'files'
test=json.loads((FILES/'test.json').read_text())
history=json.loads((FILES/'history.json').read_text())
summary=json.loads((FILES/'summary.json').read_text())
binding=json.loads((FILES/'binding.json').read_text())
names=binding['class_names'];lookup={name:i for i,name in enumerate(names)}
assert len(names)==45 and names==sorted(names) and len(set(names))==45
truth=[lookup[p.split('/')[0]] for p in test['paths']]
assert truth==test['labels'] and len(truth)==len(test['predictions'])==25200
cm=np.zeros((45,45),dtype=np.int64)
for actual,predicted in zip(truth,test['predictions']):
    assert type(predicted) is int and 0<=predicted<45
    cm[actual,predicted]+=1
assert cm.tolist()==test['metrics']['confusion_matrix']
support=cm.sum(axis=1);predicted_count=cm.sum(axis=0);tp=cm.diagonal()
assert (support==560).all() and cm.sum()==25200 and tp.sum()==23718
recall=100*tp/support;f1=200*tp/(support+predicted_count)
oa=100*tp.sum()/cm.sum();macro_f1=float(f1.mean())
assert math.isclose(float(oa),test['metrics']['accuracy_percent'],rel_tol=0,abs_tol=1e-10)
assert math.isclose(macro_f1,test['metrics']['macro_f1_percent'],rel_tol=0,abs_tol=1e-10)
for i,row in enumerate(test['metrics']['per_class']):
    assert row['class']==names[i] and row['support']==560 and row['correct']==int(tp[i])
    assert math.isclose(row['accuracy_percent'],float(recall[i]),rel_tol=0,abs_tol=1e-10)
    assert math.isclose(row['f1_percent'],float(f1[i]),rel_tol=0,abs_tol=1e-10)
cal=history['calibration'];refit=history['refit'];selected=summary['selected_epoch']
assert len(cal)==100 and len(refit)==95 and selected==summary['best_epoch']==95
assert [r['epoch'] for r in cal]==list(range(1,101))
assert [r['epoch'] for r in refit]==list(range(1,96))
best=max(cal,key=lambda r:r['val_correct'])
assert best['epoch']==95 and cal[-1]['epoch']==100
for row in cal:
    m=np.asarray(row['val_metrics']['confusion_matrix'],dtype=np.int64)
    assert m.shape==(45,45) and m.sum()==1260 and (m.sum(axis=1)==28).all()
    assert int(np.trace(m))==row['val_correct']
    assert math.isclose(100*np.trace(m)/1260,row['val_metrics']['accuracy_percent'],abs_tol=1e-10)

BLUE='#23628A';ORANGE='#C46B19';TEAL='#168377';PURPLE='#7A5299';GRAY='#6F7780'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
    'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
    'grid.alpha':0.20,'grid.linewidth':0.6,'savefig.facecolor':'white','figure.facecolor':'white',
    'svg.fonttype':'none','pdf.fonttype':42})
artifacts=[];scope='Independent HGTNet, NWPU-RESISC45, seed42; one trained model; descriptive diagnostics only.'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(fig,stem,description):
    for ext in ['png','svg','pdf']:
        path=OUT/(stem+'.'+ext)
        if path.exists():raise FileExistsError('Do not overwrite a delivered figure: '+str(path))
        kwargs=dict(bbox_inches='tight',pad_inches=.16)
        if ext=='png':kwargs['dpi']=300
        if ext=='pdf':kwargs['metadata']={'Title':description,'Subject':scope,'Author':'GAViT offline analysis'}
        fig.savefig(path,**kwargs)
        artifacts.append(dict(path=str(path),filename=path.name,bytes=path.stat().st_size,sha256=sha(path),description=description))
    plt.close(fig)

# One four-panel figure: retain all calibration/refit epochs and both best/final.
fig,axs=plt.subplots(2,2,figsize=(13.2,9.1),layout='constrained')
fig.suptitle('Independent HGTNet: calibration and fresh refit',fontsize=15,fontweight='bold')
xe=[r['epoch'] for r in cal];xr=[r['epoch'] for r in refit]
axs[0,0].plot(xe,[r['train_loss'] for r in cal],color=BLUE,label='Online training loss',lw=1.6)
axs[0,0].plot(xe,[r['val_loss'] for r in cal],color=ORANGE,label='Validation loss',lw=1.6)
axs[0,0].set(title='(a) Calibration loss: 5,040 train / 1,260 validation',xlabel='Calibration epoch',ylabel='Cross entropy (log scale)',yscale='log',xlim=(1,101))
axs[0,0].legend(loc='lower left')
axs[0,1].plot(xe,[r['train_accuracy_percent'] for r in cal],color=BLUE,label='Online training accuracy',lw=1.5)
axs[0,1].plot(xe,[r['val_metrics']['accuracy_percent'] for r in cal],color=ORANGE,label='Validation OA',lw=1.7)
axs[0,1].plot(xe,[r['val_metrics']['macro_f1_percent'] for r in cal],color=TEAL,label='Validation macro-F1',lw=1.4,ls='--')
axs[0,1].set(title='(b) Calibration: validation selects the epoch budget',xlabel='Calibration epoch',ylabel='Percent',xlim=(1,104),ylim=(40,102))
axs[0,1].scatter([95,100],[cal[94]['val_metrics']['accuracy_percent'],cal[99]['val_metrics']['accuracy_percent']],s=35,c=[PURPLE,GRAY],zorder=5)
axs[0,1].annotate('Best 95: 93.095% OA',xy=(95,cal[94]['val_metrics']['accuracy_percent']),xytext=(48,79),fontsize=9,color=PURPLE,arrowprops=dict(arrowstyle='-',color=PURPLE,lw=.8))
axs[0,1].annotate('Final 100: 92.460% OA',xy=(100,cal[99]['val_metrics']['accuracy_percent']),xytext=(48,73),fontsize=9,color=GRAY,arrowprops=dict(arrowstyle='-',color=GRAY,lw=.8))
axs[0,1].legend(loc='lower right')
for ax in axs[0]:
    ax.axvline(95,color=PURPLE,ls=':',lw=1.2,zorder=1)
    ax.axvline(100,color=GRAY,ls=':',lw=.9,zorder=1)
axs[1,0].plot(xr,[r['train_loss'] for r in refit],color=BLUE,lw=1.6)
axs[1,0].set(title='(c) Fresh refit loss: 6,300 pooled train images',xlabel='Refit epoch',ylabel='Cross entropy (log scale)',yscale='log',xlim=(1,95))
axs[1,1].plot(xr,[r['train_accuracy_percent'] for r in refit],color=BLUE,lw=1.6)
axs[1,1].set(title='(d) Fresh refit: fixed 95-epoch budget',xlabel='Refit epoch',ylabel='Online training accuracy (%)',xlim=(1,95),ylim=(40,102))
fig.get_layout_engine().set(rect=(0,.065,1,.90))
fig.text(.5,.027,'Training accuracy is accumulated while weights update; validation uses the epoch-end model.\nRefit starts from the same initial weights and has no validation curve; final test OA = 94.119%, macro-F1 = 94.105%.',ha='center',va='center',fontsize=9,color='#40464C')
save(fig,'hgtnet_seed42_training_history','All100 calibration epochs and95 fresh-refit epochs, with selected95 and final100 retained')

# All45 classes, ascending recall; no weak/strong classes are omitted.
order=sorted(range(45),key=lambda i:(recall[i],names[i]));y=np.arange(45)
fig,ax=plt.subplots(figsize=(10.6,13.8),layout='constrained')
fig.suptitle('HGTNet test performance for every class',fontsize=15,fontweight='bold')
ax.barh(y,recall[order],height=.65,color=BLUE,alpha=.87,label='Recall / class accuracy')
ax.scatter(f1[order],y,color=ORANGE,s=24,marker='D',label='F1',zorder=4)
ax.set_yticks(y,[names[i].replace('_',' ') for i in order],fontsize=9)
ax.invert_yaxis();ax.set(xlim=(0,108),xlabel='Percent',ylabel='All45 classes, ordered by recall')
ax.set_xticks([0,20,40,60,80,100]);ax.grid(axis='y',visible=False)
for yy,i in zip(y,order):ax.text(float(recall[i])+.75,yy,f'{recall[i]:.1f}',va='center',fontsize=8,color='#1B2731')
ax.axvline(float(oa),color=GRAY,lw=.9,ls='--',label='Overall OA (94.119%)')
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.59,.95),frameon=False,ncol=3,fontsize=8.5)
fig.get_layout_engine().set(rect=(0,.025,1,.91))
fig.text(.5,.01,'560 test images per class; OA 94.119%, macro-F1 94.105%. All classes and their scores are retained.',ha='center',fontsize=9,color='#40464C')
save(fig,'hgtnet_seed42_all_class_performance','Recall andF1 for all45classes; sorted ascendingrecall; 560testimages/class')

# Complete row-normalized matrix, original lexical class order.
normalized=100*cm/support[:,None]
assert np.max(np.abs(normalized.sum(axis=1)-100))<1e-10
fig,ax=plt.subplots(figsize=(15.6,14.3),layout='constrained')
fig.suptitle('HGTNet: complete test confusion matrix',fontsize=16,fontweight='bold')
image=ax.imshow(normalized,cmap='Blues',vmin=0,vmax=100,interpolation='nearest',aspect='equal')
labels=[n.replace('_',' ') for n in names]
ax.set_xticks(np.arange(45),labels,rotation=65,ha='right',rotation_mode='anchor',fontsize=8.8)
ax.set_yticks(np.arange(45),labels,fontsize=8.8)
ax.set(xlabel='Predicted class',ylabel='True class');ax.grid(False)
bar=fig.colorbar(image,ax=ax,fraction=.035,pad=.018,shrink=.82)
bar.set_label('Percent of each true class');bar.set_ticks([0,20,40,60,80,100])
fig.get_layout_engine().set(rect=(0,.035,1,.945))
fig.text(.5,.014,'Rows are true classes; columns are predicted classes. All45 x45 cells and diagonal retained.\nRow-normalized, 560 images per row, total25,200; OA94.119%, macro-F194.105%.',ha='center',va='center',fontsize=10,color='#40464C')
save(fig,'hgtnet_seed42_confusion_matrix','Complete45x45 row-normalized test confusion matrix withdiagonal andalllabels retained')

# Top10 bidirectional error pairs; preserve both directions (includingzeros).
pairs=[]
for i in range(45):
    for j in range(i+1,45):
        total=int(cm[i,j]+cm[j,i])
        if total:pairs.append(dict(first=names[i],second=names[j],first_to_second=int(cm[i,j]),second_to_first=int(cm[j,i]),total=total))
pairs.sort(key=lambda r:(-r['total'],r['first'],r['second']));top=pairs[:10]
fig,ax=plt.subplots(figsize=(12.2,6.9),layout='constrained')
fig.suptitle('HGTNet: largest bidirectional test confusions',fontsize=15,fontweight='bold')
y=np.arange(10);left=np.array([p['first_to_second'] for p in top]);right=np.array([p['second_to_first'] for p in top]);limit=1.24*max(int(left.max()),int(right.max()))+2
ax.barh(y,-left,height=.66,color=BLUE,label='First class -> second class')
ax.barh(y,right,height=.66,color=ORANGE,label='Second class -> first class')
ax.set_yticks(y,[p['first'].replace('_',' ')+' / '+p['second'].replace('_',' ') for p in top],fontsize=9.5)
ax.invert_yaxis();ax.axvline(0,color='#29343D',lw=.9)
ax.set(xlim=(-limit,limit),xlabel='Misclassified images (absolute count; side indicates direction)',ylabel='First / second class')
ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{abs(value):g}'));ax.grid(axis='y',visible=False)
for k,(a,b) in enumerate(zip(left,right)):
    ax.text(-float(a)-.7,k,str(a),ha='right',va='center',fontsize=10,color=BLUE)
    ax.text(float(b)+.7,k,str(b),ha='left',va='center',fontsize=10,color=ORANGE)
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.66,.93),frameon=False,ncol=2)
fig.get_layout_engine().set(rect=(0,.07,1,.83))
fig.text(.5,.028,'Top10 unordered pairs ranked by totalerrors; both directions shown.\nAll1,482 errors and the complete confusion matrix are retained in the source analysis; these are descriptive model errors.',ha='center',va='center',fontsize=9,color='#40464C')
save(fig,'hgtnet_seed42_top10_bidirectional_confusions','Top10 mutualconfusionpairs, bothdirections retained; completecounts available inrawmatrix')

data=dict(class_names=names,confusion_matrix=cm.tolist(),per_class=[dict(class_name=n,recall_percent=float(recall[i]),f1_percent=float(f1[i]),support=int(support[i]),correct=int(tp[i])) for i,n in enumerate(names)],
    top10_mutual_confusion_pairs=top,all_mutual_pairs_count=len(pairs),selected_epoch=selected,
    calibration_epochs=len(cal),refit_epochs=len(refit),final_calibration_metrics=cal[-1]['val_metrics'],
    best_calibration_metrics=cal[94]['val_metrics'],oa_percent=float(oa),macro_f1_percent=macro_f1)
(OUT/'plot_data.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
manifest=dict(source_files={str((FILES/n).relative_to(OUT.parents[3])):sha(FILES/n) for n in ['test.json','history.json','summary.json','binding.json']},
    script_sha256=sha(__file__),python=platform.python_version(),matplotlib=matplotlib.__version__,numpy=np.__version__,
    checks=dict(samples=25200,correct=23718,errors=1482,classes=45,support_per_class=560,
        recomputed_oa_percent=float(oa),recomputed_macro_f1_percent=macro_f1,confusion_all_cells_match=True,
        all_class_scores_match=True,all100_calibration_and95_refit_epochs_retained=True,selected95_final100_retained=True,
        normalized_row_sum_max_error=float(np.max(np.abs(normalized.sum(axis=1)-100)))),
    artifacts=artifacts,interpretation_scope=scope,
    training_metric_note='Online training aggregates changingweights; validation is epochend evaluation; refit no validation; no causaldiagnosis fromcurve shapes')
(OUT/'plot_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(artifacts=len(artifacts),metrics=manifest['checks'],top_pair=top[0]),indent=2))
