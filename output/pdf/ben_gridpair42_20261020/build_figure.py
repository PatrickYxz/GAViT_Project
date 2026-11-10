"""Two-page scientific figure from audited true captures; no model rerun."""
from pathlib import Path
import hashlib
import json
import math
import shutil

from PIL import Image
import reportlab
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
EVIDENCE=ROOT/'results/evidence/ben_gridpair42_visual_return_20261020'
SOURCE=EVIDENCE/'server_return'
DATA=json.loads((SOURCE/'captures.json').read_text())
ANALYSIS=json.loads((EVIDENCE/'case_analysis.json').read_text())
WIDTH,HEIGHT=180*72/25.4,830
SIZE,GAP=140,15
LEFT=(WIDTH-3*SIZE-2*GAP)/2
INK='#23313d'
COLORS=['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#46f0f0','#f032e6',
        '#bcf60c','#fabebe','#008080','#e6beff','#9a6324','#fffac8','#800000','#aaffc3']
FONT=Path(reportlab.__file__).resolve().parent/'fonts'
pdfmetrics.registerFont(TTFont('GridReg',str(FONT/'Vera.ttf')))
pdfmetrics.registerFont(TTFont('GridBold',str(FONT/'VeraBd.ttf')))
# Every threshold flip is shown. Extra entries are identified as unchanged
# errors rather than corrections, keeping the coastal near miss visible.
FOCUS={97389:[4],14680:[11,4],28450:[13,18],4439:[0,14]}
SHORT={0:'Urban fabric',4:'Pastures',11:'Natural grassland / sparse vegetation',
       13:'Transitional woodland / shrub',14:'Beaches / dunes / sands',18:'Marine waters'}
VERDICT={'k16_only_correct':'corrected','k9_only_correct':'new miss',
         'both_wrong':'still wrong','both_correct':'both correct'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(c,x,y,value,size=7,bold=False,align='left',color=INK):
    face='GridBold' if bold else 'GridReg'
    assert pdfmetrics.stringWidth(value,face,size)<WIDTH-2*LEFT+1,value
    c.setFillColor(HexColor(color)); c.setFont(face,size)
    {'left':c.drawString,'center':c.drawCentredString,'right':c.drawRightString}[align](x,y,value)


def boxes(assignments,k):
    out=[]
    for region in range(k):
        cells=[i for i,r in enumerate(assignments) if r==region]
        xs,ys=[i%7*32 for i in cells],[i//7*32 for i in cells]
        x,y=min(xs),min(ys); w,h=max(xs)-x+32,max(ys)-y+32
        out.append((x,y,w,h))
    return out


def panel(c,x,y,image,record,mode,placements,drawn):
    c.drawImage(str(image),x,y,SIZE,SIZE)
    placements.append(dict(source=str(image.relative_to(ROOT)),box=[x,y,SIZE,SIZE]))
    if record is None: return
    k=record['num_regions']; a=record['assignments']; b=boxes(a,k)
    c.saveState(); c.translate(x,y); c.scale(SIZE/224,SIZE/224)
    if mode=='pooling':
        for i,region in enumerate(a):
            ratio=record['pooling_weights'][i][region]*a.count(region)
            c.setFillColor(HexColor('#dc2626' if ratio>=1 else '#2563eb'))
            c.setFillAlpha(min(.65,abs(ratio-1)*.35))
            c.rect(i%7*32,224-i//7*32-32,32,32,stroke=0,fill=1)
        c.setFillAlpha(1)
    for region,(rx,ry,w,h) in enumerate(b):
        if mode=='feature':
            c.setFillColor(HexColor(COLORS[region])); c.setFillAlpha(.12)
            c.rect(rx,224-ry-h,w,h,stroke=0,fill=1); c.setFillAlpha(1)
        c.setLineWidth(.65); c.setStrokeColor(HexColor('#ffffff'))
        c.rect(rx,224-ry-h,w,h,stroke=1,fill=0)
    centers=[(rx+w/2,224-ry-h/2) for rx,ry,w,h in b]
    if mode=='feature':
        for (s,t),kind in zip(record['edges'],record['edge_kinds']):
            if kind!='feature': continue
            x1,y1=centers[s]; x2,y2=centers[t]
            dx,dy=x2-x1,y2-y1; scale=7/math.hypot(dx,dy)
            x1,y1,x2,y2=x1+dx*scale,y1+dy*scale,x2-dx*scale,y2-dy*scale
            # Reflection of the source SVG's quadratic bend in PDF coordinates.
            mx,my=(x1+x2)/2-(y2-y1)*.06,(y1+y2)/2+(x2-x1)*.06
            c.setStrokeColor(HexColor('#ed6c02')); c.setFillColor(HexColor('#ed6c02'))
            c.setLineWidth(.95)
            c.bezier(x1,y1,(x1+2*mx)/3,(y1+2*my)/3,(x2+2*mx)/3,(y2+2*my)/3,x2,y2)
            dx,dy=x2-mx,y2-my; distance=math.hypot(dx,dy); ux,uy=dx/distance,dy/distance
            path=c.beginPath(); path.moveTo(x2,y2)
            path.lineTo(x2-3.8*ux+1.6*uy,y2-3.8*uy-1.6*ux)
            path.lineTo(x2-3.8*ux-1.6*uy,y2-3.8*uy+1.6*ux); path.close()
            c.drawPath(path,stroke=0,fill=1)
            drawn.append((k,s,t))
        for region,(cx,cy) in enumerate(centers):
            c.setFillColor(HexColor('#ffffff'));c.setStrokeColor(HexColor('#39424e'));c.setLineWidth(.5)
            c.circle(cx,cy,5.6,stroke=1,fill=1)
            c.setFont('GridReg',8.2);c.setFillColor(HexColor(INK));c.drawCentredString(cx,cy-2.7,str(region))
    else:
        for region,(rx,ry,w,h) in enumerate(b):
            c.setFillColor(HexColor('#23313d'));c.setFillAlpha(.8)
            c.rect(rx+1,224-ry-11,14,10,stroke=0,fill=1);c.setFillAlpha(1)
            c.setFillColor(HexColor('#ffffff'));c.setFont('GridReg',7)
            c.drawCentredString(rx+8,224-ry-8.5,str(region))
    c.restoreState()


def build():
    assert json.loads((EVIDENCE/'validation.json').read_text())['status']=='RETURNED_ARTIFACTS_AUDIT_PASS'
    path=OUT/'ben_gridpair42_cases.pdf'
    c=Canvas(str(path),pagesize=(WIDTH,HEIGHT),pageCompression=1,invariant=1)
    c.setTitle('BigEarthNet-19: trained fixed 3x3 and 4x4, seed42')
    c.setAuthor('GAViT Project')
    c.setSubject('Four fixed historical validation inputs, actual feature edges, paired probabilities and within-region pooling.')
    pages=[]
    for page,mode in enumerate(('feature','pooling'),1):
        placements=[]; drawn=[]
        heading='Actual graphs and paired predictions' if mode=='feature' else 'Within-region pooling weights'
        text(c,LEFT,HEIGHT-18,heading,size=11,bold=True)
        text(c,LEFT,HEIGHT-32,'BigEarthNet-19 | trained seed42 models | identical inputs | all four archived cases',size=6.9)
        for col,label in enumerate(('RGB input','Fixed 3 x 3 (K=9)','Fixed 4 x 4 (K=16)')):
            text(c,LEFT+col*(SIZE+GAP)+SIZE/2,HEIGHT-49,label,size=8.2,bold=True,align='center')
        for index,case in enumerate(DATA['cases']):
            top=760-index*179; bottom=top-SIZE
            text(c,LEFT,top+6,f'({chr(97+index)}) Validation row {case["row_index"]}',size=6.7,bold=True)
            image=SOURCE/'inputs'/f'row_{case["row_index"]}.png'
            for col in range(3):
                record=None if col==0 else case['models'][str(9 if col==1 else 16)]
                panel(c,LEFT+col*(SIZE+GAP),bottom,image,record,mode,placements,drawn)
            if mode=='feature':
                for line,ci in enumerate(FOCUS[case['row_index']]):
                    r=case['comparison'][ci]
                    value=f'{SHORT[ci]} (y={r["label"]}): {r["p9"]*100:.2f}% -> {r["p16"]*100:.2f}%  |  {VERDICT[r["outcome"]]}'
                    color='#166534' if r['outcome']=='k16_only_correct' else '#b42318' if r['outcome']=='k9_only_correct' else INK
                    text(c,LEFT,bottom-13-line*10.2,value,size=6.8,color=color)
            else:
                item=ANALYSIS['cases'][index]['pooling_concentration']
                a,b=item['9'],item['16']
                value=f'Regions with max pooling mass >=0.90: 3x3 {a["non_singleton_max_ge_0_9"]}/{a["non_singleton_regions"]}; 4x4 {b["non_singleton_max_ge_0_9"]}/{b["non_singleton_regions"]} (non-singleton only).'
                text(c,LEFT,bottom-13,value,size=6.5)
        if mode=='feature':
            for i,line in enumerate(('Orange arrows: captured feature edges only (18 for K9, 32 for K16); neighbor -> query.',
                                     'Region colors and IDs indicate spatial bins. Spatial edges and all 19 scores are retained in HTML.',
                                     'At threshold 0.5: 3 corrected label decisions, 1 new miss. Four cases are not a population sample.')):
                text(c,LEFT,36-i*10,line,size=6.0)
            expected=[(k,s,t) for case in DATA['cases'] for k in (9,16)
                      for (s,t),kind in zip(case['models'][str(k)]['edges'],case['models'][str(k)]['edge_kinds']) if kind=='feature']
            assert drawn==expected and len(drawn)==200
        else:
            for i,line in enumerate(('Red: above uniform mass in the same region. Blue: below uniform. Transparent: uniform.',
                                     'Mass is normalized within each region; heat is not class attribution or a semantic segmentation.',
                                     'K16 has one single-token region: mass=1 there is compulsory. No new model training or inference.')):
                text(c,LEFT,36-i*10,line,size=6.0)
        text(c,WIDTH-LEFT,7,f'{page}/2',size=5.5,align='right')
        pages.append(dict(page=page,mode=mode,image_placements=placements,feature_edges_drawn=drawn))
        c.showPage()
    c.save()
    shutil.copyfile(SOURCE/'comparison.html',OUT/'comparison.html')
    manifest=dict(source_captures_sha256=sha(SOURCE/'captures.json'),pdf_sha256=sha(path),
                  pdf_pages=2,size_points=[WIDTH,HEIGHT],selected_focus_classes=FOCUS,
                  all_threshold_flips_shown=True,pages=pages,
                  base_pixels_unchanged=True,model_inference_performed=False,
                  source_evidence=str(EVIDENCE.relative_to(ROOT)))
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(pdf=str(path),pages=2,feature_edges_drawn=200,base_image_placements=24),indent=2))


if __name__=='__main__': build()
