import base64,hashlib,json,xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from collections import Counter
root=Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2')
p=root/'results/evidence/nwpu_regions_review_20260929'; p.mkdir(exist_ok=True)
class Gallery(HTMLParser):
    def __init__(self): super().__init__(); self.images=[]; self.articles=0; self.scripts=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='article': self.articles+=1
        if tag=='script': self.scripts+=1
        if tag=='img': self.images.append(a)
source=Path('/Users/patrick/Downloads/cases_compact.html'); raw=source.read_bytes(); h=Gallery(); h.feed(raw.decode())
assert h.articles==90 and len(h.images)==180 and h.scripts==0
ns={'s':'http://www.w3.org/2000/svg'}; rows=[]
for i in range(90):
    pair=[]
    for j,kind in enumerate(('main','memberships')):
        uri=h.images[i*2+j]['src']; assert uri.startswith('data:image/svg+xml;base64,')
        b=base64.b64decode(uri.split(',',1)[1],validate=True); e=ET.fromstring(b)
        assert e.tag=='{'+ns['s']+'}svg'
        (p/f'{i:03d}_{kind}.svg').write_bytes(b); pair.append(e)
    e,m=pair; texts=[''.join(x.itertext()) for x in e.findall('.//s:text',ns)]
    path=texts[0]; label=texts[1].split('label: ',1)[1]; preds=[x.split('Prediction: ',1)[1] for x in texts if x.startswith('Prediction: ')]
    assert len(preds)==2 and label==path.split('/')[0]
    panels=[g for g in e.findall('.//s:g',ns) if g.attrib.get('transform','').startswith('translate(')]
    assert len(panels)==4
    grids=[[x.attrib['fill'] for x in g.findall('s:rect',ns)] for g in panels[1:]]; assert all(len(x)==49 for x in grids)
    changed=sum(a!=b for a,b in zip(*grids[:2]))
    edges=[]
    for g in panels[1:3]:
        ed=[(n.attrib['class'].split()[-1],int(n.attrib['data-source']),int(n.attrib['data-target'])) for n in g.findall('s:line',ns)]
        assert Counter(t[0] for t in ed)=={'spatial':48,'feature':32}; edges.append(ed)
    assert {x for x in edges[0] if x[0]=='spatial'}=={x for x in edges[1] if x[0]=='spatial'}
    shared=len(set(x for x in edges[0] if x[0]=='feature')&set(x for x in edges[1] if x[0]=='feature'))
    png=e.find('.//s:image',ns).attrib['href']; png2=m.find('.//s:image',ns).attrib['href']; assert png==png2
    (p/f'{i:03d}_input.png').write_bytes(base64.b64decode(png.split(',',1)[1],validate=True))
    row=dict(index=i,path=path,label=label,fixed_prediction=preds[0],soft_prediction=preds[1],fixed_correct=preds[0]==label,soft_correct=preds[1]==label,displayed_argmax_changed_tokens=changed,shared_directed_feature_edges=shared,soft_argmax_regions_present=len(set(grids[1])),input_png_sha256=hashlib.sha256(base64.b64decode(png.split(',',1)[1])).hexdigest())
    rows.append(row)
assert len(set(r['path'] for r in rows))==90
assert len(Counter(r['label'] for r in rows))==45 and set(Counter(r['label'] for r in rows).values())=={2}
paired=Counter((r['fixed_correct'],r['soft_correct']) for r in rows)
summary={'source':str(source),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'articles':h.articles,'svg_count':len(h.images),'paired':{str(k):v for k,v in paired.items()},'argmax_changed_tokens':dict(min=min(r['displayed_argmax_changed_tokens'] for r in rows),max=max(r['displayed_argmax_changed_tokens'] for r in rows),sum=sum(r['displayed_argmax_changed_tokens'] for r in rows),total=90*49),'shared_feature_edges':dict(min=min(r['shared_directed_feature_edges'] for r in rows),max=max(r['shared_directed_feature_edges'] for r in rows),sum=sum(r['shared_directed_feature_edges'] for r in rows),total=90*32),'case_rows':rows}
assert paired=={(True,True):82,(False,True):3,(False,False):5}
(p/'gallery_inventory.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='case_rows'},indent=2))
print('FOCUS',json.dumps([r for r in rows if not r['fixed_correct']],indent=2))
print('CHANGE_RANGE',json.dumps(sorted(rows,key=lambda r:r['displayed_argmax_changed_tokens'])[:2]+sorted(rows,key=lambda r:r['displayed_argmax_changed_tokens'])[-2:],indent=2))
