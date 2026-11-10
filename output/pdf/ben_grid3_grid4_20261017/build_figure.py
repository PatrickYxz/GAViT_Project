"""Exact fixed-grid overlays on four archived BEN inputs; no model inference."""
from collections import Counter
from pathlib import Path
import base64
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET

from PIL import Image
import reportlab
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT/'.worktrees/sparse-hybrid-4n-top2/results/evidence/pair44_feedback_k_7ehbne/case_sheet_return_20260919'
CODE = ROOT/'.worktrees/sparse-hybrid-4n-top2/models/region_grouping.py'
ROWS = [97389, 14680, 28450, 4439]
PALETTE = ['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4',
           '#46f0f0','#f032e6','#bcf60c','#fabebe','#008080','#e6beff',
           '#9a6324','#fffac8','#800000','#aaffc3']
ALPHA = .18
WIDTH, HEIGHT = 180*72/25.4, 720
SIZE, GAP, STRIDE = 145, 12, 165
LEFT = (WIDTH-3*SIZE-2*GAP)/2
INK = '#23313d'
FONT_DIR = Path(reportlab.__file__).resolve().parent/'fonts'
pdfmetrics.registerFont(TTFont('GridRegular', str(FONT_DIR/'Vera.ttf')))
pdfmetrics.registerFont(TTFont('GridBold', str(FONT_DIR/'VeraBd.ttf')))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pixel_sha(im):
    return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()


def geometry(side):
    # Exact position formula in SpatialGrouping and AttentiveSpatialGrouping.
    assignment = [min(int(i/7*side),side-1)*side+min(int(j/7*side),side-1)
                  for i in range(7) for j in range(7)]
    regions = []
    for k in range(side**2):
        cells = [n for n,a in enumerate(assignment) if a==k]
        xs, ys = [n%7 for n in cells], [n//7 for n in cells]
        x,y = min(xs)*32,min(ys)*32
        w,h = (max(xs)-min(xs)+1)*32,(max(ys)-min(ys)+1)*32
        assert w*h==len(cells)*32**2
        regions.append(dict(id=k,x=x,y=y,width=w,height=h,tokens=len(cells),color=PALETTE[k]))
    assert sum(r['width']*r['height'] for r in regions)==224**2
    return dict(side=side,K=side**2,assignment=assignment,regions=regions,
                token_counts=[r['tokens'] for r in regions],
                display_boundaries=[i*32 for i in range(1,7) if assignment[i]!=assignment[i-1]])


def read_cases():
    evidence=json.loads((SOURCE/'evidence.json').read_text())
    source_text=CODE.read_text()
    frozen=subprocess.check_output(['git','show','ed480e60fea03ed786fdba70c144de5e815c34f5:models/region_grouping.py'],cwd=ROOT,text=True)
    for expression in ('region_r = min(int(i / G * k_side), k_side - 1)',
                       'region_c = min(int(j / G * k_side), k_side - 1)'):
        assert expression in source_text and expression in frozen
    cases=[]
    for row in ROWS:
        jp,ip=SOURCE/f'row_{row}.json',SOURCE/f'row_{row}.png'
        assert sha(jp)==evidence['source_file_sha256'][jp.name]
        assert sha(ip)==evidence['source_file_sha256'][ip.name]
        record=json.loads(jp.read_text())
        assert record['row_index']==row
        assert record['assignments']==GEOMETRIES['4']['assignment']
        assert record['region_token_counts']==GEOMETRIES['4']['token_counts']
        with Image.open(ip) as im:
            assert im.size==(224,224) and im.mode=='RGB'
            pixels=pixel_sha(im)
        cases.append(dict(row=row,source_png=str(ip.relative_to(ROOT)),source_json=str(jp.relative_to(ROOT)),
                          png_sha256=sha(ip),json_sha256=sha(jp),pixel_sha256=pixels,
                          patch_path=record['patch_path'],labels=record['labels'],
                          captured_K16_membership_matches=True))
    return cases,evidence


def text(c,x,y,value,size=8,bold=False,align='left',color=INK):
    font='GridBold' if bold else 'GridRegular'
    assert pdfmetrics.stringWidth(value,font,size)<=WIDTH-2*LEFT, value
    c.setFillColor(HexColor(color)); c.setFont(font,size)
    {'left':c.drawString,'center':c.drawCentredString,'right':c.drawRightString}[align](x,y,value)


def overlay_pdf(c,x,bottom,g):
    scale=SIZE/224
    c.saveState()
    for r in g['regions']:
        c.setFillColor(HexColor(r['color'])); c.setFillAlpha(ALPHA)
        c.rect(x+r['x']*scale,bottom+(224-r['y']-r['height'])*scale,
               r['width']*scale,r['height']*scale,stroke=0,fill=1)
    c.setFillAlpha(1)
    c.setStrokeColor(HexColor('#ffffff')); c.setStrokeAlpha(.28); c.setLineWidth(.2)
    for p in range(32,224,32):
        c.line(x+p*scale,bottom,x+p*scale,bottom+SIZE)
        c.line(x,bottom+p*scale,x+SIZE,bottom+p*scale)
    c.setStrokeAlpha(.95); c.setLineWidth(.7)
    for r in g['regions']:
        rx,ry=x+r['x']*scale,bottom+(224-r['y']-r['height'])*scale
        rw,rh=r['width']*scale,r['height']*scale
        c.rect(rx,ry,rw,rh,stroke=1,fill=0)
        value=str(r['id']); label_w=pdfmetrics.stringWidth(value,'GridBold',5.5)+4
        cx,cy=rx+rw/2,ry+rh/2
        c.setFillColor(HexColor('#15232b')); c.setFillAlpha(.72)
        c.roundRect(cx-label_w/2,cy-4,label_w,8,2,stroke=0,fill=1)
        c.setFillAlpha(1)
        text(c,cx,cy-1.8,value,size=5.5,bold=True,align='center',color='#ffffff')
    c.restoreState()


def build_pdf(cases):
    p=OUT/'ben_grid3_grid4_cases.pdf'
    c=Canvas(str(p),pagesize=(WIDTH,HEIGHT),pageCompression=1,invariant=1)
    c.setTitle('BigEarthNet: fixed 3 x 3 and 4 x 4 partition geometry')
    c.setAuthor('GAViT Project')
    c.setSubject('Four archived validation inputs. Geometry only; no new inference, graph edges or performance estimates.')
    text(c,WIDTH/2,HEIGHT-8,'PARTITION GEOMETRY ONLY',size=6.5,bold=True,align='center',color='#62717b')
    for col,heading in enumerate(('RGB input (224 x 224)','Fixed 3 x 3 (K = 9)','Fixed 4 x 4 (K = 16)')):
        text(c,LEFT+col*(SIZE+GAP)+SIZE/2,HEIGHT-23,heading,size=8.7,bold=True,align='center')
    placements=[]
    for n,case in enumerate(cases):
        top=HEIGHT-48-n*STRIDE; bottom=top-SIZE
        text(c,LEFT,top+10,f'({chr(97+n)})  Validation row {case["row"]}',size=7,bold=True)
        basename=Path(case['patch_path']).name
        text(c,WIDTH-LEFT,top+10,basename,size=4.9,align='right',color='#62717b')
        for col in range(3):
            x=LEFT+col*(SIZE+GAP)
            c.drawImage(str(ROOT/case['source_png']),x,bottom,width=SIZE,height=SIZE)
            if col:
                overlay_pdf(c,x,bottom,GEOMETRIES[str(col+2)])
            c.setStrokeColor(HexColor('#aab5ba'));c.setLineWidth(.3)
            c.rect(x,bottom,SIZE,SIZE,stroke=1,fill=0)
            placements.append(dict(row=case['row'],column=col,pdf_box=[x,bottom,SIZE,SIZE]))
    text(c,LEFT,19,'Exact 7 x 7 token membership. Colors and numbers identify regions, not semantic classes.',size=6.0)
    text(c,LEFT,8,'Four archived validation cases; no graph edges, model predictions or semantic-quality scores are shown.',size=6.0)
    c.showPage();c.save()
    return p,placements


def build_svg(case):
    ns='http://www.w3.org/2000/svg'
    ET.register_namespace('',ns)
    def node(parent,name,attrs=None,value=None):
        n=ET.SubElement(parent,'{'+ns+'}'+name,attrs or {})
        n.text=value
        return n
    root=ET.Element('{'+ns+'}svg',dict(width='760',height='342',viewBox='0 0 760 342'))
    node(root,'rect',dict(width='760',height='342',fill='white'))
    node(root,'title',value=f'BEN row {case["row"]}: fixed 3 x 3 versus 4 x 4 partition geometry')
    node(root,'desc',value='Archived validation input; exact token grouping. No model inference or graph edges.')
    uri='data:image/png;base64,'+base64.b64encode((ROOT/case['source_png']).read_bytes()).decode()
    for col,label in enumerate(('RGB input','Fixed 3 x 3 (K = 9)','Fixed 4 x 4 (K = 16)')):
        x=16+col*252
        node(root,'text',dict(x=str(x+112),y='22',fill=INK,**{'font-family':'Arial,sans-serif','font-size':'14','font-weight':'bold','text-anchor':'middle'}),label)
        node(root,'image',dict(x=str(x),y='60',width='224',height='224',href=uri))
        if col:
            g=GEOMETRIES[str(col+2)]
            for r in g['regions']:
                node(root,'rect',dict(x=str(x+r['x']),y=str(60+r['y']),width=str(r['width']),height=str(r['height']),fill=r['color'],**{'fill-opacity':str(ALPHA),'data-region':str(r['id']),'data-tokens':str(r['tokens'])}))
            for p in range(32,224,32):
                node(root,'path',dict(d=f'M{x+p},60 V284 M{x},{60+p} H{x+224}',stroke='white',**{'stroke-width':'.35','stroke-opacity':'.28','fill':'none'}))
            for r in g['regions']:
                node(root,'rect',dict(x=str(x+r['x']),y=str(60+r['y']),width=str(r['width']),height=str(r['height']),fill='none',stroke='white',**{'stroke-width':'1.1','stroke-opacity':'.95'}))
                cx,cy=x+r['x']+r['width']/2,60+r['y']+r['height']/2
                node(root,'rect',dict(x=str(cx-8),y=str(cy-6),width='16',height='12',rx='3',fill='#15232b',**{'fill-opacity':'.72'}))
                node(root,'text',dict(x=str(cx),y=str(cy+3),fill='white',**{'font-family':'Arial,sans-serif','font-size':'9','font-weight':'bold','text-anchor':'middle'}),str(r['id']))
        node(root,'rect',dict(x=str(x),y='60',width='224',height='224',fill='none',stroke='#aab5ba',**{'stroke-width':'.5'}))
    node(root,'text',dict(x='16',y='43',fill=INK,**{'font-family':'Arial,sans-serif','font-size':'11'}),f'BEN validation row {case["row"]} | partition geometry only')
    for y,line in ((307,'Same archived 224 x 224 input; exact 7 x 7 token memberships. Colors and numbers are region IDs.'),
                   (325,'No new model inference, graph edges or classification-performance claims.')):
        node(root,'text',dict(x='16',y=str(y),fill='#62717b',**{'font-family':'Arial,sans-serif','font-size':'10'}),line)
    p=OUT/f'row_{case["row"]}_grid3_grid4.svg'
    ET.ElementTree(root).write(p,encoding='utf-8',xml_declaration=True)
    return p


GEOMETRIES={str(side):geometry(side) for side in (3,4)}
if __name__=='__main__':
    cases,source_evidence=read_cases()
    pdf,placements=build_pdf(cases)
    for case in cases:
        svg=build_svg(case)
        case['svg']=svg.name
        case['svg_sha256']=sha(svg)
    result=dict(dataset='BigEarthNet-19',date='2026-10-17',scope='fixed_partition_geometry_only',
                sample_count=4,split='val',selection='Reuse all four locally archived historical threshold-flip cases; not a representative sample or a new outcome-selected sample',
                original_selection=source_evidence['selection'],
                source_evidence=str((SOURCE/'evidence.json').relative_to(ROOT)),source_evidence_sha256=sha(SOURCE/'evidence.json'),
                source_grouping_sha256=sha(CODE),frozen_model_commit='ed480e60fea03ed786fdba70c144de5e815c34f5',
                geometry=GEOMETRIES,overlay_alpha=ALPHA,input_pixels=[224,224],
                native_input_pixels_modified=False,source_file_bytes_modified=False,
                new_inference=False,new_training=False,graph_edges_shown=False,model_predictions_shown=False,
                image_level_labels_are_not_spatial_annotations=True,semantic_quality_metrics=None,
                page_size_points=[WIDTH,HEIGHT],page_size_mm=[WIDTH*25.4/72,HEIGHT*25.4/72],
                pdf=pdf.name,pdf_sha256=sha(pdf),builder_sha256=sha(Path(__file__)),cases=cases,pdf_placements=placements)
    (OUT/'figure_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pdf':str(pdf),'cases':4,'native_images_unchanged':True,'K16_matches_archived_capture':True,'scope':result['scope']},ensure_ascii=False))
