"""Verify original pixels, exact memberships, SVG layers and exported PDF geometry."""
from collections import Counter
from pathlib import Path
import base64
import hashlib
import io
import json
import math
import xml.etree.ElementTree as ET

from PIL import Image
from pypdf import PdfReader
from pypdf.generic import ContentStream

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
manifest=json.loads((OUT/'figure_manifest.json').read_text())
checks=[]
def check(ok,name):
    assert ok,name
    checks.append(name)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def pixel_sha(im):
    return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()

check(sha(OUT/manifest['pdf'])==manifest['pdf_sha256'],'PDF hash matches manifest')
check(sha(OUT/'build_figure.py')==manifest['builder_sha256'],'Builder hash matches manifest')
check(sha(ROOT/'.worktrees/sparse-hybrid-4n-top2/models/region_grouping.py')==manifest['source_grouping_sha256'],'Grouping source unchanged')
check(sha(ROOT/manifest['source_evidence'])==manifest['source_evidence_sha256'],'Archived evidence unchanged')
check([c['row'] for c in manifest['cases']]==[97389,14680,28450,4439],'All four archived cases retained in original order')
g3,g4=manifest['geometry']['3'],manifest['geometry']['4']
check(g3['display_boundaries']==[96,160] and g4['display_boundaries']==[64,128,192],'Actual token boundaries used; no equal-image divisions')
check(Counter(g3['token_counts'])=={9:1,6:4,4:4} and Counter(g4['token_counts'])=={4:9,2:6,1:1},'Exact49-token occupancy histograms')
for g in (g3,g4):
    # Independent reference: integer division, rather than the builder's float expression.
    reference=[((r*g['side'])//7)*g['side']+(c*g['side'])//7 for r in range(7) for c in range(7)]
    check(reference==g['assignment'],f'K{g["K"]} agrees with independent integer reference')
    check(len(g['regions'])==g['K'] and sum(g['token_counts'])==49,f'K{g["K"]} complete regions and49 memberships')

pixel_hashes=[]
svg_ns={'s':'http://www.w3.org/2000/svg'}
for case in manifest['cases']:
    row=case['row']; src=ROOT/case['source_png']; jp=ROOT/case['source_json']
    check(sha(src)==case['png_sha256'] and sha(jp)==case['json_sha256'],f'row{row} source bytes unchanged')
    record=json.loads(jp.read_text())
    check(record['assignments']==g4['assignment'] and record['region_token_counts']==g4['token_counts'],f'row{row} K16 agrees with archived actual capture')
    with Image.open(src) as im:
        check(im.size==(224,224) and im.mode=='RGB' and pixel_sha(im)==case['pixel_sha256'],f'row{row} native RGB pixels')
    pixel_hashes.append(case['pixel_sha256'])
    svg=ET.parse(OUT/case['svg']).getroot()
    check(sha(OUT/case['svg'])==case['svg_sha256'],f'row{row} SVG hash')
    embedded=svg.findall('s:image',svg_ns)
    check(len(embedded)==3 and all(base64.b64decode(n.attrib['href'].split(',',1)[1])==src.read_bytes() for n in embedded),f'row{row} three SVG inputs are byte-identical originals')
    region_rects=[n for n in svg.findall('s:rect',svg_ns) if 'data-region' in n.attrib]
    check(len(region_rects)==25 and sum(int(n.get('data-tokens')) for n in region_rects)==98,f'row{row} K9+K16 complete SVG overlays')
    check(all(float(n.get('fill-opacity'))==.18 for n in region_rects),f'row{row} consistent overlay transparency')
    check(not any('edge' in n.get('class','').split() for n in svg.iter()),f'row{row} no invented graph edges')
    with Image.open(OUT/f'row_{row}_grid3_grid4.png') as preview:
        check(preview.size==(2280,1026),f'row{row} rendered case preview exists')

reader=PdfReader(OUT/manifest['pdf'])
check(len(reader.pages)==1,'Single-page PDF')
page=reader.pages[0]
embedded_pixels=[]
for item in page.images:
    with Image.open(io.BytesIO(item.data)) as im:
        check(im.size==(224,224),'PDF embedded image retains native resolution')
        embedded_pixels.append(pixel_sha(im))
check(Counter(embedded_pixels)==Counter(pixel_hashes),'All four PDF embedded images retain exact original RGB pixels')
operations=ContentStream(page.get_contents(),reader).operations
check(sum(op==b'Do' for _,op in operations)==12,'PDF contains exactly12 original-image placements')
rectangles=Counter(tuple(round(float(v),3) for v in operands) for operands,op in operations if op==b're')
check(sum(rectangles.values())==212,'PDF has100 region fills,100 region outlines and12 panel borders')
for placement in manifest['pdf_placements']:
    x,bottom,size,_=placement['pdf_box']
    if placement['column']==0:
        continue
    g=manifest['geometry'][str(placement['column']+2)]
    for r in g['regions']:
        rect=(x+r['x']*size/224,bottom+(224-r['y']-r['height'])*size/224,r['width']*size/224,r['height']*size/224)
        check(rectangles[tuple(round(v,3) for v in rect)]==2,'PDF region rectangle matches exact membership coordinates')
text=page.extract_text()
check('PARTITION GEOMETRY ONLY' in text and 'no graph edges, model predictions' in text,'PDF visibly states interpretation scope')
check(all(f'Validation row {r}' in text for r in (97389,14680,28450,4439)),'All PDF row identities present')
fonts=page['/Resources']['/Font'].get_object()
embedded_font_names=[]
for font in fonts.values():
    f=font.get_object(); descriptor=f.get('/FontDescriptor')
    if descriptor and any(descriptor.get_object().get(k) for k in ('/FontFile','/FontFile2','/FontFile3')):
        embedded_font_names.append(str(f.get('/BaseFont')))
check(len(embedded_font_names)>=2,'Used regular and bold figure fonts embedded')
with Image.open(OUT/'ben_grid3_grid4_300dpi.png') as im:
    check(im.size==(2126,3000),'300dpi full-figure PNG size')
with Image.open(OUT/'preview.png') as im:
    check(im.size==(1276,1800),'Latest PDF inspection preview exists')

result=dict(status='pass',check_count=len(checks),checks=checks,
            pdf_sha256=sha(OUT/manifest['pdf']),source_pixels_verified=4,
            no_new_model_inference=True,no_new_training=True,
            visual_review='All four case PNGs and the rendered PDF reviewed; no observed clipping, overlap or missing layers',
            native_input_pixels=[224,224],input_detail_not_increased=True,
            actual_K9_model_evaluated=False,semantic_quality_quantified=False)
(OUT/'export_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'pass','checks':len(checks),'original_images':4,'PDF_placements':12,'region_overlays':100},ensure_ascii=False))
