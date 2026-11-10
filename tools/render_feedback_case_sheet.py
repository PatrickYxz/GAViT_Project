#!/usr/bin/env python3
"""Create one offline HTML from four captured original-input cases; no inference."""
import argparse
import base64
import copy
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import struct
import tempfile
import xml.etree.ElementTree as ET

PLAN = [(97389,11,0,'自然草地 / 稀疏植被'),(14680,11,1,'自然草地 / 稀疏植被'),
        (28450,14,0,'沙滩 / 沙丘 / 沙地'),(4439,14,1,'沙滩 / 沙丘 / 沙地')]
GAVIT_SHA = '80af1f7a5f0c4a5c983aa542e20749a191150d91738ae95894b95c1aff4e65bd'
ET.register_namespace('', 'http://www.w3.org/2000/svg')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def uri(mime, data):
    return 'data:'+mime+';base64,'+base64.b64encode(data).decode('ascii')


def graph_layers(svg, row):
    if b'<!DOCTYPE' in svg.upper() or b'<!ENTITY' in svg.upper():
        raise ValueError('Unsupported SVG declarations')
    tree = ET.fromstring(svg)
    allowed = {'svg','defs','marker','path','rect','g','title','circle','text'}
    actual = []
    for node in tree.iter():
        if node.tag.split('}')[-1] not in allowed or any(k.lower().startswith('on') or 'href' in k.lower() for k in node.attrib):
            raise ValueError('Unsupported SVG element or attribute')
        classes = node.get('class','').split()
        if 'edge' in classes:
            kind = next((k for k in ('spatial','feature') if k in classes), None)
            actual.append((int(node.attrib['data-source']),int(node.attrib['data-target']),kind))
    expected = [(int(a),int(b),kind) for (a,b),kind in zip(row['edges'],row['edge_kinds'])]
    if len(row['edges'])!=80 or len(row['edge_kinds'])!=80 or len(actual)!=80 or len(set(expected))!=80 or sorted(actual)!=sorted(expected):
        raise ValueError('SVG edges differ from the captured JSON')
    for kind, count in (('spatial',48),('feature',32)):
        if sum(k==kind for _,_,k in actual)!=count:
            raise ValueError('Wrong captured edge budget')
    layers=[]
    for keep in ('spatial','feature'):
        layer=copy.deepcopy(tree)
        for parent in layer.iter():
            for node in list(parent):
                classes=node.get('class','').split()
                if 'edge' in classes and keep not in classes:
                    parent.remove(node)
        layers.append(ET.tostring(layer,encoding='utf-8'))
    return layers


def render(source):
    source=Path(source).resolve()
    fingerprints={}
    def read(name):
        data=(source/name).read_bytes()
        fingerprints[name]=sha(data)
        return data
    run=json.loads(read('run.json'))
    loaded=json.loads(read('loaded_model.json'))
    if run.get('status')!='complete' or run.get('input_mode')!='original':
        raise ValueError('A completed original-input diagnostic is required')
    if loaded['checkpoint']['sha256']!=GAVIT_SHA:
        raise ValueError('Expected the recorded GAViT44 checkpoint')
    source_ids=[r['row_index'] for r in run['cases']]
    if len(source_ids)!=24 or len(set(source_ids))!=24:
        raise ValueError('Expected the fixed 24-image original run')
    cards, cases=[],[]
    for index,(row_id,c,y,name) in enumerate(PLAN):
        stem=f'row_{row_id}'
        row=json.loads(read(stem+'.json'))
        if row_id not in source_ids or row['row_index']!=row_id or len(row['labels'])!=19 or row['labels'][c]!=y:
            raise ValueError('Case identity or label mismatch')
        if not any(s['class_index']==c and s['label']==y for s in row['strata']):
            raise ValueError('Case is not in its recorded stratum')
        for key in ('normal_scores','zero_scores','cached_gavit_scores'):
            if len(row[key])!=19 or any(not math.isfinite(p) or not 0<=p<=1 for p in row[key]):
                raise ValueError('Invalid probability vector')
        error=row['cached_prediction_max_abs_error']
        measured=max(abs(a-b) for a,b in zip(row['normal_scores'],row['cached_gavit_scores']))
        if not math.isfinite(error) or not 0<=error<=1e-4 or abs(error-measured)>1e-12:
            raise ValueError('Original prediction did not match its cache')
        normal,zero=row['normal_scores'][c],row['zero_scores'][c]
        n_ok,z_ok=(normal>=.5)==bool(y),(zero>=.5)==bool(y)
        if n_ok==z_ok or n_ok!=bool(index%2==0):
            raise ValueError('Expected all four recorded threshold-flip cases')
        png=read(stem+'.png')
        if png[:8]!=bytes([137,80,78,71,13,10,26,10]) or png[12:16]!=b'IHDR' or struct.unpack('!II',png[16:24])!=(224,224):
            raise ValueError('Expected the saved 224x224 lossless PNG')
        svg=read(stem+'.svg')
        layers=graph_layers(svg,row)
        picture=uri('image/png',png)
        panels=[]
        for label,overlay in [('原图输入',None),('区域 + 空间边 · 48条',layers[0]),('区域 + 特征边 · 32条',layers[1])]:
            overlay_tag='' if overlay is None else f'<img class="overlay" alt="Captured directed edges" src="{uri("image/svg+xml",overlay)}">'
            panels.append(f'<figure><div class="scene"><img alt="Original RGB input, row {row_id}" src="{picture}">{overlay_tag}</div><figcaption>{label}</figcaption></figure>')
        effect='纠正误报' if n_ok else '增加漏报'
        truth='不属于该类（0）' if not y else '属于该类（1）'
        tag='help' if n_ok else 'harm'
        cards.append(f'<article class="{tag}" id="{stem}"><header><div><small>CASE {index+1} / ROW {row_id}</small><h2>{html.escape(name)}</h2><p>真实标签：{truth}</p></div><b class="badge">{effect}</b></header><div class="panels">'+''.join(panels)+f'</div><div class="scores"><span>屏蔽反馈 <strong>{zero:.4f}</strong></span><span class="arrow">→</span><span>保留反馈 <strong>{normal:.4f}</strong></span><span>阈值 0.5 · 原缓存 {row["cached_gavit_scores"][c]:.4f}</span></div></article>')
        cases.append(dict(row_index=row_id,class_index=c,record=row,original_svg_base64=base64.b64encode(svg).decode('ascii')))
    evidence=dict(scope='four_selected_original_validation_cases_not_population_metrics',source=str(source),
                  generated_at=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=GAVIT_SHA,
                  device=loaded.get('device'),source_file_sha256=fingerprints,cases=cases,
                  selection='All four threshold-flip cases among the previously fixed 24 images',
                  inference_performed=False,edges_recomputed=False,visual_content_reviewed=False)
    page="""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'"><title>GAViT · 原图反馈案例</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f3ee;color:#202b31;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1040px;margin:auto;padding:42px 28px}h1{font-size:32px;line-height:1.25;margin:8px 0 16px}h2{font-size:21px;margin:2px 0}p{margin:4px 0 14px}small{letter-spacing:.08em;color:#647379}.intro{max-width:880px}.legend{display:flex;gap:24px;padding:14px 0 24px;font-weight:600}.blue{color:#1976d2}.orange{color:#b95000}article{background:white;border:1px solid #d9dfdc;border-radius:12px;padding:24px;margin:0 0 26px;break-inside:avoid}article header{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:12px}.badge{border-radius:6px;padding:6px 12px;white-space:nowrap}.help .badge{background:#e7f3ef;color:#1e6650}.harm .badge{background:#faece6;color:#a44725}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}figure{margin:0}.scene{position:relative;aspect-ratio:1;background:#eef1f0}.scene img{display:block;width:100%;height:100%}.scene .overlay{position:absolute;inset:0}figcaption{font-size:14px;text-align:center;padding:8px 0}.scores{display:flex;align-items:center;flex-wrap:wrap;gap:14px;border-top:1px solid #e4e7e5;padding-top:14px;margin-top:6px;font-size:14px}.scores strong{font-size:22px;font-variant-numeric:tabular-nums;margin-left:5px}.arrow{color:#607270;font-size:22px}details{padding:16px 0;overflow-wrap:anywhere;color:#59676b}footer{border-top:1px solid #cbd4d1;padding-top:18px;font-size:14px;color:#58676c}@media(max-width:620px){main{padding:24px 14px}article{padding:16px}.panels{grid-template-columns:1fr}.scene{max-width:360px;margin:auto}h1{font-size:26px}}@media print{body{background:white}main{max-width:none;padding:0}article{border-radius:0;margin:0 0 18px;padding:14px}.panels{grid-template-columns:repeat(3,minmax(0,1fr))}details{display:none}h1{font-size:24px}}
</style></head><body><main><small>GAViT / SEED 44 / FIXED VALIDATION CASES</small><h1>图反馈的两种作用</h1><p class="intro">同一份已训练权重、同一张原图，对比保留与屏蔽图反馈。这4例覆盖固定24张案例中全部阈值翻转：两例纠正误报，两例增加漏报。</p><div class="legend"><span class="blue">蓝色 · 空间边</span><span class="orange">橙色 · 特征边</span></div>"""
    page+=''.join(cards)
    page+='<footer><p>图层直接筛选已捕获的有向图输入边，箭头为邻居 → 查询区域，区域编号0–15。区域颜色仅区分分组；边和颜色不是注意力强度，也不证明特定边导致错误。GAT内部自环未绘制。</p><p>原始224×224 RGB输入以无损PNG嵌入。案例经选择，不代表总体AP/F1；零反馈是共同训练后模型的推理干预，不是重新训练的无图模型或Swin baseline。本页不运行模型。</p></footer>'
    page+=f'<details><summary>来源与核验信息</summary><p>{html.escape(str(source))}</p><p>设备：{html.escape(str(loaded.get("device")))}；权重SHA256：{GAVIT_SHA}</p><p>4张图的正常预测均与原缓存相差不超过1e-4；图片、分数和SVG原件哈希嵌入本文件。科学图像内容仍需人工检查。</p></details>'
    payload=json.dumps(evidence,ensure_ascii=False,allow_nan=False).replace('<',chr(92)+'u003c')
    return page+f'<script type="application/json" id="case-evidence">{payload}</script></main></body></html>',evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('/home/featurize/work/pair44_feedback_k_7ehbne'))
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    page,evidence=render(args.source)
    if args.out:
        out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    else:
        out=Path(tempfile.mkdtemp(prefix='pair44_case_sheet_',dir=args.source.parent)).resolve()
    with (out/'feedback_cases.html').open('x',encoding='utf-8') as f: f.write(page)
    evidence['html_sha256']=sha(page.encode('utf-8'))
    evidence['renderer_sha256']=sha(Path(__file__).read_bytes())
    with (out/'evidence.json').open('x',encoding='utf-8') as f: json.dump(evidence,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('CASE SHEET COMPLETE:',out/'feedback_cases.html')
    print('4 original cases; 12 panels; no inference. Open or download only feedback_cases.html.')


if __name__=='__main__':
    main()
