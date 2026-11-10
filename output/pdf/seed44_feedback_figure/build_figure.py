#!/usr/bin/env python3
"""One-off paper figure from archived, captured cases. No model or graph inference."""
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from PIL import Image
import reportlab
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import registerFont, stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / 'results/evidence/pair44_feedback_k_7ehbne/case_sheet_return_20260919'
PLAN = [(97389, 11, 0), (14680, 11, 1), (28450, 14, 0), (4439, 14, 1)]
WIDTH = 180 * mm
MARGIN, SIDEBAR, GUTTER, GAP = 6, 87, 9, 10
SIZE = (WIDTH - 2 * MARGIN - SIDEBAR - GUTTER - 2 * GAP) / 3
STRIDE = SIZE + 23
HEIGHT = 30 + 4 * STRIDE + 18
START = MARGIN + SIDEBAR + GUTTER
INK, MUTED = '#222930', '#59646b'
BLUE, ORANGE = '#1976d2', '#b95300'
GREEN, RED = '#246352', '#963f30'
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
QPATH = re.compile(rf'M({NUMBER}),({NUMBER}) Q({NUMBER}),({NUMBER}) ({NUMBER}),({NUMBER})')
FONT_DIR = Path(reportlab.__file__).resolve().parent / 'fonts'
registerFont(TTFont('FigureRegular', str(FONT_DIR / 'Vera.ttf')))
registerFont(TTFont('FigureBold', str(FONT_DIR / 'VeraBd.ttf')))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_cases():
    audit = json.loads((SOURCE / 'review_audit.json').read_text())
    for name, expected in audit['archived_file_sha256'].items():
        if sha(SOURCE / name) != expected:
            raise ValueError(f'Archived file changed: {name}')
    result = []
    for row_id, target, label in PLAN:
        row = json.loads((SOURCE / f'row_{row_id}.json').read_text())
        if row['row_index'] != row_id or row['labels'][target] != label:
            raise ValueError('Case identity mismatch')
        with Image.open(SOURCE / f'row_{row_id}.png') as im:
            if im.size != (224, 224) or im.mode != 'RGB':
                raise ValueError('Expected unchanged RGB input')
        row['figure_class'] = target
        row['figure_label'] = label
        result.append(row)
    return result


def text(c, x, y, value, size=8, bold=False, color=INK, align='left'):
    c.setFillColor(HexColor(color))
    c.setFont('FigureBold' if bold else 'FigureRegular', size)
    draw = c.drawCentredString if align == 'center' else c.drawString
    draw(x, y, value)


def graph(c, source, row, kind, x, bottom):
    """Draw the captured SVG geometry as PDF vectors; only typography/strokes change."""
    scale = SIZE / 224
    root = ET.fromstring(source.read_bytes())
    edges = [n for n in root.iter() if 'edge' in n.get('class', '').split()]
    expected = [(a, b) for (a, b), k in zip(row['edges'], row['edge_kinds']) if k == kind]
    actual = [(int(n.attrib['data-source']), int(n.attrib['data-target'])) for n in edges]
    if actual != expected or len(edges) != {'spatial': 48, 'feature': 32}[kind]:
        raise ValueError('Filtered SVG differs from captured graph')
    def point(px, py):
        return x + px * scale, bottom + SIZE - py * scale
    c.saveState()
    for n in root:
        if n.tag.split('}')[-1] != 'rect':
            continue
        px, py, w, h = [float(n.get(k)) for k in ('x', 'y', 'width', 'height')]
        c.setFillColor(HexColor(n.get('fill')))
        c.setFillAlpha(float(n.get('fill-opacity', '1')))
        c.rect(x + px * scale, bottom + SIZE - (py + h) * scale, w * scale, h * scale, stroke=0, fill=1)
        c.setFillAlpha(1)
        c.setStrokeColor(HexColor('#ffffff')); c.setLineWidth(.25)
        c.rect(x + px * scale, bottom + SIZE - (py + h) * scale, w * scale, h * scale, stroke=1, fill=0)
    color = BLUE if kind == 'spatial' else ORANGE
    c.setStrokeColor(HexColor(color)); c.setFillColor(HexColor(color))
    c.setLineWidth(.5)
    drawn = []
    for n in edges:
        match = QPATH.fullmatch(n.attrib['d'])
        if not match:
            raise ValueError('Unexpected captured curve syntax')
        x0, y0, qx, qy, x1, y1 = map(float, match.groups())
        a, q, b = point(x0, y0), point(qx, qy), point(x1, y1)
        # Exact quadratic-to-cubic conversion, retaining the source SVG path.
        first = tuple(a[i] + 2 / 3 * (q[i] - a[i]) for i in (0, 1))
        second = tuple(b[i] + 2 / 3 * (q[i] - b[i]) for i in (0, 1))
        path = c.beginPath(); path.moveTo(*a); path.curveTo(*first, *second, *b)
        c.drawPath(path, stroke=1, fill=0)
        dx, dy = b[0] - q[0], b[1] - q[1]
        length = math.hypot(dx, dy); ux, uy = dx / length, dy / length
        # Arrow tip follows the target-side tangent; topology is not recomputed.
        arrow = c.beginPath(); arrow.moveTo(*b)
        arrow.lineTo(b[0] - 3 * ux - 1.25 * uy, b[1] - 3 * uy + 1.25 * ux)
        arrow.lineTo(b[0] - 3 * ux + 1.25 * uy, b[1] - 3 * uy - 1.25 * ux)
        arrow.close(); c.drawPath(arrow, fill=1, stroke=0)
        drawn.append({'source': int(n.get('data-source')), 'target': int(n.get('data-target')), 'captured_path': n.get('d')})
    nodes = [n for n in root if n.tag.split('}')[-1] == 'g']
    if len(nodes) != 16:
        raise ValueError('Expected 16 captured nodes')
    for group in nodes:
        circle = next(n for n in group if n.tag.split('}')[-1] == 'circle')
        label = next(n for n in group if n.tag.split('}')[-1] == 'text').text
        px, py = point(float(circle.get('cx')), float(circle.get('cy')))
        c.setFillColor(HexColor('#ffffff')); c.setStrokeColor(HexColor('#354049')); c.setLineWidth(.35)
        c.circle(px, py, 4.3, stroke=1, fill=1)
        text(c, px, py - 2.15, label, size=6.4, align='center')
    c.restoreState()
    return drawn


def build():
    rows = read_cases()
    filename = OUT / 'gavit_seed44_feedback_cases.pdf'
    c = Canvas(str(filename), pagesize=(WIDTH, HEIGHT), pageCompression=1, invariant=1)
    c.setTitle('GAViT seed44: graph-feedback case analysis')
    c.setAuthor('GAViT Project')
    c.setSubject('All four threshold flips among 24 fixed validation cases; same trained weights')
    for index, (label, color) in enumerate([('RGB input', INK), ('Spatial edges (48)', BLUE), ('Feature edges (32)', ORANGE)]):
        text(c, START + index * (SIZE + GAP) + SIZE / 2, HEIGHT - 14, label, size=9, bold=True, color=color, align='center')
    text(c, MARGIN, HEIGHT - 14, 'Target class', size=8.5, bold=True)
    manifest = []
    for index, row in enumerate(rows):
        row_id, target, label = row['row_index'], row['figure_class'], row['figure_label']
        top = HEIGHT - 30 - index * STRIDE
        bottom = top - SIZE
        normal, zero = row['normal_scores'][target], row['zero_scores'][target]
        corrected = ((normal >= .5) == bool(label))
        if ((zero >= .5) == bool(label)) == corrected:
            raise ValueError('Expected a threshold flip')
        outcome = 'FP corrected' if corrected else 'FN introduced'
        effect_color = GREEN if corrected else RED
        text(c, MARGIN, top - 11, f'({chr(97 + index)})', size=11, bold=True)
        text(c, MARGIN, top - 25, f'row {row_id}', size=7, color=MUTED)
        lines = ['Natural grassland', 'and sparsely', 'vegetated areas'] if target == 11 else ['Beaches, dunes,', 'sands']
        for line_index, line in enumerate(lines):
            text(c, MARGIN, top - 44 - 10 * line_index, line, size=8.1)
        text(c, MARGIN, top - 85, f'True label: {label}', size=8)
        text(c, MARGIN, top - 106, outcome, size=8.2, bold=True, color=effect_color)
        capture = dict(row_index=row_id, class_index=target, label=label, normal=normal, zero=zero,
                       original_png_sha256=sha(SOURCE / f'row_{row_id}.png'), panels=[])
        for col in range(3):
            x = START + col * (SIZE + GAP)
            c.drawImage(str(SOURCE / f'row_{row_id}.png'), x, bottom, width=SIZE, height=SIZE)
            if col:
                kind = 'spatial' if col == 1 else 'feature'
                drawn = graph(c, SOURCE / f'row_{row_id}_{kind}.svg', row, kind, x, bottom)
                capture['panels'].append({'kind': kind, 'edges': drawn})
            c.setStrokeColor(HexColor('#c5cbd0')); c.setLineWidth(.3)
            c.rect(x, bottom, SIZE, SIZE, stroke=1, fill=0)
        label_off = f'Feedback off: {zero:.4f}'
        label_on = f'Feedback on: {normal:.4f}'
        font_size = 8.6
        widths = [stringWidth(v, 'FigureBold', font_size) for v in (label_off, label_on)]
        block_width = widths[0] + 30 + widths[1]
        sx = START + (3 * SIZE + 2 * GAP - block_width) / 2
        sy = bottom - 12
        text(c, sx, sy, label_off, size=font_size, bold=True)
        ax = sx + widths[0] + 8
        c.setStrokeColor(HexColor(MUTED)); c.setLineWidth(.65)
        c.line(ax, sy + 2.5, ax + 13, sy + 2.5)
        c.line(ax + 13, sy + 2.5, ax + 9.5, sy + 4.6)
        c.line(ax + 13, sy + 2.5, ax + 9.5, sy + .4)
        text(c, sx + widths[0] + 30, sy, label_on, size=font_size, bold=True, color=effect_color)
        manifest.append(capture)
    text(c, MARGIN, 14, 'Threshold: 0.5. Arrows: neighbor to query. Region colors indicate group identity; GAT self-loops are omitted.', size=7, color=MUTED)
    text(c, MARGIN, 4, 'Selected cases; same trained weights. Feedback off zeros the feedback projection output at inference.', size=7, color=MUTED)
    c.showPage(); c.save()
    result = dict(source=str(SOURCE.relative_to(ROOT)), source_html_sha256=sha(SOURCE / 'feedback_cases.html'),
                  pdf_sha256=sha(filename), builder_sha256=sha(Path(__file__)), page_size_mm=[WIDTH / mm, HEIGHT / mm],
                  native_input_pixels=[224, 224], image_pixels_modified=False, new_inference=False,
                  visual_changes='PDF typography, line weights, arrowheads and node label size; retained captured paths, node positions, token rectangles and fill opacity',
                  records=manifest)
    (OUT / 'figure_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'Created {filename}; {WIDTH / mm:.1f} x {HEIGHT / mm:.1f} mm; 4 cases, 12 panels, 320 captured directed edges.')


if __name__ == '__main__':
    build()
