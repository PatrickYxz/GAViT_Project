"""Artifact tests: no model inference, and no invented graph edges or input images."""
import base64
import importlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zlib

from tests import test_bigearth_feedback_diagnostic as helpers


def renderer():
    try:
        return importlib.import_module('tools.render_feedback_case_sheet')
    except ModuleNotFoundError as exc:
        raise AssertionError('case sheet renderer is not implemented') from exc


def make_fixture(root):
    """Synthetic PNG/graph fixture for layout tests, not scientific evidence."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run = dict(status='complete', input_mode='original', cases=[dict(row_index=i) for i in [97389,14680,28450,4439,*range(20)]])
    (root/'run.json').write_text(json.dumps(run))
    (root/'loaded_model.json').write_text(json.dumps(dict(device='SYNTHETIC LAYOUT FIXTURE',
        checkpoint=dict(sha256='80af1f7a5f0c4a5c983aa542e20749a191150d91738ae95894b95c1aff4e65bd'))))
    assignments, edges = helpers.FeedbackDiagnosticTests().graph()
    counts, kinds = helpers.diagnostic().validate_topology(assignments, edges)
    svg = helpers.diagnostic().graph_svg(assignments, edges, kinds, [1.0]*16)
    def chunk(kind, data):
        return struct.pack('!I', len(data))+kind+data+struct.pack('!I', zlib.crc32(kind+data)&0xffffffff)
    rows = b''.join(b'\x00'+bytes(v for x in range(224) for v in (x, y, 120)) for y in range(224))
    png = b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',224,224,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b'')
    for row, c, y, normal, zero in [(97389,11,0,.4858,.9691),(14680,11,1,.4704,.9170),
                                    (28450,14,0,.4518,.9972),(4439,14,1,.2812,.9974)]:
        ns=[.2]*19; ns[c]=normal
        zs=[.2]*19; zs[c]=zero
        labels=[0]*19; labels[c]=y
        record=dict(row_index=row, strata=[dict(class_index=c,label=y,bucket='fixture')],labels=labels,
                    normal_scores=ns,zero_scores=zs,cached_gavit_scores=ns,
                    cached_prediction_max_abs_error=0.0,assignments=assignments,edges=edges,
                    edge_kinds=kinds,region_token_counts=counts,feedback_L2=[1.0]*16,
                    logit_delta=[0.0]*19,gate_max_abs_errors={k:0.0 for k in ('capture','reconstruction','zero_branch','restoration')})
        (root/f'row_{row}.json').write_text(json.dumps(record))
        (root/f'row_{row}.png').write_bytes(png)
        (root/f'row_{row}.svg').write_text(svg)
    return root


class CaseSheetTests(unittest.TestCase):
    def test_embeds_four_real_assets_and_separates_captured_edge_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=make_fixture(tmp)
            page, evidence=renderer().render(root)
            self.assertEqual(page.count('<article '),4)
            self.assertEqual(page.count('data:image/png;base64,'),12)
            self.assertEqual(len(evidence['cases']),4)
            self.assertIn('0.4858',page)
            self.assertIn('0.9974',page)
            self.assertNotRegex(page,r'(?:src|href)="(?:https?://|file:|row_)')
            images=re.findall('data:image/svg\\+xml;base64,([^"\\s]+)',page)
            self.assertEqual(len(images),8)
            for i, image in enumerate(images):
                tree=ET.fromstring(base64.b64decode(image))
                edges=[e for e in tree.iter() if 'edge' in e.get('class','').split()]
                self.assertEqual(len(edges),48 if i%2==0 else 32)
                kind='spatial' if i%2==0 else 'feature'
                self.assertTrue(all(kind in e.get('class','').split() for e in edges))
                if kind=='feature':
                    self.assertIn(('5','0'),[(e.get('data-source'),e.get('data-target')) for e in edges])
            png=re.search('data:image/png;base64,([^"\\s]+)',page)[1]
            self.assertEqual(base64.b64decode(png),(root/'row_97389.png').read_bytes())

    def test_rejects_gallery_or_incomplete_run(self):
        for change in ({'input_mode':'gallery'},{'status':'failed'}):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=make_fixture(tmp)
                path=root/'run.json'; run=json.loads(path.read_text());run.update(change);path.write_text(json.dumps(run))
                with self.assertRaises(ValueError): renderer().render(root)

    def test_rejects_wrong_row_label_and_cache_mismatch(self):
        for change in ({'row_index':123},{'labels':[1]*19},{'cached_prediction_max_abs_error':.2}):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=make_fixture(tmp)
                path=root/'row_97389.json';row=json.loads(path.read_text());row.update(change);path.write_text(json.dumps(row))
                with self.assertRaises(ValueError): renderer().render(root)

    def test_rejects_svg_edge_identity_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=make_fixture(tmp)
            path=root/'row_97389.svg'
            path.write_text(path.read_text().replace('data-source="5" data-target="0"','data-source="6" data-target="0"'))
            with self.assertRaises(ValueError): renderer().render(root)

    def test_missing_image_stops_instead_of_creating_a_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=make_fixture(tmp); (root/'row_4439.png').unlink()
            with self.assertRaises(FileNotFoundError): renderer().render(root)

    def test_cli_writes_single_html_evidence_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=make_fixture(Path(tmp)/'source'); out=Path(tmp)/'output'
            cmd=[sys.executable,renderer().__file__,'--source',str(source),'--out',str(out)]
            result=subprocess.run(cmd,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('CASE SHEET COMPLETE',result.stdout)
            page=(out/'feedback_cases.html').read_bytes()
            self.assertTrue((out/'evidence.json').is_file())
            again=subprocess.run(cmd,text=True,capture_output=True)
            self.assertNotEqual(again.returncode,0)
            self.assertEqual((out/'feedback_cases.html').read_bytes(),page)


if __name__=='__main__':
    unittest.main()
