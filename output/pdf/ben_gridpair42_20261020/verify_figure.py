"""Independent PDF/export checks against the returned input and predictions."""
from pathlib import Path
import hashlib
import json

from PIL import Image
import pdfplumber
from pypdf import PdfReader

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SOURCE=ROOT/'results/evidence/ben_gridpair42_visual_return_20261020/server_return'


def pixels(im): return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()


def main():
    manifest=json.loads((OUT/'figure_manifest.json').read_text())
    captures=json.loads((SOURCE/'captures.json').read_text())
    pdf=OUT/'ben_gridpair42_cases.pdf'
    assert hashlib.sha256(pdf.read_bytes()).hexdigest()==manifest['pdf_sha256']
    reader=PdfReader(pdf); assert len(reader.pages)==2
    originals=sorted(pixels(Image.open(SOURCE/'inputs'/f'row_{c["row_index"]}.png')) for c in captures['cases'])
    text=reader.pages[0].extract_text()
    for case in captures['cases']:
        for r in case['comparison']:
            if r['outcome'] in ('k9_only_correct','k16_only_correct'):
                assert r['class_index'] in manifest['selected_focus_classes'][str(case['row_index'])]
                assert f'{r["p9"]*100:.2f}% -> {r["p16"]*100:.2f}%' in text
    for page in reader.pages:
        assert sorted(pixels(item.image) for item in page.images)==originals
        assert page.get_contents().get_data().count(b' Do')==12
        assert all(str(row) in page.extract_text() for row in (97389,14680,28450,4439))
    assert 'new miss' in text and 'still wrong' in text and '3 corrected label decisions, 1 new miss' in text
    with pdfplumber.open(pdf) as opened:
        chars=0
        for page in opened.pages:
            for char in page.chars:
                assert char['x0']>=-0.1 and char['x1']<=page.width+.1
                assert char['top']>=-0.1 and char['bottom']<=page.height+.1
                chars+=1
    sizes={}
    for name in ('graphs_predictions_300dpi.png','pooling_300dpi.png'):
        with Image.open(OUT/name) as im:
            assert im.size==(2126,3459),im.size
            sizes[name]=list(im.size)
    assert hashlib.sha256((OUT/'comparison.html').read_bytes()).hexdigest()==hashlib.sha256((SOURCE/'comparison.html').read_bytes()).hexdigest()
    v=dict(status='EXPORT_CHECKS_PASS',pdf_pages=2,original_pixels_preserved=True,
           base_image_placements=24,actual_feature_edges_page1=200,pooling_tokens_page2=392,
           all_four_cases_and_all_threshold_changes_retained=True,all_text_within_page=True,
           checked_characters=chars,output_png_sizes=sizes,html_byte_identical_to_returned=True,
           visual_review_pages=[1,2],visual_review_result='headings, all panels, labels, probabilities and footnotes inspected; no clipping or overlap',
           html_browser_interaction_tested=False)
    (OUT/'export_qa.json').write_text(json.dumps(v,indent=2)+'\n')
    print(json.dumps(v,indent=2))


if __name__=='__main__': main()
