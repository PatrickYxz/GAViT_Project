"""Verify final offline outputs, then package a review copy. Never replace files."""
from pathlib import Path
import csv
import hashlib
import io
import json
import re
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
checks = {}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, ok):
    checks[name] = bool(ok)
    if not ok:
        raise ValueError(name)


def new_json(path, obj):
    with path.open('x') as stream:
        stream.write(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


receipt = read(HERE / 'archive_receipt.json')
check('preserved_archive', sha(HERE / 'reported_results.tar.gz') == receipt['sha256'])
check('seven_members', len(receipt['members']) == 7)
for member in receipt['members']:
    path = HERE / 'files' / member['path']
    check('raw:' + member['path'], path.stat().st_size == member['bytes'] and sha(path) == member['sha256'])

for rel, count in [('validation.json', 742), ('review_predictions/validation.json', 131),
                   ('review_history/validation.json', 498), ('figures/visual_validation.json', 26)]:
    result = read(HERE / rel)
    check('audit:' + rel, result['passed'] is True and result.get('count', result.get('checks_count')) == count)
check('delivered_bundle_unchanged', sha(ROOT / 'output/hgtnet_nwpu_formal_3090_20261103_v1.pyz') ==
      'b2bc9077cf37e75c83fbb72e7c4047d2fb9ede04db8fb4e4a56344500ff572e1')

plot = read(HERE / 'figures/plot_manifest.json')
check('twelve_final_figures', len(plot['artifacts']) == 12)
check('plot_source', sha(HERE / 'figures/plot_diagnostics.py') == plot['script_sha256'])
for rel, digest in plot['source_files'].items():
    check('plot_input:' + rel, sha(ROOT / rel) == digest)
for item in plot['artifacts']:
    path = HERE / 'figures' / item['filename']
    check('figure:' + item['filename'], path.stat().st_size == item['bytes'] and sha(path) == item['sha256'])
for item in read(HERE / 'figures/visual_validation.json')['records']:
    check('render:' + item['pdf'], sha(ROOT / item['render']) == item['render_sha256'] and
          item['pages'] == 1 and item['visually_inspected'] is True and item['clipping_or_overlap_found'] is False)

table = ROOT / 'results/external_baselines_nwpu_seed42_20261030.csv'
before = HERE / 'before_update/results/external_baselines_nwpu_seed42_20261030.csv'
old_lines, new_lines = before.read_bytes().splitlines(), table.read_bytes().splitlines()
check('all_non_hgt_lines_unchanged', len(old_lines) == len(new_lines) == 6 and
      all(a == b for a, b in zip(old_lines, new_lines) if not a.lower().startswith(b'hgtnet')))
old_rows = list(csv.DictReader(io.StringIO(before.read_text())))
new_rows = list(csv.DictReader(io.StringIO(table.read_text())))
for i, (old, new) in enumerate(zip(old_rows, new_rows)):
    check('all_original_values:' + str(i), all(old[k] == new[k] for k in old if k not in ('correct_basis', 'evidence')))
hgt = next(r for r in new_rows if r['model'].lower().startswith('hgtnet'))
check('hgt_verified_basis', hgt['correct_basis'] == 'locally_recomputed_from_returned_test_predictions' and
      hgt['evidence'] == 'results/evidence/hgtnet_results_return_20261104/files/test.json')
record = read(ROOT / 'results/hgtnet_nwpu_seed42_20261103.json')
analysis = read(ROOT / 'results/hgtnet_nwpu_seed42_analysis_20261104.json')
check('record_promoted', record['status'] == 'DOWNLOADED_RESULTS_METRICS_HISTORY_LOCALLY_RECOMPUTED' and
      record['prediction_audit']['status'] == 'verified')
check('record_metrics_agree', record['metrics'] == analysis['metrics'] and analysis['correct'] == 23718 and analysis['errors'] == 1482)
docs = [ROOT / 'results/hgtnet_nwpu_seed42_analysis_20261104.md',
        ROOT / 'results/hgtnet_nwpu_seed42_20261103.md', HERE / 'README.md']
for doc in docs:
    for link in re.findall(r'!?\[[^]]*\]\(([^)]+)\)', doc.read_text()):
        if not link.startswith(('https:', '#')):
            check('link:' + str(doc.relative_to(ROOT)) + ':' + link, (doc.parent / link).exists())
plain = (ROOT / 'output/hgtnet_seed42_verified_summary_20261104.txt').read_text()
check('copyable_without_table_or_tabs', '\t' not in plain and not any(line.strip().startswith('|') for line in plain.splitlines()))
check('diary_current', '## 2026-11-04 — HGTNet原始结果回收' in (ROOT / 'research_diary.md').read_text())
diff = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
check('git_diff_check', diff.returncode == 0)
new_json(HERE / 'final_validation.json', {'passed': True, 'count': len(checks), 'checks': checks,
         'scope': 'Final identity, records, documentation, existing numerical audits and visually reviewed figure artifacts; no new training.'})

selected = set()
for path in (ROOT / 'results').glob('hgtnet_nwpu_seed42_*20261104.*'):
    if path.suffix in ('.csv', '.json', '.md'):
        selected.add(path)
selected.add(table)
selected.add(ROOT / 'output/hgtnet_seed42_verified_summary_20261104.txt')
for name in ['README.md', 'archive_receipt.json', 'reported_results.tar.gz', 'analyze_results.py',
             'validation.json', 'analysis_stdout.json', 'record_update.json', 'final_validation.json', 'verify_and_package.py']:
    selected.add(HERE / name)
for subdir in ['files', 'review_predictions', 'review_history', 'figures']:
    for path in (HERE / subdir).rglob('*'):
        if path.is_file() and not {'draft_01', '__pycache__'} & set(path.relative_to(HERE).parts):
            selected.add(path)
manifest = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(selected)]
new_json(HERE / 'bundle_manifest.json', {'files': manifest, 'payload_count': len(manifest),
         'scope': 'Review data, figures, raw returned JSON, audit receipts and scripts. Scripts require the original project; this is not a standalone model runtime.'})
selected.add(HERE / 'bundle_manifest.json')
bundle = ROOT / 'output/hgtnet_seed42_analysis_assets_20261104.zip'
with zipfile.ZipFile(bundle, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(selected):
        archive.write(path, str(path.relative_to(ROOT)))
with zipfile.ZipFile(bundle) as archive:
    check('zip_crc', archive.testzip() is None)
    check('zip_inventory', set(archive.namelist()) == {str(p.relative_to(ROOT)) for p in selected})
    for item in manifest:
        check('zip_sha:' + item['path'], hashlib.sha256(archive.read(item['path'])).hexdigest() == item['sha256'])
new_json(HERE / 'delivery_receipt.json', {'passed': True, 'checks_count': len(checks),
         'bundle': str(bundle.relative_to(ROOT)), 'bytes': bundle.stat().st_size, 'sha256': sha(bundle),
         'members': len(selected), 'crc_and_all_member_hashes_verified': True})
print(json.dumps({'passed': True, 'checks_count': len(checks), 'bundle': str(bundle),
                  'bytes': bundle.stat().st_size, 'members': len(selected), 'sha256': sha(bundle)}))
