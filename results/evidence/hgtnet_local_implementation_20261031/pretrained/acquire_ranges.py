"""One bounded parallel HTTPS range attempt; never accepts incomplete/misranged bytes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
OUT = ROOT / 'checkpoints/hgtnet_pretrained_20261031'
URL = 'https://download.pytorch.org/models/swin_b-68c6b09e.pth'
TOTAL = 351654207
PREFIX = OUT / 'swin_b-68c6b09e.attempt1.part'
start = time.monotonic()
report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'url': URL,
          'kind': 'torchvision.Swin_B_Weights.IMAGENET1K_V1', 'is_original_hgtnet_weights': False,
          'required_total_bytes': TOTAL, 'segments': []}
base_bytes = PREFIX.stat().st_size
workers = 16
step = (TOTAL - base_bytes + workers - 1) // workers
ranges = [(index, base_bytes + index * step, min(TOTAL - 1, base_bytes + (index + 1) * step - 1)) for index in range(workers)]
def get_segment(item):
    index, first, last = item
    path = OUT / f'swin_b-68c6b09e.range{index:02d}.part'
    headers = EVIDENCE / f'range{index:02d}.headers'
    log = EVIDENCE / f'range{index:02d}.log'
    with log.open('wb') as stream:
        result = subprocess.run(['curl','--fail','--silent','--show-error','--location','--proto','=https','--tlsv1.2','--connect-timeout','20','--max-time','150','--range',f'{first}-{last}','--dump-header',str(headers),'--output',str(path),URL],stdout=stream,stderr=subprocess.STDOUT)
    header_text = headers.read_text() if headers.exists() else ''
    correct_range = bool(re.search(rf'content-range:\s*bytes\s+{first}-{last}/{TOTAL}\s*$', header_text, re.IGNORECASE | re.MULTILINE))
    size = path.stat().st_size if path.exists() else 0
    return {'index':index, 'first':first, 'last':last, 'exit_code':result.returncode,
            'bytes':size, 'range_header_matched':correct_range,
            'valid':result.returncode == 0 and correct_range and size == last-first+1, 'path':str(path)}
with ThreadPoolExecutor(max_workers=workers) as pool:
    report['segments'] = list(pool.map(get_segment, ranges))
if all(item['valid'] for item in report['segments']):
    assembled = OUT / 'swin_b-68c6b09e.assembled.part'
    with assembled.open('xb') as destination:
        for part in [PREFIX] + [Path(item['path']) for item in report['segments']]:
            with part.open('rb') as source:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
    with assembled.open('rb') as source:
        sha = hashlib.file_digest(source, 'sha256').hexdigest()
    report.update(sha256=sha, bytes=assembled.stat().st_size)
    if sha.startswith('68c6b09e') and assembled.stat().st_size == TOTAL:
        final = OUT / 'swin_b-68c6b09e.pth'
        if final.exists():
            raise RuntimeError('Refusing overwrite')
        assembled.rename(final)
        report.update(status='downloaded_official_candidate_bytes', path=str(final))
    else:
        report['status'] = 'pending_official_candidate_acquisition_hash_mismatch'
else:
    report['status'] = 'pending_official_candidate_acquisition'
report['elapsed_seconds'] = time.monotonic() - start
report['ended_utc'] = datetime.now(timezone.utc).isoformat()
(EVIDENCE / 'acquisition_ranges.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key: value for key, value in report.items() if key != 'segments'},indent=2))
