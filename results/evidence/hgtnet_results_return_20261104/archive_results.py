"""Archive a known seven-file gzip tar, reading only regular expected members."""
import hashlib,json,tarfile
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent
source=Path('/Users/patrick/Downloads/hgtnet_seed42_results_20261103.tar.tar')
expected={'test.json','history.json','summary.json','binding.json','initialization.json','checkpoint_reload.json','attempts/0001/runtime.json'}
raw=source.read_bytes();out=HERE/'files'
if out.exists():raise FileExistsError('Do not overwrite a previous results receipt')
archive=HERE/'reported_results.tar.gz'
with archive.open('xb') as f:f.write(raw)
receipt=dict(source=str(source),supplied_extension='.tar.tar',detected_format='gzip-compressed tar',
    bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),members=[])
with tarfile.open(source,'r:*') as tar:
    members=tar.getmembers()
    if len(members)!=len(expected) or {m.name for m in members}!=expected:raise ValueError('Unexpected/duplicate archive members')
    if sum(m.size for m in members)>20*2**20:raise ValueError('Unexpected expanded archive size')
    for m in members:
        p=PurePosixPath(m.name)
        if not m.isfile() or m.issym() or m.islnk() or p.is_absolute() or '..' in p.parts or p.as_posix()!=m.name:
            raise ValueError('Archive member must be a normalized expected regular file')
        data=tar.extractfile(m).read()
        if len(data)!=m.size:raise ValueError('Truncated archive member')
        json.loads(data) # Parse data; no code/instruction is executed.
        path=out/m.name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(data)
        receipt['members'].append(dict(path=m.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
test=next(m for m in receipt['members'] if m['path']=='test.json')
if test['sha256']!='869986974e4801f2234fd2fa3bace383af2f092fd20359fbe1a252d4af7e37ae':raise ValueError('Returned test differs from completed status SHA')
(HERE/'archive_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2,sort_keys=True))
