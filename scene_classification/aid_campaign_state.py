"""Small, Torch-free commit protocol for the frozen AID campaign.

Only ``latest.json`` accepts an epoch. Immutable transaction directories are
written first; a failed write leaves the preceding accepted epoch intact.
"""
import hashlib
import json
import os
from pathlib import Path
import uuid


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def establish_identity(output, identity, *, resume):
    output = Path(output)
    if output.exists():
        if not resume:
            raise FileExistsError(f'Existing output requires explicit resume: {output}')
        path = output / 'job_identity.json'
        if not path.is_file() or json.loads(path.read_text()) != identity:
            raise ValueError('Existing output identity mismatch or missing identity')
    else:
        if resume:
            raise FileNotFoundError(f'Cannot resume missing output: {output}')
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / 'job_identity.json', identity)


def commit_transaction(phase, epoch, writer):
    phase = Path(phase)
    transactions = phase / 'transactions'
    transactions.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    pending = transactions / ('.pending-' + token)
    pending.mkdir()
    writer(pending)
    files = {p.relative_to(pending).as_posix(): file_digest(p)
             for p in sorted(pending.rglob('*')) if p.is_file()}
    if not files:
        raise ValueError('Empty transaction')
    name = f'epoch-{epoch:03d}-{token}'
    pending.rename(transactions / name)
    prior = json.loads((phase / 'latest.json').read_text()) if (phase / 'latest.json').is_file() else None
    receipt = {'schema_version': 1, 'epoch': epoch, 'transaction': 'transactions/' + name,
               'previous_transaction': prior['transaction'] if prior else None,
               'files_sha256': files}
    atomic_json(phase / 'latest.json', receipt)
    return phase / receipt['transaction'], receipt


def verified_latest(phase):
    phase = Path(phase)
    path = phase / 'latest.json'
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    relative = Path(receipt['transaction'])
    if (relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 2
            or relative.parts[0] != 'transactions'):
        raise ValueError('Invalid transaction path')
    folder = phase / relative
    files = receipt.get('files_sha256', {})
    if not files:
        raise ValueError('Missing transaction hash list')
    for name, expected in files.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Invalid transaction artifact path')
        artifact = folder / rel
        if not artifact.is_file() or file_digest(artifact) != expected:
            raise ValueError(f'Transaction artifact hash mismatch: {artifact}')
    return folder, receipt
