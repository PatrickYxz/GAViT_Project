"""Local, exclusive creation of the two-member standard-library inventory bundle."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'tools/inspect_hgtnet_aid_inputs.py'
BUNDLE = ROOT / 'output/hgtnet_aid_inputs_20261104.pyz'
source = SOURCE.read_bytes()
identity = dict(schema='hgtnet-aid-inputs-source-identity-v1',
                source_path='tools/inspect_hgtnet_aid_inputs.py',
                source_sha256=hashlib.sha256(source).hexdigest(), source_size_bytes=len(source),
                purpose='read_only_AID_input_inventory', training_entry_available=False,
                manifest_sha256='30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22',
                weights_sha256='68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893')
identity_bytes = (json.dumps(identity, sort_keys=True, indent=2) + '\n').encode()
# x refuses existing regular files, symlinks and directories. No overwrite mode.
with zipfile.ZipFile(BUNDLE, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    for name, data in [('__main__.py', source), ('source_identity.json', identity_bytes)]:
        info = zipfile.ZipInfo(name, date_time=(2026, 11, 4, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, data)
print(json.dumps(dict(bundle=str(BUNDLE), bundle_sha256=hashlib.sha256(BUNDLE.read_bytes()).hexdigest(),
                     bundle_size_bytes=BUNDLE.stat().st_size, source_identity=identity), indent=2, sort_keys=True))
