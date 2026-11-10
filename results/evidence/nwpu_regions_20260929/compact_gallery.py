"""Remove repeated embedded PNGs from the completed gallery; no model inference."""
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def compact_gallery(source, expected_cases=90):
    source = Path(source)
    destination = source.with_name('cases_compact.html')
    if destination.exists():
        raise FileExistsError(destination)
    original = source.read_text(encoding='utf-8')
    seen = []

    def compact(match):
        svg = base64.b64decode(match[1], validate=True).decode('utf-8')
        images = re.findall(r'<image\b[^>]*?/>', svg)
        expected = 4 if len(seen) % 2 == 0 else 16
        assert len(images) == expected and len(set(images)) == 1, 'Unexpected SVG image structure'
        assert 'compact-pixels' not in svg
        node = ET.fromstring(images[0])
        assert node.attrib.keys() == {'width', 'height', 'href'}
        assert node.attrib['href'].startswith('data:image/png;base64,')
        definition = '<defs>' + images[0].replace('<image ', '<image id="compact-pixels" ', 1) + '</defs>'
        use = '<use href="#compact-pixels"/>'
        result = svg.replace(images[0], use).replace('>', '>' + definition, 1)
        ET.fromstring(result)
        assert result.replace(definition, '', 1).replace(use, images[0]) == svg
        seen.append(expected)
        return 'data:image/svg+xml;base64,' + base64.b64encode(result.encode()).decode()

    result = re.sub(r'data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)', compact, original)
    assert len(seen) == 2 * expected_cases and original.count('<article>') == expected_cases
    data = result.encode('utf-8')
    with destination.open('xb') as stream:
        stream.write(data)
    report = dict(status='COMPACT_COMPLETE', path=str(destination), cases=expected_cases,
                  svg_count=len(seen), original_bytes=source.stat().st_size, compact_bytes=len(data),
                  compact_sha256=hashlib.sha256(data).hexdigest(), original_preserved=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    compact_gallery(sys.argv[1])
