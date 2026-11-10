"""Validate reference identity/link consistency, not scientific correctness or TeX compilation."""
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[3]
folder = Path(__file__).parent
draft = root / 'docs/thesis_related_work_20261006.md'
bib = draft.with_suffix('.bib')
catalog = folder / 'sources.json'
data = json.loads(catalog.read_text())
sources = data['sources']
text = draft.read_text()
body, references = text.split('## 参考文献', 1)
assert len(sources) == 12
assert [r['id'] for r in sources] == [f'R{i}' for i in range(1, 13)]
assert len({r['key'] for r in sources}) == 12
entries = re.findall(r'@(\w+)\{([^,]+),\n(.*?)\n\}', bib.read_text(), re.S)
assert len(entries) == 12
by_key = {key: (kind, fields) for kind, key, fields in entries}
assert set(by_key) == {r['key'] for r in sources}
for r in sources:
    kind, fields = by_key[r['key']]
    assert kind == r['type']
    assert re.search(r'\[' + r['id'] + r'：[^\]]+\]\(https://[^)]+\)', body), r['id']
    assert f"- {r['id']}. " in references and f"`{r['key']}`" in references
    assert f"**{r['title']}.**" in references, r['id']
    for field in ('title', 'author', 'booktitle', 'journal', 'year', 'volume', 'number', 'pages', 'doi', 'url', 'note'):
        if field in r:
            value = str(r[field])
            if field == 'title':
                value = '{' + value + '}'
            assert f'  {field} = {{{value}}}' in fields, (r['id'], field)
    assert fields.count('{') == fields.count('}'), r['id']

links = []
for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
    if target.startswith(('http://', 'https://', '#')):
        continue
    p = (draft.parent / target.split('#')[0]).resolve()
    if p == folder / 'validation.json':
        continue
    assert p.is_file(), target
    links.append(str(p.relative_to(root)))
assert all(marker in text for marker in ('遥感', '图神经网络', '固定网格', 'k-means', '软分配', '不声称是穷尽', '尚未完成', '锁定commit'))
assert not re.search(r'turn\d+(?:search|view)\d+', text)
assert not any(line.rstrip() != line for line in text.splitlines())
source_files = [catalog, draft, bib]
for item in data['implementation_evidence']:
    p = root / item['path']
    assert p.is_file(), p
    source_files.append(p)

result = {
    'date': '2026-10-06', 'status': 'PASS', 'references': len(entries),
    'checks': ['12 unique keys and R1-R12 coverage', 'All exported BibTeX fields match verified metadata catalog',
               'Bibliography titles and citation keys match chapter', 'Balanced generated field braces',
               'Local Markdown/code links exist', 'Three requested themes and scope boundaries present',
               'No internal browser reference IDs or trailing whitespace'],
    'local_link_targets': sorted(set(links)),
    'sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
    'review_scope': 'Primary-source audits by two agents completed and integrated; main agent read current code and reconciled the draft. Follow-up independent draft reviews failed due to usage limit; no independent final review claimed.',
    'limits': ['Not an exhaustive literature review or novelty assessment', 'No external code executed or new GPU experiments',
               'No bibliography-manager import or TeX compilation performed', 'Static citation checks do not independently establish semantic correctness',
               'Primary-page access failures and historical pinned-code scope are recorded in sources.json']
}
(folder / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'references': len(entries), 'local_link_targets': len(set(links)), 'hashed_files': len(source_files)}))
