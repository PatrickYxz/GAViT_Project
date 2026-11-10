"""Read-only audit of explicitly listed historical research artifacts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as handle:
        value = hashlib.sha256()
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            value.update(block)
        return value.hexdigest()


def check(path, expected):
    target = ROOT / path
    actual = digest(target) if target.is_file() else None
    return {'path': path, 'expected_sha256': expected, 'actual_sha256': actual,
            'exists': target.is_file(), 'bytes': target.stat().st_size if target.is_file() else None,
            'matches': actual == expected}


def group(reference, records):
    return {'reference': reference, 'reference_sha256': digest(ROOT / reference),
            'count': len(records), 'matches': sum(item['matches'] for item in records),
            'missing': sum(not item['exists'] for item in records),
            'mismatches': sum(item['exists'] and not item['matches'] for item in records),
            'records': records}


def main():
    baseline_name = 'results/evidence/external_baselines_20261027/baseline_state.json'
    baseline = json.loads((ROOT / baseline_name).read_text())
    metric_records = [check(item['path'], item['sha256']) for item in baseline['protected_artifacts']]
    entrypoint_name = 'results/evidence/external_baselines_20261027/entrypoint_comparison.json'
    entrypoints = json.loads((ROOT / entrypoint_name).read_text())
    source_records = []
    for item in entrypoints:
        source_records.append(check(item['path'], item['root_sha256']))
        source_records.append(check('.worktrees/sparse-hybrid-4n-top2/' + item['path'], item['research_worktree_sha256']))
    dfagcn_name = 'results/evidence/dfagcn_formal_resource_fix_20261030/protected_inputs.json'
    dfagcn = json.loads((ROOT / dfagcn_name).read_text())
    dfagcn_records = [check(path, expected) for path, expected in dfagcn.items()]
    groups = {
        'historical_metrics': group(baseline_name, metric_records),
        'legacy_root_and_research_sources': group(entrypoint_name, source_records),
        'dfagcn_sources_recipes_and_old_bundles': group(dfagcn_name, dfagcn_records),
    }
    current_paths = [
        'results/dfagcn_nwpu_seed42_20261030.json',
        'results/dfagcn_nwpu_seed42_20261030.md',
        'results/dfagcn_nwpu_seed42_analysis_20261031.json',
        'results/dfagcn_nwpu_seed42_confusion_matrix_20261031.csv',
        'results/dfagcn_nwpu_seed42_confusion_pairs_20261031.csv',
        'results/dfagcn_nwpu_seed42_confusions_20261031.csv',
        'results/dfagcn_nwpu_seed42_error_analysis_20261031.md',
        'results/dfagcn_nwpu_seed42_error_analysis_20261031.txt',
        'results/dfagcn_nwpu_seed42_errors_20261031.csv',
        'results/dfagcn_nwpu_seed42_per_class_20261031.csv',
        'results/dfagcn_nwpu_seed42_predictions_20261031.csv',
        'results/external_baselines_nwpu_seed42_20261030.csv',
        'docs/supervisor_email_draft_20261012.md',
        'docs/supervisor_email_draft_20261012_plaintext.txt',
        'docs/supervisor_progress_report_20261012.md',
        'docs/supervisor_progress_report_20261012_plaintext.txt',
        'output/dfagcn_nwpu_formal_3080_20261030_fix1.pyz',
    ]
    current = []
    for path in current_paths:
        target = ROOT / path
        current.append({'path': path, 'sha256': digest(target) if target.is_file() else None,
                        'exists': target.is_file(), 'bytes': target.stat().st_size if target.is_file() else None,
                        'historical_expected_sha256': None, 'historical_preservation_verified': False})
    report = {
        'schema': 'hgtnet-historical-artifact-preservation-audit-v1',
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'read-only stream hashing of explicitly listed paths; no data or whole-repository scan',
        'root': str(ROOT),
        'groups': groups,
        'total_historical_paths': sum(value['count'] for value in groups.values()),
        'total_matches': sum(value['matches'] for value in groups.values()),
        'total_mismatches': sum(value['mismatches'] for value in groups.values()),
        'total_missing': sum(value['missing'] for value in groups.values()),
        'all_historical_hashes_match': all(item['matches'] for value in groups.values() for item in value['records']),
        'current_snapshot_only': {
            'note': 'No pre-HGTNet-turn snapshot was available for these newer DFAGCN analyses/advisor drafts/fix1 bundle. These are current hashes, not proof of preservation since turn start.',
            'count': len(current), 'records': current,
        },
    }
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = [
        '# 既有产物保护审计', '',
        '仅对指定路径做只读流式 SHA256 核对，没有修改既有文件，没有扫描原始数据或全仓库。', '',
        f"历史快照共 {report['total_historical_paths']} 条：匹配 {report['total_matches']}，缺失 {report['total_missing']}，不匹配 {report['total_mismatches']}。", '',
    ]
    for name, value in groups.items():
        lines.append(f"- {name}: {value['matches']}/{value['count']} 匹配；依据 {value['reference']}。")
    lines.extend(['', 'root 与 research worktree 的源码分别对照各自历史哈希；没有要求两者字节相同。', '',
                  '另记录17个较新 DFAGCN 分析结果、导师稿及 fix1 包的当前哈希。这些文件没有本轮实施前的冻结快照，因此不声称已验证它们自本轮开始未变。', '',
                  '逐路径 expected/actual/counts/matches 见同目录 audit.json。'])
    (OUT / 'audit.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({key: report[key] for key in ('total_historical_paths', 'total_matches', 'total_mismatches', 'total_missing', 'all_historical_hashes_match')}, indent=2))


if __name__ == '__main__':
    main()
