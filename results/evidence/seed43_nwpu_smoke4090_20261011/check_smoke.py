"""Read-only gate before the already authorized NWPU43 formal queue; no Torch."""
import importlib.metadata
import math
import sys
from pathlib import Path


def check_smoke(ns, smoke):
    read, require, digest = ns['read'], ns['require'], ns['digest']
    require((smoke / 'run.exit').read_text().strip() == 'exit_code=0', 'Smoke exit missing/nonzero')
    r = read(smoke / 'run.json')
    require(r['status'] == 'complete' and r['phase'] == 'smoke' and r['model'] == 'gavit'
            and r['dataset'] == 'NWPU-RESISC45' and r['seed'] == 43, 'Wrong/incomplete smoke identity')
    require(r['counts'] == {'train': 360, 'val': 90} and r['final_epoch'] == r['best_epoch'] == 1,
            'Wrong smoke sample/epoch counts')
    require(r['manifest_sha256'] == ns['MANIFEST_SHA'] and r['pretrained_sha256'] == ns['PRE_SHA'],
            'Smoke inputs changed')
    expected_training = {'seed': 43, 'batch_size': 32, 'workers': 2, 'optimizer': 'AdamW',
                         'lr': .0003, 'weight_decay': .0001, 'scheduler_t_max': 30,
                         'loss': 'CrossEntropyLoss', 'precision': 'float32_no_amp', 'tf32': False}
    require(all(r['training'][key] == value for key, value in expected_training.items()), 'Wrong smoke recipe')
    require(r['architecture']['grouping'] == 'soft_prototype' and r['architecture']['num_classes'] == 45
            and r['architecture']['edge_type'] == 'sparse_hybrid'
            and r['architecture']['integration'] == 'token_feedback', 'Wrong smoke architecture')
    require(r['code']['git'] == {'commit': ns['COMMIT'][:7], 'dirty': False}, 'Wrong smoke source')
    for relative, sha in r['code']['files_sha256'].items():
        path = (ns['REPO'] / relative).resolve()
        require(path.is_relative_to(ns['REPO'].resolve()) and digest(path) == sha, 'Smoke source file changed')
    hw = r['hardware']
    require(hw['device'] == 'cuda' and hw['device_name'] == ns['GPU_NAME'] and hw['torch_num_threads'] == 8,
            'Wrong smoke device/threads')
    require(hw['python'] == sys.version and all(importlib.metadata.version(k) == v for k, v in hw['packages'].items()),
            'Software changed since smoke')
    require(r['graph_capture'] == {'edges_per_image': 80, 'spatial_edges': 48, 'feature_edges': 32,
                                 'actual_forward_checked': True}, 'Smoke graph not verified')
    require(math.isfinite(r['reload_max_logit_diff']) and 0 <= r['reload_max_logit_diff'] <= 1e-5,
            'Smoke reload mismatch')
    best = smoke / 'best.pth'
    require(digest(best) == r['checkpoint_sha256'], 'Smoke checkpoint bytes changed')
    meta = read(smoke / 'best.meta.json')
    require(meta['checkpoint_sha256'] == r['checkpoint_sha256'] and meta['execution']['phase'] == 'smoke'
            and meta['data']['counts'] == r['counts'] and meta['training'] == r['training']
            and meta['code'] == r['code'] and meta['architecture'] == r['architecture'], 'Smoke metadata mismatch')
    history = read(smoke / 'history.json')
    require(len(history) == 1 and history[0]['epoch'] == 1 and math.isfinite(history[0]['train_loss']),
            'Incomplete/nonfinite smoke history')
    require(all((smoke / name).is_file() for name in ('last.pth', 'last.train_state.pth',
            'assignment_probe/epoch_000.pt', 'assignment_probe/epoch_001.pt',
            'assignment_probe/best.pt', 'assignment_probe/final.pt')), 'Missing smoke artifacts')
    return {'status': 'NWPU43_SMOKE_ARTIFACTS_PASS', 'checkpoint_sha256': r['checkpoint_sha256'],
            'reload_max_logit_diff': r['reload_max_logit_diff'], 'hardware': hw, 'counts': r['counts']}
