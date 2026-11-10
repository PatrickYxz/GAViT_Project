"""Local integration only: real Swin, random weights/inputs, no research metric."""
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from models.gavit import GAViT
from tools.diagnose_nwpu_feedback import ALPHAS, capture_batch

torch.set_num_threads(2)
torch.manual_seed(42)
started = time.perf_counter()
rows = {}
for grouping in ('attentive_spatial', 'soft_prototype'):
    model = GAViT(pretrained=False, num_classes=45, grouping=grouping,
                  edge_type='sparse_hybrid', integration='token_feedback').eval()
    x = torch.randn(2, 3, 224, 224)
    cache, checks = capture_batch(model, x)
    checks['actual_scaled_forward_max_logit_diff'] = 0.
    for i, alpha in enumerate(ALPHAS):
        hook = model.feedback_proj.register_forward_hook(lambda m, a, o: o*alpha)
        try:
            with torch.inference_mode():
                actual = model(x)
        finally:
            hook.remove()
        torch.testing.assert_close(actual, cache['scaled_logits'][i], atol=1e-5, rtol=1e-5)
        assert torch.equal(actual.argmax(-1), cache['scaled_logits'][i].argmax(-1))
        checks['actual_scaled_forward_max_logit_diff'] = max(
            checks['actual_scaled_forward_max_logit_diff'],
            float((actual-cache['scaled_logits'][i]).abs().max()))
    cache['classifier_state'] = model.classifier.state_dict()
    cache['identity'] = {'scope': 'random-weight real-Swin CPU integration', 'alphas': ALPHAS}
    with tempfile.TemporaryDirectory(prefix='nwpu-feedback-cache-') as folder:
        path = Path(folder)/'cache.pt'
        torch.save(cache, path)
        restored = torch.load(path, map_location='cpu', weights_only=True)
    head = copy.deepcopy(model.classifier)
    head.load_state_dict(restored['classifier_state'], strict=True)
    with torch.inference_mode():
        replay = torch.stack([head(restored['mean_tokens']+a*restored['mean_feedback']) for a in ALPHAS])
    torch.testing.assert_close(replay, restored['scaled_logits'], atol=1e-5, rtol=1e-5)
    checks['cache_reload_max_logit_diff'] = float((replay-restored['scaled_logits']).abs().max())
    rows[grouping] = checks
    print(grouping, checks, flush=True)
    del model, head, cache, restored, replay, x
report = dict(status='passed', python=platform.python_version(), torch=str(torch.__version__),
              device='cpu', inputs=[2, 3, 224, 224], alphas=ALPHAS, grouping_checks=rows,
              tool_sha256=hashlib.sha256((ROOT/'tools/diagnose_nwpu_feedback.py').read_bytes()).hexdigest(),
              scope='Synthetic input and randomly initialized real Swin/GAViT; no trained server weights, CUDA or accuracy validation.',
              elapsed_seconds=time.perf_counter()-started)
Path(__file__).with_name('real_cpu_validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
