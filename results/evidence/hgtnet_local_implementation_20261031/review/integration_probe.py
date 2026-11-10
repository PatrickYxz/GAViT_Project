"""Independent small actual-tensor integration probe, not a full-model smoke."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from external_baselines.hgtnet.model import HGTNet
from external_baselines.hgtnet.graph import GraphBlock
from external_baselines.hgtnet.config import config_sha256, load_config
from external_baselines.hgtnet.checkpoint import save_checkpoint, load_checkpoint


torch.set_num_threads(1)
torch.manual_seed(984)
blocks = nn.ModuleList([GraphBlock(4, 20), GraphBlock(4, 20)]).eval()
features = torch.randn(2, 4, 5, 7, requires_grad=True)
traced, trace = HGTNet._graph_group(features, blocks, True)
ordinary, none = HGTNet._graph_group(features, blocks, False)
torch.testing.assert_close(traced, ordinary, rtol=0, atol=0)
assert none == []
manual = features.flatten(2).transpose(1, 2)
for index, block in enumerate(blocks):
    manual, indices = block(manual, return_neighbors=True)
    assert trace[index]['indices_shape'] == [2, 35, 20]
    assert trace[index]['self_edges'] == 0
    assert trace[index]['indices_sha256'] == hashlib.sha256(indices.contiguous().numpy().tobytes()).hexdigest()
torch.testing.assert_close(traced.flatten(2).transpose(1, 2), manual, rtol=0, atol=0)
for index in range(2):
    individual, _ = HGTNet._graph_group(features[index:index+1], blocks, False)
    torch.testing.assert_close(traced[index:index+1], individual)
traced.square().mean().backward()
assert features.grad is not None and torch.isfinite(features.grad).all()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in blocks.parameters())


class TinyIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = load_config(ROOT / 'configs/hgtnet_independent_v0.json')
        self.config_sha256 = config_sha256(self.config)
        self.fc = nn.Linear(3, 2)
        self.register_buffer('integer_buffer', torch.tensor([3, 7]))


with tempfile.TemporaryDirectory() as temporary:
    checkpoint = Path(temporary) / 'new.pt'
    original, restored = TinyIdentity(), TinyIdentity()
    saved = save_checkpoint(original, checkpoint)
    loaded = load_checkpoint(restored, checkpoint, expected_sha256=saved['sha256'])
    for key, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    previous = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    try:
        save_checkpoint(restored, checkpoint)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Existing checkpoint was accepted')
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == previous
    assert loaded['loaded'] and loaded['scope'] == 'local_engineering_only'

print(json.dumps({'passed': True, 'scope': 'small actual CPU tensors; full model smoke owned by parent',
                  'graph_group_trace_equal_to_normal': True, 'trace_digest_matches_actual_indices': True,
                  'graph_batch_isolation': True, 'graph_backward_finite': True,
                  'checkpoint_real_source_roundtrip': True, 'checkpoint_no_overwrite': True}, indent=2))
