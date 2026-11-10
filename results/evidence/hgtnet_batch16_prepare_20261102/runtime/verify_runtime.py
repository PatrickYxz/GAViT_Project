"""Local CPU environment check; this is not an HGTNet or GPU measurement."""
import json
import os
import platform
import sys
import time
from pathlib import Path

start = time.perf_counter()
import numpy
import PIL
import torch
import torchvision

expected = {"torch": "2.2.2", "torchvision": "0.17.2", "numpy": "1.26.4", "pillow": "10.3.0"}
actual = {"torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": numpy.__version__, "pillow": PIL.__version__}
assert actual == expected, (actual, expected)
torch.set_num_threads(1)
torch.manual_seed(42)
model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 2)).cpu()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
before = [parameter.detach().clone() for parameter in model.parameters()]
inputs = torch.randn(3, 4, device="cpu")
labels = torch.tensor([0, 1, 0], device="cpu")
outputs = model(inputs)
loss = torch.nn.functional.cross_entropy(outputs, labels)
loss.backward()
assert torch.isfinite(loss).item()
assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in model.parameters())
optimizer.step()
assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
result = {
    "schema": "gavit_hgtnet_local_cpu_runtime_v1",
    "verified": True,
    "scope": "local tiny CPU forward/backward/optimizer only; no HGTNet/GPU/NWPU claim",
    "executable": sys.executable,
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "versions": actual,
    "torch_threads": torch.get_num_threads(),
    "environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "KMP_INIT_AT_FORK")},
    "device": str(outputs.device),
    "output_shape": list(outputs.shape),
    "loss": float(loss.detach()),
    "finite_gradients": True,
    "parameters_updated": True,
    "wall_seconds": time.perf_counter() - start,
}
path = Path(__file__).with_name("runtime_verification.json")
path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
