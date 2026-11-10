"""Offline CPU entrypoint harness: real models/TIFF IO, no download/workers."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import timm
import torch
import torch.utils.data
import utils

torch.set_num_threads(2)
create_model = timm.create_model
loader = torch.utils.data.DataLoader


def offline_model(*args, **kwargs):
    kwargs['pretrained'] = False
    return create_model(*args, **kwargs)


class CpuLoader(loader):
    def __init__(self, *args, **kwargs):
        kwargs.update(num_workers=0, pin_memory=False)
        super().__init__(*args, **kwargs)


timm.create_model = offline_model
torch.utils.data.DataLoader = CpuLoader
original_save = utils.save_epoch_artifacts


def verify_saved_model(model, optimizer, scheduler, path, metadata, **kwargs):
    result = original_save(model, optimizer, scheduler, path, metadata, **kwargs)
    if result[1]:
        from models.gavit import GAViT
        saved = utils.load_checkpoint_metadata(path)
        rng = torch.get_rng_state().clone()
        reconstructed = (GAViT(**saved['architecture'], pretrained=False)
                         if saved['model'] == 'gavit' else type(model)())
        if saved['model'] == 'swin':
            from models.swin_features import build_swin_model_kwargs
            reconstructed.backbone = create_model('swin_tiny_patch4_window7_224',
                                                   **build_swin_model_kwargs(False, None))
        reconstructed.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        reconstructed.eval()
        sample = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
        with torch.no_grad():
            delta = float((model(sample) - reconstructed(sample)).abs().max())
        torch.set_rng_state(rng)
        if delta > 1e-5:
            raise AssertionError(f'Entry reload mismatch {delta}')
        print(f'ENTRY RELOAD PASS {delta}', flush=True)
    return result


utils.save_epoch_artifacts = verify_saved_model
if os.environ.get('SOFT_TEST_STOP_AFTER'):
    from soft_assignment_diagnostics import SoftAssignmentProbe
    original_record = SoftAssignmentProbe.record
    def stop_after_record(self, epoch, **kwargs):
        row = original_record(self, epoch, **kwargs)
        if epoch == int(os.environ['SOFT_TEST_STOP_AFTER']):
            print('CONTROLLED STOP AFTER SAVED EPOCH', epoch, flush=True)
            raise SystemExit(7)
        return row
    SoftAssignmentProbe.record = stop_after_record

entry = ROOT / sys.argv[1]
sys.argv = [str(entry), *sys.argv[2:]]
runpy.run_path(str(entry), run_name='__main__')
