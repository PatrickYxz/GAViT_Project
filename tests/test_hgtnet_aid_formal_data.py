"""AID full-metadata and generated RGB600 CPU image fixtures, not real assets."""
from collections import Counter
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tests.test_hgtnet_aid_inputs import manifest_fixture


def api():
    try: return importlib.import_module('external_baselines.hgtnet.aid_formal_data')
    except ModuleNotFoundError as exc: raise AssertionError('AID formal data interface must be implemented') from exc


class AidFormalDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir='/private/tmp'); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def prepare(self, value=None, raw=None):
        raw = json.dumps(manifest_fixture() if value is None else value).encode() if raw is None else raw
        path = self.root/'manifest.json'; path.write_bytes(raw)
        return api().prepare_inputs(path, expected_sha256=hashlib.sha256(raw).hexdigest())

    def image(self, split='train', mode='RGB', size=(600,600)):
        from PIL import Image
        path = self.root/'Airport'/'pattern.png'; path.parent.mkdir(exist_ok=True)
        image = Image.new(mode,size,255 if mode=='L' else (255,0,0))
        if mode == 'RGB':
            for x in range(600): image.paste((x%256,64,128),(x,0,min(x+1,size[0]),size[1])) if x<size[0] else None
        image.save(path); raw=path.read_bytes()
        return path, dict(path='Airport/pattern.png',label=0,split=split,sha256=hashlib.sha256(raw).hexdigest(),pixel_sha256='1'*64)

    def test_selects_all_original_assignments_with_unequal_counts_and_duplicate_group(self):
        value=manifest_fixture(); group=[r for r in value['records'] if r['path'].startswith('Industrial/')][:2]
        group[1]['pixel_sha256']=group[0]['pixel_sha256']
        value['duplicates']=dict(unique_images=9999,extra_copies=1,groups=[dict(pixel_sha256=group[0]['pixel_sha256'],
            label=group[0]['label'],paths=[r['path'] for r in group],split='train')])
        prepared=self.prepare(value); selected=prepared['selected']
        self.assertEqual(prepared['counts'],dict(train=4000,val=1000,pool=5000,test=5000))
        self.assertEqual(Counter(r['label'] for r in selected['train']),{i:80 if i<10 else 120 if i<20 else 200 for i in range(30)})
        self.assertEqual(len(selected['pool']),5000); self.assertEqual(len(selected['test']),5000)
        self.assertEqual({r['path'] for r in selected['train']+selected['val']+selected['test']},{r['path'] for r in value['records']})
        for rows in selected.values(): self.assertEqual([r['path'] for r in rows],sorted(r['path'] for r in rows))
        duplicate=[r for r in selected['train'] if r['path'] in {r['path'] for r in group}]
        self.assertEqual(len(duplicate),2);self.assertEqual(duplicate[0]['pixel_sha256'],duplicate[1]['pixel_sha256'])
        selected['train'][0]['label']=29;self.assertEqual(selected['pool'][0]['label'],0)

    def test_manifest_sha_strict_json_and_original_metadata_are_required(self):
        path=self.root/'manifest.json';path.write_text(json.dumps(manifest_fixture()))
        with self.assertRaises(ValueError): api().prepare_inputs(path)
        for raw in [b'{"dataset":"AID","dataset":"AID"}',b'NaN',b'[]']:
            with self.subTest(raw=raw),self.assertRaises(ValueError):self.prepare(raw=raw)
        for key,bad in [('split','pool'),('label',True),('path','../bad.jpg'),('pixel_sha256','bad')]:
            value=manifest_fixture();value['records'][0][key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):self.prepare(value)

    def test_test_bytes_require_explicit_gate_and_can_never_enter_training(self):
        path,row=self.image('test')
        for call in [lambda:api()._records([row]),lambda:api().validate_selected(self.root,[row]),
                     lambda:api().make_dataset(self.root,[row]),
                     lambda:api().make_dataset(self.root,[row],training=True,allow_test=True)]:
            with mock.patch.object(api().inspector,'open_regular_readonly',side_effect=AssertionError('test opened before gate')):
                with self.assertRaises(ValueError):call()
        result=api().validate_selected(self.root,[row],allow_test=True)
        self.assertEqual(result,dict(checked_count=1,verified_count=1,test_image_bytes_read=path.stat().st_size,verified_bytes=path.stat().st_size))
        self.assertEqual(api().make_dataset(self.root,[row],allow_test=True)[0][1].item(),0)

    def test_unsafe_paths_symlinks_changed_bytes_and_cross_split_pixels_fail(self):
        path,row=self.image()
        for relative in ['../a.png','/a.png','Airport//pattern.png','Airport/./pattern.png','Airport\\pattern.png']:
            with self.subTest(relative=relative),self.assertRaises(ValueError):api().validate_selected(self.root,[dict(row,path=relative)])
        with self.assertRaises(ValueError):api()._records([row,dict(row,path='Airport/b.png',split='val')])
        path.write_bytes(b'changed')
        with self.assertRaises(ValueError):api().validate_selected(self.root,[row])
        path.unlink();path.symlink_to(self.root/'missing')
        with self.assertRaises(ValueError):api().validate_selected(self.root,[row])
        path.unlink();path.parent.rename(self.root/'real');path.parent.symlink_to(self.root/'real',target_is_directory=True)
        with self.assertRaises(ValueError):api().validate_selected(self.root,[row])

    def test_eval_matches_bilinear256_center224_and_owns_records(self):
        import torch
        from PIL import Image
        path,row=self.image(); rows=[copy.deepcopy(row)]; dataset=api().make_dataset(self.root,rows)
        actual,label=dataset[0]
        with Image.open(path) as source:
            image=source.resize((256,256),Image.Resampling.BILINEAR).crop((16,16,240,240))
            expected=torch.frombuffer(bytearray(image.tobytes()),dtype=torch.uint8).reshape(224,224,3).permute(2,0,1).float()/255
        expected=(expected-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(label.dtype,torch.long)
        rows[0]['label']=29; exposed=dataset.records;exposed[0]['label']=29
        self.assertEqual(dataset.paths,('Airport/pattern.png',));self.assertEqual(dataset.labels,(0,));self.assertEqual(dataset.records,[row])

    def test_training_uses_seeded_crop_flip_and_rejects_non_rgb600(self):
        import torch
        _,row=self.image();dataset=api().make_dataset(self.root,[row],training=True)
        torch.manual_seed(42);actual,_=dataset[0];torch.manual_seed(42);repeated,_=dataset[0]
        self.assertTrue(torch.equal(actual,repeated));self.assertFalse(torch.equal(actual,api().make_dataset(self.root,[row])[0][0]))
        self.assertEqual(actual.shape,(3,224,224));self.assertTrue(torch.isfinite(actual).all())
        for mode,size in [('L',(600,600)),('RGB',(224,224))]:
            _,bad=self.image(mode=mode,size=size);dataset=api().make_dataset(self.root,[bad])
            with self.assertRaises(ValueError):dataset[0]

    def test_decoder_consumes_the_identical_hashed_bytes(self):
        from PIL import Image
        path,row=self.image();dataset=api().make_dataset(self.root,[row]);real=Image.open
        def guarded(source,*args,**kwargs):
            self.assertIsInstance(source,io.BytesIO);path.write_bytes(b'changed after hash');return real(source,*args,**kwargs)
        with mock.patch.object(Image,'open',side_effect=guarded):self.assertEqual(dataset[0][1].item(),0)
        with self.assertRaises(ValueError):dataset[0]

    def test_full_metadata_and_byte_checks_are_stdlib_only(self):
        path=self.root/'manifest.json';path.write_text(json.dumps(manifest_fixture()));digest=hashlib.sha256(path.read_bytes()).hexdigest()
        script='''import builtins,sys\noriginal=builtins.__import__\ndef guarded(name,*args,**kwargs):\n if name.split('.')[0] in {'torch','torchvision','PIL','numpy','timm'}: raise AssertionError(name)\n return original(name,*args,**kwargs)\nbuiltins.__import__=guarded\nfrom external_baselines.hgtnet import aid_formal_data as d,aid_formal_protocol\nassert d.prepare_inputs(sys.argv[1],expected_sha256=sys.argv[2])['counts']['pool']==5000\nassert not hasattr(d,'recover_selected')\n'''
        result=subprocess.run([sys.executable,'-c',script,str(path),digest],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__':unittest.main()
