"""Formal selection/recovery and real JPEG CPU fixtures, never GPU/NWPU results."""
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
from unittest.mock import patch
import zipfile

from test_hgtnet_nwpu_data import manifest_fixture, digest, image_record

ROOT=Path(__file__).resolve().parents[1]


def api():
    try:return importlib.import_module('external_baselines.hgtnet.formal_data')
    except ModuleNotFoundError as exc:raise AssertionError('formal data adapter missing') from exc


class FormalDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def manifest(self,value=None):
        path=self.root/'manifest.json';path.write_text(json.dumps(value or manifest_fixture()));return path

    def test_full_manifest_counts_disjoint_sorted_copies(self):
        path=self.manifest();data=api().prepare_inputs(path,expected_sha256=digest(path));rows=data['selected']
        self.assertEqual(data['counts'],dict(train=5040,val=1260,pool=6300,test=25200))
        self.assertEqual(data['manifest_sha256'],digest(path));self.assertEqual(len(data['class_names']),45)
        for split,count in [('train',112),('val',28),('test',560)]:
            self.assertEqual(Counter(r['label'] for r in rows[split]),dict.fromkeys(range(45),count))
            self.assertEqual([r['path'] for r in rows[split]],sorted(r['path'] for r in rows[split]))
        self.assertEqual(len({r['path'] for r in rows['train']+rows['val']+rows['test']}),31500)
        self.assertEqual([r['path'] for r in rows['pool']],sorted(r['path'] for r in rows['train']+rows['val']))
        rows['train'][0]['label']=33
        self.assertEqual(rows['pool'][0]['label'],0)

    def test_manifest_frozen_sha_and_structural_failures(self):
        path=self.manifest()
        with self.assertRaisesRegex(ValueError,'SHA'):api().prepare_inputs(path)
        for mode in ['counts','path','label','duplicate','split']:
            value=manifest_fixture()
            if mode=='counts':value['counts']['test']=25199
            elif mode=='path':value['records'][0]['path']='../outside.jpg'
            elif mode=='label':value['records'][0]['label']=True
            elif mode=='duplicate':value['records'][1]=copy.deepcopy(value['records'][0])
            else:value['records'][0]['split']='pool'
            path=self.manifest(value)
            with self.subTest(mode=mode),self.assertRaises(ValueError):api().prepare_inputs(path,expected_sha256=digest(path))

    def test_test_rows_allowed_but_duplicate_invalid_label_and_symlink_rejected(self):
        folder=self.root/'class_00';folder.mkdir();(folder/'a.jpg').write_bytes(b'a')
        row=image_record('class_00/a.jpg',b'a',split='test');self.assertTrue(api().validate_selected(self.root,[row]))
        for rows in [[row,row],[dict(row,label=True)],[dict(row,label=45)],[dict(row,path='../outside.jpg')],[dict(row,split='pool')]]:
            with self.subTest(rows=rows),self.assertRaises(ValueError):api().validate_selected(self.root,rows)
        link=self.root/'linked';link.mkdir();(link/'class_00').symlink_to(folder,target_is_directory=True)
        with self.assertRaises(ValueError):api().validate_selected(link,[row])

    def zip_fixture(self,extra=None):
        archive=self.root/'fixture.zip'
        with zipfile.ZipFile(archive,'w') as stream:
            stream.writestr('NWPU-RESISC45/class_00/a.jpg',b'a')
            stream.writestr('NWPU-RESISC45/class_01/b.jpg',b'b')
            stream.writestr('NWPU-RESISC45/class_00/unused.jpg',b'unused')
            if extra:extra(stream)
        rows=[image_record('class_00/a.jpg',b'a',split='test'),image_record('class_01/b.jpg',b'b',label=1,split='test')]
        return archive,rows

    def test_only_selected_test_recovery_truthful_receipt_and_explicit_resume(self):
        archive,rows=self.zip_fixture();destination=self.root/'test'
        output=api().recover_selected(archive,rows,destination,expected_sha256=digest(archive))
        self.assertEqual(output,destination)
        receipt=json.loads((output/'recovery.json').read_text())
        self.assertEqual(receipt['test_images_extracted'],2);self.assertEqual(receipt['selected_images'],2)
        self.assertTrue(receipt['complete']);self.assertEqual(receipt['archive_sha256'],digest(archive))
        self.assertFalse((output/'class_00/unused.jpg').exists())
        with self.assertRaises(FileExistsError):api().recover_selected(archive,rows,destination,expected_sha256=digest(archive))
        self.assertEqual(api().recover_selected(archive,rows,destination,expected_sha256=digest(archive),resume=True),destination)
        (output/'class_00/a.jpg').write_bytes(b'changed')
        with self.assertRaises(ValueError):api().recover_selected(archive,rows,destination,expected_sha256=digest(archive),resume=True)

    def test_interrupted_receipt_write_keeps_images_without_partial_final_receipt(self):
        archive,rows=self.zip_fixture();destination=self.root/'receipt_interrupted'
        def interrupted_dump(value,stream,**kwargs):
            stream.write('{"schema":')
            raise OSError('receipt write interrupted')
        with patch.object(api().json,'dump',interrupted_dump):
            with self.assertRaisesRegex(OSError,'receipt write interrupted'):
                api().recover_selected(archive,rows,destination,expected_sha256=digest(archive))
        self.assertTrue(destination.is_dir())
        self.assertEqual((destination/'class_00/a.jpg').read_bytes(),b'a')
        self.assertEqual((destination/'class_01/b.jpg').read_bytes(),b'b')
        self.assertFalse((destination/'recovery.json').exists())
        self.assertEqual(list(destination.glob('recovery.json.*.tmp')),[])
        self.assertTrue(api().validate_selected(destination,rows))

    def test_resume_rejects_foreign_or_changed_selection_receipt(self):
        archive,rows=self.zip_fixture();destination=self.root/'pool'
        output=api().recover_selected(archive,rows,destination,expected_sha256=digest(archive))
        receipt=json.loads((output/'recovery.json').read_text());receipt['test_images_extracted']=0
        (output/'recovery.json').write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):api().recover_selected(archive,rows,destination,expected_sha256=digest(archive),resume=True)
        with self.assertRaises(ValueError):api().recover_selected(archive,rows[:1],destination,expected_sha256=digest(archive),resume=True)

    def test_recovery_resume_rejects_foreign_directory_and_symlink_receipt(self):
        archive,rows=self.zip_fixture();foreign=self.root/'foreign';foreign.mkdir()
        with self.assertRaises(ValueError):api().recover_selected(archive,rows,foreign,expected_sha256=digest(archive),resume=True)
        output=api().recover_selected(archive,rows,self.root/'owned',expected_sha256=digest(archive))
        stored=self.root/'external_receipt.json';stored.write_bytes((output/'recovery.json').read_bytes())
        (output/'recovery.json').unlink();(output/'recovery.json').symlink_to(stored)
        with self.assertRaises(ValueError):api().recover_selected(archive,rows,output,expected_sha256=digest(archive),resume=True)
        linked=self.root/'linked_dest';linked.symlink_to(output,target_is_directory=True)
        with self.assertRaises(ValueError):api().recover_selected(archive,rows,linked,expected_sha256=digest(archive),resume=True)

    def test_archive_unsafe_ambiguous_or_image_hash_mismatch_has_no_complete_receipt(self):
        for mode in ['unsafe','ambiguous','symlink','hash','digest']:
            def extra(stream):
                if mode=='unsafe':stream.writestr('../outside.jpg',b'x')
                elif mode=='ambiguous':stream.writestr('class_00/a.jpg',b'a')
                elif mode=='symlink':
                    item=zipfile.ZipInfo('class_00/link.jpg');item.create_system=3;item.external_attr=0o120777<<16
                    stream.writestr(item,'a.jpg')
            archive,rows=self.zip_fixture(extra);destination=self.root/mode
            if mode=='hash':rows[0]['sha256']='0'*64
            expected='0'*64 if mode=='digest' else digest(archive)
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                api().recover_selected(archive,rows,destination,expected_sha256=expected)
            self.assertFalse((destination/'recovery.json').exists())
        self.assertFalse((self.root/'outside.jpg').exists())

    def test_preparation_recovery_and_validation_never_import_ml(self):
        path=self.manifest()
        script='''import hashlib,importlib.abc,json,sys,tempfile,zipfile
from pathlib import Path
class BlockML(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','torchvision','numpy','PIL'}:raise AssertionError('ML import '+fullname)
sys.meta_path.insert(0,BlockML())
from external_baselines.hgtnet.formal_data import prepare_inputs,validate_selected,recover_selected
assert prepare_inputs(sys.argv[1],expected_sha256=sys.argv[2])['counts']['test']==25200
with tempfile.TemporaryDirectory() as d:
 root=Path(d);archive=root/'fixture.zip'
 with zipfile.ZipFile(archive,'w') as z:z.writestr('class_00/a.jpg',b'a')
 rows=[dict(path='class_00/a.jpg',label=0,split='test',sha256=hashlib.sha256(b'a').hexdigest())]
 out=recover_selected(archive,rows,root/'images',expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
 assert validate_selected(out,rows)
print('stdlib formal data passed')
'''
        done=subprocess.run([sys.executable,'-c',script,str(path),digest(path)],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(done.returncode,0,done.stderr);self.assertIn('stdlib formal data passed',done.stdout)

    def jpeg(self,split='train'):
        from PIL import Image
        import numpy as np
        path=self.root/'class_00/pattern.jpg';path.parent.mkdir(exist_ok=True)
        image=np.zeros((256,256,3),dtype=np.uint8);image[:,:,0]=np.arange(256)[None,:]
        image[:,:,1]=np.arange(256)[:,None];image[:,:,2]=64
        Image.fromarray(image).save(path,quality=100,subsampling=0)
        return image_record('class_00/pattern.jpg',path.read_bytes(),split=split)

    def test_eval_exact_bilinear256_center224_imagenet_and_test_order(self):
        import torch
        import numpy as np
        from PIL import Image
        row=self.jpeg('test');dataset=api().make_dataset(self.root,[row],training=False)
        actual,label=dataset[0]
        with Image.open(self.root/row['path']) as source:
            cropped=source.convert('RGB').resize((256,256),Image.Resampling.BILINEAR).crop((16,16,240,240))
            expected=torch.from_numpy(np.array(cropped)).permute(2,0,1).float()/255
        expected=(expected-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        self.assertTrue(torch.equal(actual,expected));self.assertTrue(torch.equal(dataset[0][0],actual))
        self.assertEqual(label.dtype,torch.long);self.assertEqual(label.item(),0)

    def test_training_transform_seeded_random_crop_flip_real_tensor(self):
        import torch
        row=self.jpeg();dataset=api().make_dataset(self.root,[row],training=True)
        torch.manual_seed(42);actual,label=dataset[0]
        torch.manual_seed(42);repeated,_=dataset[0]
        self.assertTrue(torch.equal(actual,repeated));self.assertEqual(actual.shape,(3,224,224))
        self.assertEqual(actual.dtype,torch.float32);self.assertTrue(torch.isfinite(actual).all())
        self.assertFalse(torch.equal(actual,api().make_dataset(self.root,[row],training=False)[0][0]))
        with self.assertRaises(ValueError):api().make_dataset(self.root,[dict(row,split='test')],training=True)

    def test_dataset_owns_snapshot_and_checks_same_read_bytes_before_decode(self):
        import torch
        row=self.jpeg();rows=[row];dataset=api().make_dataset(self.root,rows,training=False)
        row['label']=40;rows.clear();self.assertEqual(dataset[0][1].item(),0)
        path=self.root/'class_00/pattern.jpg';path.write_bytes(b'invalid changed bytes')
        with self.assertRaisesRegex(ValueError,'SHA'):dataset[0]
        with self.assertRaisesRegex(ValueError,'SHA'):api().make_dataset(self.root,[dict(row,label=0)])

    def test_approved_bytes_are_decoded_even_if_path_changes_after_the_read(self):
        import torch
        row=self.jpeg();dataset=api().make_dataset(self.root,[row],training=False)
        expected,_=dataset[0];image_path=self.root/row['path'];original=Path.read_bytes
        def racing_read(path):
            raw=original(path)
            if path.resolve()==image_path.resolve():path.write_bytes(b'changed after read before decoder')
            return raw
        with patch.object(Path,'read_bytes',racing_read):
            actual,_=dataset[0]
        self.assertTrue(torch.equal(actual,expected))
        with self.assertRaisesRegex(ValueError,'SHA'):dataset[0]


if __name__=='__main__':unittest.main()
