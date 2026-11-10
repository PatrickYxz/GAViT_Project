import copy
import importlib
import tempfile
from pathlib import Path
import unittest
import torch
from torch import nn

class FormalCheckpointTests(unittest.TestCase):
    def modules(self):
        return importlib.import_module('external_baselines.hgtnet.aid_formal_protocol'),importlib.import_module('external_baselines.hgtnet.aid_formal_checkpoints')
    def binding(self):
        p,_=self.modules();r=copy.deepcopy(p.RECIPE);r.update(max_epochs=2,batch_size=2,eval_batch_size=2)
        return dict(schema='hgtnet-aid-formal-binding-v1',input_scope='cpu_fixture',recipe=r,
            recipe_sha256=p.recipe_sha256(r),class_names=['c%02d'%i for i in range(30)],
            counts=dict(train=2,val=2,pool=4,test=2),records={k:[dict(path=k+str(i),label=i) for i in range(2)] for k in ['val','test']},
            manifest_sha256='a'*64,config_sha256='b'*64,weights_sha256='c'*64,source_identity={'fixture.py':'d'*64})
    def test_recipe_and_binding_reject_unfrozen_real_values(self):
        p,_=self.modules();b=self.binding();self.assertEqual(p.validate_binding(b),b)
        wrong=copy.deepcopy(b);wrong['input_scope']='real_aid'
        with self.assertRaises(ValueError):p.validate_binding(wrong)
        wrong=copy.deepcopy(b);wrong['recipe']['lr']=1e-4
        with self.assertRaises(ValueError):p.validate_binding(wrong)
    def test_model_real_tensor_roundtrip_and_bad_sha_binding_rejected(self):
        _,c=self.modules();b=self.binding();m=nn.Linear(3,30);original=c.state_sha256(m)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d).resolve()/'model.pt';r=c.save_model(m,path,binding=b,metadata={'epoch':1})
            fresh=nn.Linear(3,30);c.load_model(fresh,path,expected_sha256=r['sha256'],binding=b)
            self.assertEqual(c.state_sha256(fresh),original)
            with self.assertRaises(ValueError):c.load_model(fresh,path,expected_sha256='0'*64,binding=b)
            wrong=copy.deepcopy(b);wrong['source_identity']['fixture.py']='e'*64
            with self.assertRaises(ValueError):c.load_model(fresh,path,expected_sha256=r['sha256'],binding=wrong)
    def test_two_slots_preserve_previous_when_newest_bytes_corrupt(self):
        _,c=self.modules();b=self.binding();m=nn.Linear(3,30);opt=torch.optim.Adam(m.parameters(),lr=1e-5,weight_decay=5e-5)
        loss=m(torch.randn(2,3)).sum();loss.backward();opt.step()
        with tempfile.TemporaryDirectory() as d:
            output=Path(d).resolve()
            for generation in [1,2]:
                if generation==2:
                    opt.zero_grad();m(torch.randn(2,3)).sum().backward();opt.step()
                payload=dict(schema='hgtnet-aid-formal-snapshot-v1',binding=b,generation=generation,phase='calibration',epoch=generation,
                    model_state=m.state_dict(),optimizer_state=opt.state_dict(),calibration_history=[])
                receipt=c.save_snapshot(output,payload)
            restored,record=c.load_snapshot(output,b)
            self.assertEqual(restored['generation'],2);self.assertTrue(restored['optimizer_state']['state'])
            Path(receipt['path']).write_bytes(b'broken')
            restored,record=c.load_snapshot(output,b)
            self.assertEqual(restored['generation'],1)
    def test_incompatible_state_or_nan_never_copied(self):
        _,c=self.modules();m=nn.Linear(3,30);before=c.state_sha256(m)
        state=dict(m.state_dict());state['weight']=state['weight'].clone();state['weight'][0,0]=float('nan')
        with self.assertRaises(ValueError):c.apply_model_state(m,state)
        self.assertEqual(c.state_sha256(m),before)

    def test_recipe_boolean_flags_cannot_be_replaced_by_equal_integer(self):
        p,_=self.modules();b=self.binding();b['recipe']['amp']=0
        b['recipe_sha256']=p.recipe_sha256(b['recipe'])
        with self.assertRaises(ValueError):p.validate_binding(b)

    def test_snapshot_rejects_out_of_budget_epoch_and_optimizer_policy(self):
        _,c=self.modules();b=self.binding();m=nn.Linear(3,30);o=torch.optim.Adam(m.parameters(),lr=1e-3)
        m(torch.randn(2,3)).sum().backward();o.step()
        with tempfile.TemporaryDirectory() as d:
            payload=dict(schema='hgtnet-aid-formal-snapshot-v1',binding=b,generation=1,phase='calibration',epoch=1,
                model_state=m.state_dict(),optimizer_state=o.state_dict())
            with self.assertRaises(ValueError):c.save_snapshot(Path(d).resolve(),payload)
            o.param_groups[0].update(lr=1e-5,weight_decay=5e-5);payload.update(optimizer_state=o.state_dict(),epoch=999,generation=999)
            with self.assertRaises(ValueError):c.save_snapshot(Path(d).resolve(),payload)

    def test_restored_adam_moment_shape_must_match_parameter(self):
        _,c=self.modules();b=self.binding();m=nn.Linear(3,30);o=torch.optim.Adam(m.parameters(),lr=1e-5,weight_decay=5e-5)
        m(torch.randn(2,3)).sum().backward();o.step()
        c.validate_optimizer(o,b['recipe'])
        o.state[m.weight]['exp_avg']=torch.zeros(1)
        with self.assertRaises(ValueError):c.validate_optimizer(o,b['recipe'])

if __name__=='__main__':unittest.main()
