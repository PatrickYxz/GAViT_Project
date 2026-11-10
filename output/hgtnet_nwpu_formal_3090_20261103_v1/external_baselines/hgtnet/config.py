"""Strict architecture identity, distinct from a formal training recipe."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


FIXED={
    'schema':'hgtnet-independent-architecture-v0','implementation':'hgtnet_independent_v0',
    'official':False,'scope':'local_engineering_only','image_size':224,'in_channels':3,'patch_size':4,
    'stage_channels':[128,256,512],'stb_counts':[1,1,9],'heads':[4,8,16],
    'stage_output_channels':[256,512,512],
    'stage_downsamples':['patch_merge','patch_merge','conv2_same_channels'],
    'window_size':7,'shift_size':3,'mlp_ratio':4.0,'da_reduction':16,
    'graph_channels':[256,512],'graph_depths':[6,2],'graph_k':20,'graph_exclude_self':True,
    'graph_ffn_ratio':4,'graph_transition_strides':[1,1],'graph_transition_paddings':[1,0],
    'shape_policy':'independent_text_7_to_5','fusion':'gap_sum',
}
KEYS=set(FIXED)|{'dropout','attention_dropout','drop_path','norm_eps','graph_operator',
                 'num_classes','seed','pretrained_candidate','choices'}


def validate_config(value):
    if not isinstance(value,dict) or set(value)!=KEYS:
        raise ValueError('HGTNet config fields are missing or unknown')
    try:json.dumps(value,allow_nan=False)
    except (ValueError,TypeError) as exc:raise ValueError('config must be finite JSON data') from exc
    for name,wanted in FIXED.items():
        actual=value[name]
        if type(actual) is not type(wanted) or actual!=wanted:
            raise ValueError('unsupported independent architecture choice: '+name)
        if isinstance(wanted,list) and any(type(a) is not type(b) for a,b in zip(actual,wanted)):
            raise ValueError('invalid architecture list element type: '+name)
    if value['graph_operator'] not in ('mr_max','relative_mean'):
        raise ValueError('unknown graph operator')
    if type(value['num_classes']) is not int or value['num_classes'] not in (30,45):
        raise ValueError('num_classes must be30 or45')
    if type(value['seed']) is not int or not 0<=value['seed']<2**63:
        raise ValueError('seed must be a nonnegative integer below2**63')
    for key in ('dropout','attention_dropout','drop_path','norm_eps'):
        number=value[key]
        if type(number) not in (int,float) or not math.isfinite(number):raise ValueError('invalid '+key)
        if (key=='norm_eps' and number<=0) or (key!='norm_eps' and not 0<=number<1):raise ValueError('invalid '+key)
    candidate=value['pretrained_candidate']
    expected={'kind':'torchvision_swin_b_imagenet1k_v1',
              'url':'https://download.pytorch.org/models/swin_b-68c6b09e.pth',
              'sha256_prefix':'68c6b09e','full_file_identity':'must_be_supplied_explicitly_to_loader'}
    if candidate!=expected:raise ValueError('candidate pretrained source identity changed')
    if not isinstance(value['choices'],dict) or not value['choices'] or any(not isinstance(v,str) or not v for v in value['choices'].values()):
        raise ValueError('independent choices must be explicitly described')
    return deepcopy(value)


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def config_sha256(value):
    return hashlib.sha256(json.dumps(validate_config(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def architecture_shapes(value):
    validate_config(value)
    return {'patch':[128,56,56],'stage_outputs':[[256,28,28],[512,14,14],[512,7,7]],
            'graph_outputs':[[256,7,7],[512,5,5]]}
