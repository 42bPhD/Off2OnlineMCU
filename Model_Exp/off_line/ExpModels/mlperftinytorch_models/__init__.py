from .ds_cnn import *
from .fc_ae import *
from .mbv2 import *
from .resnet import *

import os
this_dir, this_filename = os.path.split(__file__)
pickle_path = os.path.join(this_dir,'pickles')

def get_ks_model():

    ks_model = DS_CNN()
    ks_path = os.path.join(pickle_path,'ks_state')
    ks_model.load_state_dict(torch.load(ks_path))
    return ks_model

def get_ic_model():
    ic_model = MLPerfTiny_ResNet_Baseline(num_classes=10)
    ic_path = os.path.join(pickle_path,'ic_state')
    ic_model.load_state_dict(torch.load(ic_path))
    return ic_model

def get_vww_model():
    vww_model = KL_MBV2_forVWW()
    vww_path = os.path.join(pickle_path,'vww_state')
    vww_model.load_state_dict(torch.load(vww_path))
    return vww_model

def get_ad_model():
    ad_model = FC_AE()
    ad_path = os.path.join(pickle_path,'ad_state')
    ad_model.load_state_dict(torch.load(ad_path))
    return ad_model
