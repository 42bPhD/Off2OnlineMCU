

import torch
from torch import nn
from torchprofile import profile_macs
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import torchvision.transforms as T
from collections import OrderedDict

def get_children(model: nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


from collections import OrderedDict
def nested_children(m: torch.nn.Module, parent_prefix: str = '') -> OrderedDict:
    children = m.named_children()
    output = OrderedDict()
    
    for name, child in children:
        # 현재 모듈의 이름이 포함된 전체 경로를 생성합니다.
        if parent_prefix:
            key_name = f"{parent_prefix}_{name}"
        else:
            key_name = name

        # 하위 모듈이 없는 경우, 즉 child가 마지막 자식 모듈인 경우
        if not list(child.named_children()):
            # 현재 모듈을 저장합니다.
            output[key_name] = child
        else:
            # 재귀적으로 하위 모듈을 탐색하고, 반환된 딕셔너리를 현재 딕셔너리에 병합합니다.
            output.update(nested_children(child, key_name))

    return output

def reconstruction_model(model, device = 'cuda:0'):
    # flatten_list = get_children(model)
    flatten_dict = nested_children(model)
    conv_idx = 1
    fcl_idx = 1
    pool_idx = 1
    bn_idx = 1
    act_idx = 1
    modules = OrderedDict()
    # for idx, module in enumerate(flatten_list, start=1):
    for idx, (types, module) in enumerate(flatten_dict.items(), start=1):
        name = module.__class__.__name__.upper()
        type_name = types.upper()
        if isinstance(module, nn.Conv2d):
            modules[f'{type_name}'] = module
            conv_idx +=1
        elif isinstance(module, nn.Linear):
            modules[f'{type_name}'] = module
            fcl_idx += 1
        elif isinstance(module, nn.MaxPool2d):
            modules[f'{type_name}'] = module
            pool_idx += 1
        elif isinstance(module, nn.BatchNorm2d):
            modules[f'{type_name}'] = module
            bn_idx +=1
        elif isinstance(module, nn.ReLU):
            modules[f'{type_name}'] = module
            act_idx +=1
        else:
            modules[f'{type_name}{idx}'] = module
    
    return nn.Sequential(modules).to(device)