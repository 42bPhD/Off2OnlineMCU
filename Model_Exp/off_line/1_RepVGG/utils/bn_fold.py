import torch
from torch import nn
from torchsummary import summary
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.train_eval import get_accuracy
import torch
import torch.nn as nn

def _fuse_conv_bn(conv: nn.Module, bn: nn.Module) -> nn.Module:
    """Fuse conv and bn into one module.

    Args:
        conv (nn.Module): Conv to be fused.
        bn (nn.Module): BN to be fused.

    Returns:
        nn.Module: Fused module.
    """
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w * factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv

def fuse_conv_bn(module: nn.Module) -> nn.Module:
    """Recursively fuse conv and bn in a module.

    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.

    Args:
        module (nn.Module): Module to be fused.

    Returns:
        nn.Module: Fused module.
    """
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child, (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module

def bn_fold(model, dataloader:DataLoader):
    
    def fuse_conv_bn(conv:nn.Conv2d, bn:nn.BatchNorm2d):
        # modified from https://mmcv.readthedocs.io/en/latest/_modules/mmcv/cnn/utils/fuse_conv_bn.html
        conv_w = conv.weight
        conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
            bn.running_mean)

        factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        conv.weight = nn.Parameter(conv_w * factor.reshape([conv.out_channels, 1, 1, 1]))
        conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)

        return conv

    print('Before conv-bn fusion: backbone length', len(model))
    #  fuse the batchnorm into conv layers
    from collections import OrderedDict
    model_fused:nn.Sequential = copy.deepcopy(model)
    fused_backbone = OrderedDict()
    ptr = 0
    
    while ptr < len(model_fused):
        name = list(model_fused._modules.items())[ptr][0]
        if isinstance(model_fused[ptr], nn.Conv2d) and isinstance(model_fused[ptr + 1], nn.BatchNorm2d):
            fused_backbone[name] = fuse_conv_bn(model_fused[ptr], model_fused[ptr+ 1])
            ptr += 2
        else:
            fused_backbone[name] = model_fused[ptr]
            ptr += 1
    model_fused = nn.Sequential(fused_backbone)

    print('After conv-bn fusion: backbone length', len(model_fused))
    # sanity check, no BN anymore
    for m in model_fused.modules():
        assert not isinstance(m, nn.BatchNorm2d)

    return model_fused