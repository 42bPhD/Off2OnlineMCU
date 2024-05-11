from torch import nn
import torch
import copy
def download_url(url, model_dir='.', overwrite=False):
    import os, sys
    from urllib.request import urlretrieve
    target_dir = url.split('/')[-1]
    model_dir = os.path.expanduser(model_dir)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        # remove lock file so download can be executed next time.
        os.remove(os.path.join(model_dir, 'download.lock'))
        sys.stderr.write('Failed to download from url %s' % url + '\n' + str(e) + '\n')
        return None
    

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


def fold_bn(model):
    def fuse_conv_bn(conv:nn.Conv2d, bn:nn.BatchNorm2d):
        # modified from https://mmcv.readthedocs.io/en/latest/_modules/mmcv/cnn/utils/fuse_conv_bn.html
        assert conv.bias is None

        factor = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
        conv.weight.data = conv.weight.data * factor.reshape(-1, 1, 1, 1)
        conv.bias = nn.Parameter(- bn.running_mean.data * factor + bn.bias.data)

        return conv
    model_fused = copy.deepcopy(model)
    fused_backbone = []
    ptr = 0    
    while ptr < len(model_fused.backbone):
        if isinstance(model_fused.backbone[ptr], nn.Conv2d) and \
            isinstance(model_fused.backbone[ptr + 1], nn.BatchNorm2d):
            fused_backbone.append(fuse_conv_bn(
                model_fused.backbone[ptr], model_fused.backbone[ptr+ 1]))
            ptr += 2
        else:
            fused_backbone.append(model_fused.backbone[ptr])
            ptr += 1
    model_fused.backbone = nn.Sequential(*fused_backbone)
    print('After conv-bn fusion: backbone length', len(model_fused.backbone))
    # sanity check, no BN anymore
    for m in model_fused.modules():
        assert not isinstance(m, nn.BatchNorm2d)
    return model_fused
    

def get_model_size(model: nn.Module, data_width=32):
    """
    calculate the model size in bits
    :param data_width: #bits per element
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width
import torchvision
from torchvision import transforms
def get_cifar10_loader(batch_size=512,
                       image_size=32):
    print('=> loading cifar10 data...')
    # normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    train_dataset = torchvision.datasets.CIFAR10(
            root='E:/2_Quantization/torch2cmsis/examples/cifar/data/data_cifar10',
            train=True,
            download=True,
            transform=transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
            ]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(
            root='E:/2_Quantization/torch2cmsis/examples/cifar/data/data_cifar10',
            train=False,
            download=True,
            transform=transforms.Compose([
            transforms.ToTensor(),
            # normalize,
            ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader