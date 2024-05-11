"""
Pytorch 모델을 Batch Norm Fold 및 Sequantial 형태로 다시 Wrapping해서 
Python 파일로 만드는 방법을 설명합니다.

1. RepVGG 모델을 생성합니다.
2. RepVGG 모델을 Load합니다.
3. RepVGG 모델을 Deploy 모드로 변환합니다.
4. Deploy 모델을 저장합니다.
5. Deploy 모델을 다시 Wrapping합니다.
6. Wrapping된 모델을 Python 파일로 저장합니다.



"""
import torch
from repvgg import repvgg_model_convert, create_RepVGG_C0, create_RepVGG_C1, create_RepVGG_LeNet
from torchsummary import summary
from torchprofile import profile_macs
import os
import argparse
from collections import OrderedDict
from torch import nn
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
        elif isinstance(module, (nn.ReLU, nn.ReLU6)):
            modules[f'{type_name}'] = module
            act_idx +=1
        else:
            modules[f'{type_name}{idx}'] = module
    
    return nn.Sequential(modules).to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RepVGG(plus) Conversion')
    parser.add_argument('--load', 
                        # metavar='LOAD', 
                        default='./trained_models/RepVGG_LeNet/qformat_cifar10_relu_notste/best_ckpt.pth', 
                        help='path to the weights file')
    parser.add_argument('--save', 
                        # metavar='SAVE', 
                        default="./trained_models/RepVGG_LeNet/qformat_cifar10_relu_notste/", 
                        help='path to the weights file')
    parser.add_argument('-a', 
                        '--arch',
                        metavar='ARCH', 
                        default='RepVGG-LeNet')
    args = parser.parse_args()
    
        
    # train_model = create_RepVGG_C0(deploy=False, num_classes=2)
    train_model = create_RepVGG_LeNet(deploy=False, num_classes=10)
    # train_model.load_state_dict(torch.load('./trained_models/RepVGG-C0/default/best_ckpt.pth')['model'])          # or train from scratch
    # do whatever you want with train_model
    # deploy_model = repvgg_model_convert(train_model, save_path='./trained_models/RepVGG-C0/default/deploy.pth')
    # do whatever you want with deploy_model
    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        print(ckpt.keys())
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    if 'plus' in args.arch:
        train_model.switch_repvggplus_to_deploy()
        torch.save(train_model.state_dict(), args.save)
    else:
        deploy_model = repvgg_model_convert(train_model, save_path=os.path.join(args.save, "best_ckpt_deploy.pth"))
        print("Done!")
        
    # summary(train_model, (3, 96, 96), device='cpu')
    summary(deploy_model, (3, 32, 32), device='cpu')
    # for name, param in deploy_model.named_modules():
        # if isinstance(param, torch.nn.Conv2d):
        #     print(name, param.kernel_size, param.stride, param.padding)
    flatten_model = reconstruction_model(deploy_model).cpu()
    
    model_order, forward_order = [], []
    for idx, (name, param) in enumerate(flatten_model.named_modules()):
        if idx == 0:
            continue
        if isinstance(param, nn.Identity):
            continue
        model_order.append(f'self.{name} = nn.{param}\n        ')
        forward_order.append(f'x = self.{name}(x)\n        ')
    # a = flatten_model(torch.randn(1, 3, 96, 96))
    # summary(flatten_model.cpu(), (3, 96, 96), device='cpu')   
    model_code = f"""
import torch
from torch import nn
class MCU_VGGRep_LeNet(nn.Module):
    def __init__(self, quant=True):
        super(MCU_VGGRep_LeNet, self).__init__()
        if quant:
            self.quant = torch.quantization.QuantStub()	# 입력을 양자화 하는 QuantStub()
            self.dequant = torch.quantization.DeQuantStub()	# 출력을 역양자화 하는 DeQuantStub()
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()
        {''.join(model_order)}
    def forward(self, x):
        x = self.quant(x)
        {''.join(forward_order)}
        x = self.dequant(x)
        return x
    
    def get_shape(self, batch_size, input_shape):
        x = torch.randn(size=(batch_size, input_shape))
        {''.join(forward_order[:-3])}
        return x.shape[1:]
"""
    with open(f'{os.path.join(args.save, "vggrep_lenet.py")}', 'w') as f:
        f.write(model_code)
    torch.save(flatten_model.state_dict(), os.path.join(args.save, "best_ckpt_deploy.pth"))