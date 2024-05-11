import copy
import math
import random
from collections import OrderedDict, defaultdict


import numpy as np
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torch.functional import F
from torchvision.datasets import *
from torchvision.transforms import *
from fxpmath import Fxp

from utils.models import VGG
from utils.metrics import evaluate, train
from utils.etc import get_model_size, get_cifar10_loader, reconstruction_model
from utils.etc import fold_bn, get_cifar10_loader

from utils.qmodules import QuantizedConv2d, QuantizedLinear, QuantizedMaxPool2d, QuantizedAvgPool2d, QuantizedAdaptiveAvgPool2d
from utils.qmodules import get_quantization_scale_and_zero_point, linear_quantize_weight_per_channel, linear_quantize_bias_per_output_channel
from utils.qmodules import shift_quantized_conv2d_bias, shift_quantized_linear_bias
from utils.qmodules import peek_linear_quantization, plot_weight_distribution

from utils.qfmt_modules import ( QConv2d, QLinear, qfmt_quanize)

if __name__ =='__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    Byte = 8
    KiB = 1024 * Byte
    MiB = 1024 * KiB
    GiB = 1024 * MiB
    
    checkpoint_url = "E:\\pytorch_mcu\\tutorials\\vgg.cifar.pretrained.pth"
    checkpoint = torch.load(checkpoint_url, map_location="cpu")
    model = VGG().cuda()
    
    print(f"=> loading checkpoint '{checkpoint_url}'")
    model.load_state_dict(checkpoint['state_dict'])
    recover_model = lambda : model.load_state_dict(checkpoint['state_dict'])
    

    trainloader, testloader = get_cifar10_loader(batch_size=32)
    
    fp32_model_accuracy = evaluate(model, testloader)
    fp32_model_size = get_model_size(model)
    print(f"fp32 model has accuracy={fp32_model_accuracy:.2f}%")
    print(f"fp32 model has size={fp32_model_size/MiB:.2f} MiB")
    
        
    recover_model()
    # plot_weight_distribution(model)
    # peek_linear_quantization()
    
    #  fuse the batchnorm into conv layers
    recover_model()
    model_fused = copy.deepcopy(model)
    model_fused = fold_bn(model_fused)
    fused_acc = evaluate(model_fused, testloader)
    print(f'Accuracy of the fused model={fused_acc:.2f}%')

    # Make a Sequential model
    # model_fused = reconstruction_model(model_fused)
    print(model_fused)
            
    # add hook to record the min max value of the activation
    from collections import Counter, defaultdict
    input_activation = defaultdict(Counter)
    output_activation = defaultdict(Counter)

    def add_range_recoder_hook(model):
        import functools
        
        def _record_range(self, x, y, module_name):
            x = x[0]
            # _, f = qfmt_quanize(x.detach())
            # try:
            #     input_activation[module_name][f.item()] +=1
            # except:
            #     input_activation[module_name][f] += 1
            input_activation[module_name] = x.detach()  
            output_activation[module_name] = y.detach()
            # _, f = qfmt_quanize(y.detach())
            # try:
            #     output_activation[module_name][f.item()] +=1
            # except:
            #     output_activation[module_name][f] += 1

        all_hooks = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU)):
                all_hooks.append(m.register_forward_hook(
                    functools.partial(_record_range, module_name=name)))
        return all_hooks
    
        
    hooks = add_range_recoder_hook(model_fused)
    # sample_data = iter(trainloader).__next__()[0]
    # test_sample_data = iter(testloader).__next__()[0]
    # exit()
    
    for idx, (imgs, labels) in tqdm(enumerate(trainloader)):
        with torch.no_grad():
            imgs = imgs.cuda()
            model_fused(imgs)
            if idx > 100:
                break
    # for k, v in input_activation.items():
    #     input_activation[k] = v.most_common(1)[0][0]
    # for k, v in output_activation.items():
    #     output_activation[k] = v.most_common(1)[0][0]
    # print("Input Activation", input_activation)
    # print("Output Activation", output_activation)
    
    # remove hooks
    for h in hooks:
        h.remove()

    named_modules = [(name, module) for name, module in model_fused.named_modules()][1:]
    named_modules
    
    # we use int8 quantization, which is quite popular
    feature_bitwidth = weight_bitwidth = 8 
    
    
    quantized_model = copy.deepcopy(model_fused)
    quantized_backbone = []
    ptr = 0
    
    quantized_model.cpu()
    while ptr < len(quantized_model.backbone):
        # print(ptr, '/', len(quantized_model.backbone))
        # Fusing Conv2d and ReLU
        if isinstance(quantized_model.backbone[ptr], nn.Conv2d) and \
            isinstance(quantized_model.backbone[ptr + 1], nn.ReLU):
            conv = quantized_model.backbone[ptr]
            conv_name = f'backbone.{ptr}'            
            relu = quantized_model.backbone[ptr + 1]
            relu_name = f'backbone.{ptr + 1}'
            
            q_weight, w_f = qfmt_quanize(conv.weight.data)
            
            q_weight = torch.quantize_per_tensor(conv.weight.data, scale=1/(2**w_f), zero_point=0, dtype=torch.qint8).int_repr()
            
            q_weight = q_weight.to(torch.int8)
            
            q_bias, b_f = qfmt_quanize(conv.bias.data)
            q_bias = torch.quantize_per_tensor(conv.bias.data, scale=1/(2**b_f), zero_point=0, dtype=torch.qint8).int_repr()
            q_bias = q_bias.to(torch.int8)
            
            _, i_f = qfmt_quanize(input_activation[conv_name])
            _, o_f = qfmt_quanize(output_activation[relu_name])
            
            quantized_conv = QConv2d(
                q_weight, q_bias,
                w_f, b_f,
                i_f, o_f,
                conv.stride, conv.padding, conv.dilation, conv.groups,
                feature_bitwidth=feature_bitwidth, 
                weight_bitwidth=weight_bitwidth
            )
            quantized_backbone.append(quantized_conv)
            ptr += 2
        elif isinstance(quantized_model.backbone[ptr], nn.MaxPool2d):
            quantized_backbone.append(QuantizedMaxPool2d(
                kernel_size=quantized_model.backbone[ptr].kernel_size,
                stride=quantized_model.backbone[ptr].stride
                ))
            ptr += 1
        elif isinstance(quantized_model.backbone[ptr], nn.AvgPool2d):
            quantized_backbone.append(QuantizedAvgPool2d(
                kernel_size=quantized_model.backbone[ptr].kernel_size,
                stride=quantized_model.backbone[ptr].stride
                ))
            ptr += 1
        elif isinstance(quantized_model.backbone[ptr], nn.AdaptiveAvgPool2d):
            quantized_backbone.append(QuantizedAdaptiveAvgPool2d(
                output_size=quantized_model.backbone[ptr].output_size
                ))
            ptr += 1
        elif isinstance(quantized_model.backbone[ptr], nn.Flatten):
            quantized_backbone.append(nn.Flatten())
            ptr+=1
            continue
        
        else:
            ptr+=1
            raise NotImplementedError(type(quantized_model.backbone[ptr]))  # should not happen
    quantized_model = nn.Sequential(*quantized_backbone)

    # finally, quantized the classifier
    fc_name = 'classifier'
    fc = model.classifier
    
    # weight_bitwidth
    
    weight_fxp, w_f = qfmt_quanize(fc.weight.data)
    weight_fxp = torch.quantize_per_tensor(fc.weight.data, scale=1/(2**w_f), zero_point=0, dtype=torch.qint8).int_repr()
    weight_fxp = weight_fxp.to(torch.int8)
    # bias_bitwidth
    bias_fxp, b_f = qfmt_quanize(fc.bias.data)
    bias_fxp = torch.quantize_per_tensor(fc.bias.data, scale=1/(2**b_f), zero_point=0, dtype=torch.qint8).int_repr()
    bias_fxp = bias_fxp.to(torch.int8)
    
    _, i_f = qfmt_quanize(input_activation[fc_name])
    _, o_f = qfmt_quanize(output_activation[fc_name])
    
    quantized_model.classifier = QLinear(
        weight=weight_fxp,
        bias=bias_fxp,
        weight_n_frac=w_f,
        bias_n_frac=b_f,
        input_n_frac=i_f,
        output_n_frac=o_f
    )
    
    # print(quantized_model)
    quantized_model = quantized_model.cuda()
    def extra_preprocess(x):
        # hint: you need to convert the original fp32 input of range (0, 1)
        #  into int8 format of range (-128, 127)
        # x = qfmt_quanize(x)
        # return torch.tensor(x.get_val(), dtype=torch.float32)
        x, _ = qfmt_quanize(x)
        return x
        # return (x * 255 - 128).clamp(-128, 127).to(torch.int8)


    int8_model_accuracy = evaluate(model=quantized_model, 
                                   dataloader= testloader,
                                extra_preprocess=[extra_preprocess])
    print(f"int8 model has accuracy={int8_model_accuracy:.2f}%")
    