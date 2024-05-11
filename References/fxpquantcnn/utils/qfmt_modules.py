import torch
from torch import nn
from typing import Tuple, List, Union
from fxpmath import Fxp
import numpy as np
from torch.nn import functional as F
def get_quantized_range(bitwidth):
    quantized_max = (1 << (bitwidth - 1)) - 1
    quantized_min = -(1 << (bitwidth - 1))
    return quantized_min, quantized_max

def qfmt_quanize(x, n_bits=8, signed=True):
    range_min, range_max = torch.min(x), torch.max(x)
    range_abs = torch.max(torch.abs(range_min), torch.abs(range_max))
    int_bits = torch.ceil(torch.log2(range_abs)).type(torch.int8)
    frac_bits = n_bits - int_bits
    if signed:
        range_int_min = -(2 ** n_bits)
        range_int_max = (2 ** n_bits) - 1
        
        # frac_bits = 7 if frac_bits >= 8 else frac_bits - 1
        frac_bits -= 1
    else:
        range_int_min = 0
        range_int_max = (2 ** n_bits)
    # Quantization the input
    
    x_int = torch.round(x * (2 ** (frac_bits))).to(torch.int8)
    x_float = torch.clamp(x_int * (1/(2 ** (frac_bits))), range_int_min, range_int_max)
    # quant_error = torch.mean((x - x_float) ** 2)
    frac_bits = frac_bits if isinstance(frac_bits, int) else frac_bits.item()
    return x_int, frac_bits
    
        
class QConv2d(nn.Module):
    def __init__(self, weight, bias, 
                 weight_n_frac, bias_n_frac, 
                 input_n_frac, output_n_frac,
                 stride, padding, dilation, groups,
                 feature_bitwidth=8, weight_bitwidth=8):
        super().__init__()
        # current version Pytorch does not support IntTensor as nn.Parameter
        self.register_buffer('weight',weight.clone().detach())
        self.weight_n_frac = weight_n_frac
        self.register_buffer('bias', bias.clone().detach())
        self.bias_n_frac = bias_n_frac
        self.input_n_frac = input_n_frac
        self.output_n_frac = output_n_frac

        self.stride = stride
        self.padding = (padding[1], padding[0])
        self.dilation = dilation
        self.groups = groups

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

    def forward(self, x):
        
        assert(isinstance(self.input_n_frac, int))
        assert(isinstance(self.output_n_frac, int))
        assert(isinstance(self.weight_n_frac, int))
        assert(isinstance(self.bias_n_frac, int))
        # Step 1: calculate integer-based 2d convolution (8-bit multiplication with 32-bit accumulation)
        
        # input = torch.nn.functional.pad(input, padding, 'constant', input_zero_point)
        output = torch.nn.functional.conv2d(x.float(), self.weight.float(), None, 
                                            self.stride, self.padding, self.dilation, self.groups)
        output = output.round().to(torch.int32)
        if self.bias is not None:
            self.bias = self.bias.view(1, -1, 1, 1).to(torch.int32)
            # self.bias = torch.bitwise_left_shift(self.bias, self.input_n_frac + self.weight_n_frac - self.bias_n_frac).to(torch.int32)
            self.bias = self.bias * (1 << (self.input_n_frac + self.weight_n_frac - self.bias_n_frac))
            output = output + self.bias
            
        # output = torch.bitwise_right_shift(output, self.input_n_frac + self.weight_n_frac - self.output_n_frac)
        output = output * (1 / (1 << (self.input_n_frac + self.weight_n_frac - self.output_n_frac)))
        output = output.clamp(*get_quantized_range(self.feature_bitwidth)).to(torch.int8)
        return output

class QLinear(nn.Module):
    def __init__(self, weight, bias, 
                 input_n_frac, output_n_frac,
                 weight_n_frac, bias_n_frac,
                 feature_bitwidth=8, weight_bitwidth=8):
        super().__init__()
        # current version Pytorch does not support IntTensor as nn.Parameter
        self.register_buffer('weight', weight.clone().detach())
        self.weight_n_frac = weight_n_frac

        self.register_buffer('bias', bias.clone().detach())
        self.bias_n_frac = bias_n_frac

        self.input_n_frac = input_n_frac
        self.output_n_frac = output_n_frac

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

    def forward(self, x):    
        assert(isinstance(self.input_n_frac, int))
        assert(isinstance(self.output_n_frac, int))
        assert(isinstance(self.bias_n_frac, int))
        assert(isinstance(self.weight_n_frac, int))
        if x.dim() != 2:
            x = torch.flatten(x, 1)
        output = torch.nn.functional.linear(x.float(), self.weight.float(), None).to(torch.int32)
        if self.bias is not None:
            self.bias = self.bias * (1 << (self.input_n_frac + self.weight_n_frac - self.bias_n_frac))
            # self.bias = torch.bitwise_left_shift(self.bias.to(torch.int32), self.input_n_frac + self.weight_n_frac - self.bias_n_frac).to(torch.int32)
            output = (output + self.bias).to(torch.int32)
        # output = torch.bitwise_right_shift(output, self.input_n_frac + self.weight_n_frac - self.output_n_frac).view(1, -1)
        output = output * (1 / (1 << (self.input_n_frac + self.weight_n_frac - self.output_n_frac)))
        output = output.round().clamp(*get_quantized_range(self.feature_bitwidth)).to(torch.int8)
        return output


class QMaxPool2d(nn.MaxPool2d):
    def forward(self, x):
        # current version PyTorch does not support integer-based MaxPool
        return super().forward(x.float()).to(torch.int8)

class QAvgPool2d(nn.AvgPool2d):
    def forward(self, x):
        # current version PyTorch does not support integer-based AvgPool
        return super().forward(x.float()).to(torch.int8)

class QAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, x):
        # current version PyTorch does not support integer-based AdaptiveAvgPool
        return super().forward(x.float()).to(torch.int8)