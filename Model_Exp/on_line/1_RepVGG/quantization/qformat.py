from torch import nn
from abc import abstractmethod

import copy
import torch

from torch.nn import functional as F
from torch.autograd import Variable

def fold_CBR_qfmt(conv, bn, relu):
    print(conv, bn, relu)
    folded = QConvBNReLU(conv, bn, relu, 8)
    conv = folded
    bn = nn.Identity()
    relu = nn.Identity()
    return conv, bn, relu
def fold_CB_qfmt(conv, bn):
    print(conv, bn)
    folded = QConvBNReLU(conv, bn, nn.Identity(), 8)
    conv = folded
    bn = nn.Identity()
    
    return conv, bn

class QParam():#nn.Module
    def __init__(self,num_bits=8):
        super().__init__()

        self.num_bits = num_bits
        self.frac_bit = torch.tensor([], requires_grad=False, dtype=torch.int8)
        self.min = torch.tensor([], requires_grad=False, dtype=torch.float32)
        self.max = torch.tensor([], requires_grad=False, dtype=torch.float32)
        
        # 양자화 매개변수를 저장하기 위해 Register_buffer를 사용
        # 기울기가 생성되지 않으며 parameter로 사용되지는 않지만 state_dict로 조회가 가능하다.
        # 이러한 방식으로 모델 가중치와 양자화 매개변수를 모델에 독립적으로 저장할 수 있다.
        # frac_bit = torch.tensor([], requires_grad=False, dtype=torch.int8)
        # min = torch.tensor([], requires_grad=False, dtype=torch.float32)
        # max = torch.tensor([], requires_grad=False, dtype=torch.float32)
        # self.register_buffer('frac_bit', frac_bit)
        # self.register_buffer('min', min)
        # self.register_buffer('max', max)
        
    def update(self, x:torch.Tensor):
        self.max = torch.max(x)
        self.min = torch.min(x)
        
        # int_bits = int(torch.ceil(torch.log2(torch.max(torch.abs(max_value), torch.abs(min_value)))).item())
        int_bits = torch.round(torch.log2(torch.max(torch.abs(self.max), torch.abs(self.min))))
        # int_bits = int_bits if int_bits > 0 else 0
        
        self.frac_bit =  self.num_bits - 1 - int_bits
    
    def quantize_tensor(self, weight:torch.Tensor):
        return torch.clamp((weight*(2**self.frac_bit)), 
                            -2**(self.num_bits-1), 
                            2**(self.num_bits-1)-1)
        
    def dequantize_tensor(self, weight):
        return weight / (2**self.frac_bit)


    # Load quantization parameters from 'state_dict'
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key_names = ['frac_bit', 'min', 'max']
        for key in key_names:
            value = getattr(self, key)
            value.data = state_dict[prefix + key].data
            state_dict.pop(prefix + key)

    def __str__(self):
        info = 'frac_bit(Q-frac):%d '  % self.frac_bit
        info += 'min:%d '  % self.min
        info += 'max:%d'  % self.max
        return info



class QModule(nn.Module):
    def __init__(self, input_param:QParam, weight_param:QParam, output_param:QParam):
        super().__init__()
        self.input_param = input_param
        self.weight_param = weight_param
        self.output_param = output_param
    
    def freeze(self):
        pass
    
    @abstractmethod
    def quantize_input(self, x:torch.Tensor):
        raise NotImplementedError("Quantized input is not implemented!")


class QConvBNReLU(nn.Module):

    def __init__(self, conv_module:nn.Conv2d, bn_module:nn.BatchNorm2d, act_module:nn.ReLU, num_bits=8):
        super(QConvBNReLU, self).__init__()
        self.conv_module = conv_module
        self.bn_module = bn_module
        self.act_module = act_module
        self.num_bits = num_bits
        self.input_param = QParam(num_bits)
        self.weight_param = QParam(num_bits)
        self.output_param = QParam(num_bits)
        self.bias_param = copy.deepcopy(self.weight_param) if conv_module.bias != None else None
        self.out_rshift = 0
        self.bias_lshift = 0
        

    def fold_bn(self, mean, std):
        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std
            weight = self.conv_module.weight * gamma_.view(self.conv_module.out_channels, 1, 1, 1)
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.conv_module.weight * gamma_
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
            
        return weight, bias


    def forward(self, x):
        self.input_param.update(x)
        x = FakeQuantize.apply(x, self.input_param)
        
        if self.training:
            y = F.conv2d(x, self.conv_module.weight, 
                            self.conv_module.bias, 
                            stride=self.conv_module.stride,
                            padding=self.conv_module.padding,
                            dilation=self.conv_module.dilation,
                            groups=self.conv_module.groups)
            y = y.permute(1, 0, 2, 3) # NCHW -> CNHW
            y = y.contiguous().view(self.conv_module.out_channels, -1) # CNHW -> C,NHW
            # mean = y.mean(1)
            # var = y.var(1)
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn_module.running_mean = \
                (1 - self.bn_module.momentum) * self.bn_module.running_mean + \
                self.bn_module.momentum * mean
            self.bn_module.running_var = \
                (1 - self.bn_module.momentum) * self.bn_module.running_var + \
                self.bn_module.momentum * var
        else:
            mean = Variable(self.bn_module.running_mean)
            var = Variable(self.bn_module.running_var)

        std = torch.sqrt(var + self.bn_module.eps)

        weight, bias = self.fold_bn(mean, std)

        self.weight_param.update(self.conv_module.weight.data)
        if self.bias_param:
            self.bias_param.update(self.conv_module.bias.data)
        
        x = F.conv2d(x, 
                     FakeQuantize.apply(weight, self.weight_param), 
                     FakeQuantize.apply(bias, self.bias_param), 
                stride=self.conv_module.stride,
                padding=self.conv_module.padding, dilation=self.conv_module.dilation, 
                groups=self.conv_module.groups)

        x = self.act_module(x)

        
        self.output_param.update(x)
        x = FakeQuantize.apply(x, self.output_param)

        return x

    def freeze(self):
        self.input_param.frac_bit
        self.weight_param.frac_bit
        self.bias_param.frac_bit
        self.output_param.frac_bit
        
        self.bias_shift = self.input_param.frac_bit + self.weight_param.frac_bit\
                                - self.bias_param.frac_bit if self.bias_param is not None else 0
        self.out_shift = self.input_param.frac_bit + self.weight_param.frac_bit\
                                - self.output_param.frac_bit
                                
        std = torch.sqrt(self.bn_module.running_var + self.bn_module.eps)

        weight, bias = self.fold_bn(self.bn_module.running_mean, std)
        self.conv_module.weight.data = self.weight_param.quantize_tensor(weight.data)
        if self.bias_param is not None:
            self.conv_module.bias.data = self.bias_param.quantize_tensor(bias.data)
        
    
    def quantize_inference(self, x):
        x = self.conv_module(x)
        if self.conv_module.bias is not None:
            bias = torch.bitwise_left_shift(self.conv_module.bias, self.bias_lshift)
            # if you want arm_nn_truncation, uncomment the following line
            # bias = bias + (torch.ones_like(bias, dtype=torch.int32) << (self.out_rshift-1))
        else:
            bias = None
        
        x = F.conv2d(x, 
                     weight=self.conv_module.weight, 
                     bias= bias, 
                     stride = self.stride,
                     padding = self.padding,
                     dilation = self.dilation,
                     groups = self.groups)
        
        x = torch.bitwise_right_shift(x, self.out_rshift)
        return torch.clamp(x, -2**(self.output_param.num_bits-1), 2**(self.output_param.num_bits-1)-1)
        
    
    def show_params(self):
        print('Input:', self.input_param)
        print('Weight:', self.weight_param)
        print('Output:', self.output_param)
        if self.bias_param is not None:
            print('Bias:', self.bias_param)

class FakeQuantize(torch.autograd.Function):
    # Fake 양자화 노드, 양자화 및 역양자화 수행
    # 양자화 전후의 오류를 시뮬레이션합니다. 이러한 부동 추론은 양자화 int 추론과 동일한 정확도를 갖습니다.

    # 역전파를 통해 기울기를 구하려면 STE를 사용하세요.이 부분은 PTQ에서 역전파되지 않으므로 당분간 무시해도 됩니다.
    # Function Class는 매개변수가 없는 모듈과 유사합니다. 상속을 위해서는 순방향 및 역방향를 다시 작성해야 합니다.
    @staticmethod
    def forward(ctx, x, qparam:QParam):
        """
        def forward(ctx,input,*args)
            ctx: 수동으로 전달할 필요가 없습니다. 다음과 같은 몇 가지 작업을 수행하여 기울기 계산을 용이하게 할 수 있습니다.
            ctx.save_for_backward(tensor)  순방향 전파 중에 일부 텐서를 저장합니다.
            tensor = ctx.saved_tensors    역전파 중에도 액세스할 수 있습니다(역방향 함수 내부).

            input:함수 입력
            *args:기타 선택적 매개변수
        """
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output:다음 레이어의 기울기

        QAT에서는 역전파(backpropagation)에 관한 것입니다.
        이 레이어(Fake 양자화 노드)의 경우 기울기가 계산되지 않고 다음 레이어의 기울기가 직접 전달됩니다.

        backward의 반환 값은 해당 입력의 기울기를 나타내는 전달 값과 일치해야 합니다.
        q_parm은 양자화 매개변수이므로 기울기를 계산할 필요가 없으므로 None이 반환됩니다.
        따라서 grad_output,None을 직접 반환합니다.
        """
        return grad_output, None
    



class QConv2d(nn.Conv2d):
    def __init__(self, 
                 in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.input_param = QParam(8)
        self.weight_param = QParam(8)
        self.bias_param = copy.deepcopy(self.weight_param) if bias else None
        self.output_param = QParam(8)
        self.out_rshift = 0
        self.bias_lshift = 0
        
    def forward(self, x):
        self.input_param.update(x)
        x = FakeQuantize.apply(x, self.input_param)
        
        self.weight_param.update(self.weight.data)
        if self.bias_param:
            self.bias_param.update(self.bias.data)
        x = F.conv2d(x, FakeQuantize.apply(self.weight, self.weight_param), 
                     self.bias, 
                     stride = self.stride,
                     padding = self.padding,
                     dilation = self.dilation,
                     groups = self.groups)
        
        self.output_param.update(x)
        x = FakeQuantize.apply(x, self.output_param)
        return x

    def freeze(self):
        self.input_param.frac_bit
        self.weight_param.frac_bit
        self.bias_param.frac_bit
        self.output_param.frac_bit
        
        self.bias_shift = self.input_param.frac_bit + self.weight_param.frac_bit\
                                - self.bias_param.frac_bit if self.bias_param is not None else 0
        self.out_shift = self.input_param.frac_bit + self.weight_param.frac_bit\
                                - self.output_param.frac_bit
        self.weight.data = self.weight_param.quantize_tensor(self.weight.data)
        if self.bias_param is not None:
            self.bias.data = self.bias_param.quantize_tensor(self.bias.data)
        # return self
    
    def quantize_inferene(self, x:torch.Tensor):
        x = self.input_param.quantize_tensor(x)
        if self.bias is not None:
            bias = torch.bitwise_left_shift(self.bias, self.bias_lshift)
            # if you want arm_nn_truncation, uncomment the following line
            # bias = bias + (torch.ones_like(bias, dtype=torch.int32) << (self.out_rshift-1))
        else:
            bias = None
        
        x = F.conv2d(x, 
                     weight=self.weight, 
                     bias= bias, 
                     stride = self.stride,
                     padding = self.padding,
                     dilation = self.dilation,
                     groups = self.groups)
        
        x = torch.bitwise_right_shift(x, self.out_rshift)
        return torch.clamp(x, -2**(self.output_param.num_bits-1), 2**(self.output_param.num_bits-1)-1)
    
    
    def show_params(self):
        print('Input:', self.input_param)
        print('Weight:', self.weight_param)
        print('Output:', self.output_param)
        if self.bias_param is not None:
            print('Bias:', self.bias_param)


    

class QLinear(nn.Linear):
    def __init__(self, 
                 in_features,
                 out_features,
                 bias=True):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.input_param = QParam(8)
        self.weight_param = QParam(8)
        self.bias_param = copy.deepcopy(self.weight_param) if bias else None
        self.output_param = QParam(8)
        self.out_rshift = 0
        self.bias_lshift = 0
        
    def forward(self, x):
        self.input_param.update(x)
        x = FakeQuantize.apply(x, self.input_param)
        
        self.weight_param.update(self.weight.data)
        if self.bias_param is not None:
            self.bias_param.update(self.bias.data)
        x = F.linear(x, 
                     FakeQuantize.apply(self.weight, self.weight_param),
                     self.bias)
        
        self.output_param.update(x)
        x = FakeQuantize.apply(x, self.output_param)
        return x

    def freeze(self):
        self.input_param.frac_bit
        self.weight_param.frac_bit
        self.bias_param.frac_bit
        self.output_param.frac_bit
        
        self.bias_shift = self.input_param.frac_bit + self.weight_param.frac_bit\
                                - self.bias_param.frac_bit if self.bias_param is not None else 0
        self.out_shift = self.input_param.frac_bit + self.weight_param.frac_bit\
                                - self.output_param.frac_bit
        self.weight.data = self.weight_param.quantize_tensor(self.weight.data)
        if self.bias_param is not None:
            self.bias.data = self.bias_param.quantize_tensor(self.bias.data)
        # return self
    
    def quantize_inferene(self, x:torch.Tensor):
        x = self.input_param.quantize_tensor(x)
        if self.bias is not None:
            bias = torch.bitwise_left_shift(self.bias, self.bias_lshift)
            # if you want arm_nn_truncation, uncomment the following line
            # bias = bias + (torch.ones_like(bias, dtype=torch.int32) << (self.out_rshift-1))
        else:
            bias = None
        
        x = F.linear(x, 
                     weight=self.weight, 
                     bias= bias)
        
        x = torch.bitwise_right_shift(x, self.out_rshift)
        return torch.clamp(x, -2**(self.output_param.num_bits-1), 2**(self.output_param.num_bits-1)-1)