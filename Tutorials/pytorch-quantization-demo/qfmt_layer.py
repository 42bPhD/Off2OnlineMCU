from torch import nn
from abc import abstractmethod

import copy
import torch

from torch.nn import functional as F
    


class QParam(nn.Module):
    def __init__(self,num_bits=8):
        super().__init__()

        self.num_bits = num_bits
        frac_bit = torch.tensor([], requires_grad=False, dtype=torch.int8)
        min = torch.tensor([], requires_grad=False, dtype=torch.float32)
        max = torch.tensor([], requires_grad=False, dtype=torch.float32)
        
        # 양자화 매개변수를 저장하기 위해 Register_buffer를 사용
        # 기울기가 생성되지 않으며 parameter로 사용되지는 않지만 state_dict로 조회가 가능하다.
        # 이러한 방식으로 모델 가중치와 양자화 매개변수를 모델에 독립적으로 저장할 수 있다.
        self.register_buffer('frac_bit', frac_bit)
        self.register_buffer('min', min)
        self.register_buffer('max', max)
        
    def update(self, x:torch.Tensor):
        self.max = torch.max(x)
        self.min = torch.min(x)
        
        # int_bits = int(torch.ceil(torch.log2(torch.max(torch.abs(max_value), torch.abs(min_value)))).item())
        int_bits = torch.round(torch.log2(torch.max(torch.abs(self.max), torch.abs(self.min))))
        # int_bits = int_bits if int_bits > 0 else 0
        
        self.frac_bit =  self.num_bits - 1 - int_bits
    
    def quantize_tensor(self, weight:torch.Tensor):
        
        return torch.clamp((weight*(2**self.frac_bit)), 
                            -2**(self.num_bits), 
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
    



class QConv2d(QModule):
    def __init__(self, 
                 conv2d:nn.Conv2d, 
                 input_param:QParam=QParam(8), 
                 weight_param:QParam=QParam(8), 
                 output_param:QParam=QParam(8)):
        super(QConv2d, self).__init__(input_param=input_param, weight_param=weight_param, output_param=output_param)
        self.conv2d = conv2d
        self.input_param = input_param
        self.weight_param = weight_param
        self.bias_param = copy.deepcopy(weight_param) if conv2d.bias is not None else None
        self.output_param = output_param
        self.out_rshift = 0
        self.bias_lshift = 0
        self.conv2d = torch.nn.utils.parametrizations.weight_norm(self.conv2d, name='weight', dim=0)
        
        # 가중치 정규화를 해제하려면 torch.nn.utils.parametrizations.remove_weight_norm 함수를 사용할 수 있습니다. 
        # 이 함수는 가중치 정규화가 적용된 레이어를 인자로 받아 원래 상태로 복원합니다.
        # torch.nn.utils.parametrizations.remove_weight_norm(self.conv2d)


        
    @property
    def in_channels(self):
        return self.conv2d.in_channels
    @property
    def out_channels(self):
        return self.conv2d.out_channels
    @property
    def kernel_size(self):
        return self.conv2d.kernel_size
    @property
    def stride(self):
        return self.conv2d.stride
    @property
    def padding(self):
        return self.conv2d.padding
    @property
    def dilation(self):
        return self.conv2d.dilation
    @property
    def groups(self):
        return self.conv2d.groups
    @property
    def bias(self):
        return self.conv2d.bias
    @property
    def weight(self):
        return self.conv2d.weight
    @weight.setter
    def weight(self, value):
        self.conv2d.weight = value
    @property
    def bias(self):
        return self.conv2d.bias
    @bias.setter
    def bias(self, value):
        self.conv2d.bias = value
    
    def forward(self, x):
        self.input_param.update(x)
        x = FakeQuantize.apply(x, self.input_param)
        # Weights normalization
        
        self.weight_param.update(self.conv2d.weight)
        if self.bias_param is not None:
            self.bias_param.update(self.conv2d.bias.data)
        x = F.conv2d(x, FakeQuantize.apply(self.conv2d.weight, self.weight_param), 
                     self.conv2d.bias, 
                     stride = self.conv2d.stride,
                     padding = self.conv2d.padding,
                     dilation = self.conv2d.dilation,
                     groups = self.conv2d.groups)
        
        self.output_param.update(x)
        x = FakeQuantize.apply(x, self.output_param)
        return x

    def freeze(self):
        self.input_param.frac_bit
        self.weight_param.frac_bit
        self.bias_param.frac_bit
        self.output_param.frac_bit
        
        self.bias_lshift = self.input_param.frac_bit + self.weight_param.frac_bit\
                                - self.bias_param.frac_bit if self.bias_param is not None else 0
        self.out_rshift = self.input_param.frac_bit + self.weight_param.frac_bit\
                                - self.output_param.frac_bit
        self.conv2d.weight.data = self.weight_param.quantize_tensor(self.conv2d.weight.data)
        if self.bias_param is not None:
            self.conv2d.bias.data = self.bias_param.quantize_tensor(self.conv2d.bias.data)
        # return self
    
    def quantize_inferene(self, x:torch.Tensor):
        x = self.input_param.quantize_tensor(x)
        if self.conv2d.bias is not None:
            bias = torch.bitwise_left_shift(self.conv2d.bias, self.bias_lshift)
            # if you want arm_nn_truncation, uncomment the following line
            # bias = bias + (torch.ones_like(bias, dtype=torch.int32) << (self.out_rshift-1))
        else:
            bias = None
        
        x = F.conv2d(x, 
                     weight=self.conv2d.weight, 
                     bias= bias, 
                     stride = self.conv2d.stride,
                     padding = self.conv2d.padding,
                     dilation = self.conv2d.dilation,
                     groups = self.conv2d.groups)
        
        x = torch.bitwise_right_shift(x, self.out_rshift)
        return torch.clamp(x, -2**(self.output_param.num_bits), 2**(self.output_param.num_bits-1)-1)
    
    
    def show_params(self):
        print('Input:', self.input_param)
        print('Weight:', self.weight_param)
        print('Output:', self.output_param)
        if self.bias_param is not None:
            print('Bias:', self.bias_param)


class QLinear(QModule):
    def __init__(self, 
                 linear:nn.Linear, 
                 input_param:QParam=QParam(8), 
                 weight_param:QParam=QParam(8), 
                 output_param:QParam=QParam(8)):
        super(QLinear, self).__init__(input_param=input_param, weight_param=weight_param, output_param=output_param)
        self.linear = linear
        self.input_param = input_param
        self.weight_param = weight_param
        self.bias_param = copy.deepcopy(weight_param) if linear.bias is not None else None
        self.output_param = output_param
        self.out_rshift = 0
        self.bias_lshift = 0
        
    def forward(self, x):
        self.input_param.update(x)
        x = FakeQuantize.apply(x, self.input_param)
        
        self.weight_param.update(self.linear.weight.data)
        if self.bias_param is not None:
            self.bias_param.update(self.linear.bias.data)
        x = F.linear(x, 
                     FakeQuantize.apply(self.linear.weight, self.weight_param),
                     self.linear.bias)
        
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
        self.linear.weight.data = self.weight_param.quantize_tensor(self.linear.weight.data)
        if self.bias_param is not None:
            self.linear.bias.data = self.bias_param.quantize_tensor(self.linear.bias.data)
        # return self
    
    def quantize_inferene(self, x:torch.Tensor):
        x = self.input_param.quantize_tensor(x)
        if self.linear.bias is not None:
            bias = torch.bitwise_left_shift(self.linear.bias, self.bias_lshift)
            # if you want arm_nn_truncation, uncomment the following line
            # bias = bias + (torch.ones_like(bias, dtype=torch.int32) << (self.out_rshift-1))
        else:
            bias = None
        
        x = F.linear(x, 
                     weight=self.linear.weight, 
                     bias= bias)
        
        x = torch.bitwise_right_shift(x, self.out_rshift)
        return torch.clamp(x, -2**(self.output_param.num_bits), 2**(self.output_param.num_bits-1)-1)