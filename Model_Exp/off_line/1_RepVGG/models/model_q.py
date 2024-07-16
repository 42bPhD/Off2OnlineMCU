
import torch
from torch import nn


class MCU_VGGRep(nn.Module):
    def __init__(self, num_classes=2):
        super(MCU_VGGRep, self).__init__()
        self.num_classes = num_classes
        self.STAGE0_RBR_REPARAM = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.STAGE0_NONLINEARITY = nn.ReLU()
        self.STAGE1_0_RBR_REPARAM = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.STAGE1_0_NONLINEARITY = nn.ReLU()
        self.STAGE1_1_RBR_REPARAM = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.STAGE1_1_NONLINEARITY = nn.ReLU()
        self.STAGE2_0_RBR_REPARAM = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.STAGE2_0_NONLINEARITY = nn.ReLU()
        self.STAGE2_1_RBR_REPARAM = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.STAGE2_1_NONLINEARITY = nn.ReLU()
        self.STAGE3_0_RBR_REPARAM = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.STAGE3_0_NONLINEARITY = nn.ReLU()
        self.STAGE3_1_RBR_REPARAM = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.STAGE3_1_NONLINEARITY = nn.ReLU()
        self.STAGE4_0_RBR_REPARAM = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.STAGE4_0_NONLINEARITY = nn.ReLU()
        self.GAP25 = nn.AdaptiveAvgPool2d(output_size=1)
        self.FLATTEN26 = nn.Flatten(start_dim=1, end_dim=-1)
        self.LINEAR = nn.Linear(in_features=256, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x = self.STAGE0_RBR_REPARAM(x)
        x = self.STAGE0_NONLINEARITY(x)
        x = self.STAGE1_0_RBR_REPARAM(x)
        x = self.STAGE1_0_NONLINEARITY(x)
        x = self.STAGE1_1_RBR_REPARAM(x)
        x = self.STAGE1_1_NONLINEARITY(x)
        x = self.STAGE2_0_RBR_REPARAM(x)
        x = self.STAGE2_0_NONLINEARITY(x)
        x = self.STAGE2_1_RBR_REPARAM(x)
        x = self.STAGE2_1_NONLINEARITY(x)
        x = self.STAGE3_0_RBR_REPARAM(x)
        x = self.STAGE3_0_NONLINEARITY(x)
        x = self.STAGE3_1_RBR_REPARAM(x)
        x = self.STAGE3_1_NONLINEARITY(x)
        x = self.STAGE4_0_RBR_REPARAM(x)
        x = self.STAGE4_0_NONLINEARITY(x)
        x = self.GAP25(x)
        x = self.FLATTEN26(x)
        x = self.LINEAR(x)
        
        return x
    
    def get_shape(self, batch_size, input_shape):
        x = torch.randn(size=(batch_size, *input_shape))
        x = self.STAGE0_RBR_REPARAM(x)
        x = self.STAGE0_NONLINEARITY(x)
        x = self.STAGE1_0_RBR_REPARAM(x)
        x = self.STAGE1_0_NONLINEARITY(x)
        x = self.STAGE1_1_RBR_REPARAM(x)
        x = self.STAGE1_1_NONLINEARITY(x)
        x = self.STAGE2_0_RBR_REPARAM(x)
        x = self.STAGE2_0_NONLINEARITY(x)
        x = self.STAGE2_1_RBR_REPARAM(x)
        x = self.STAGE2_1_NONLINEARITY(x)
        x = self.STAGE3_0_RBR_REPARAM(x)
        x = self.STAGE3_0_NONLINEARITY(x)
        x = self.STAGE3_1_RBR_REPARAM(x)
        x = self.STAGE3_1_NONLINEARITY(x)
        x = self.STAGE4_0_RBR_REPARAM(x)
        x = self.STAGE4_0_NONLINEARITY(x)
        x = self.GAP25(x)
        return x.shape[1:]


from torch.quantization import QuantStub, DeQuantStub
from torch.nn import functional as F


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
    return x_float, frac_bits


class MCU_VGGRepC1(nn.Module):
    def __init__(self, num_classes=2, quant=True):
        super(MCU_VGGRepC1, self).__init__()
        self.num_classes = num_classes
        self.quantize = quant
        if self.quantize:
            self.quant = QuantStub()
        
        self.STAGE0_CONV = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.STAGE0_RELU = nn.ReLU()
        self.STAGE1_0_CONV = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.STAGE1_0_RELU = nn.ReLU()
        self.STAGE2_0_CONV = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.STAGE2_0_RELU = nn.ReLU()
        self.STAGE3_0_CONV = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.STAGE3_0_RELU = nn.ReLU()
        self.STAGE4_0_CONV = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.STAGE4_0_RELU = nn.ReLU()
        # self.GAP21 = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)
        self.GAP21 = nn.AdaptiveAvgPool2d(output_size=1)
        self.FLATTEN22 = nn.Flatten(start_dim=1, end_dim=-1)
        self.LINEAR = nn.Linear(in_features=128, out_features=self.num_classes, bias=True)
        if self.quantize:
            self.dequant = DeQuantStub()
    def quantize(self, x):
        x, frac_bits = qfmt_quanize(x)
        return x
    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        x = self.STAGE0_CONV(x)
        x = self.STAGE0_RELU(x)
        
        # x = F.relu(x)
        x = self.STAGE1_0_CONV(x)
        x = self.STAGE1_0_RELU(x)
        # x = F.relu(x)
        x = self.STAGE2_0_CONV(x)
        x = self.STAGE2_0_RELU(x)
        # x = F.relu(x)
        x = self.STAGE3_0_CONV(x)
        x = self.STAGE3_0_RELU(x)
        # x = F.relu(x)
        x = self.STAGE4_0_CONV(x)
        x = self.STAGE4_0_RELU(x)
        # x = F.relu(x)
        x = self.GAP21(x)
        x = self.FLATTEN22(x)
        x = self.LINEAR(x)
        if self.quantize:
            x = self.dequant(x)
        return x
    
    def get_shape(self, batch_size, input_shape):
        x = torch.randn(size=(batch_size, *input_shape))
        x = self.STAGE0_CONV(x)
        # x = self.STAGE0_RELU(x)
        
        x = self.STAGE1_0_CONV(x)
        # x = self.STAGE1_0_RELU(x)
        
        x = self.STAGE2_0_CONV(x)
        # x = self.STAGE2_0_RELU(x)
        
        x = self.STAGE3_0_CONV(x)
        # x = self.STAGE3_0_RELU(x)
        
        x = self.STAGE4_0_CONV(x)
        # x = self.STAGE4_0_RELU(x)
        
        x = self.GAP21(x)
        return x.shape[1:]


import torch
from torch import nn
class MCU_VGGRep_LeNet(nn.Module):
    def __init__(self):
        super(MCU_VGGRep_LeNet, self).__init__()
        self.quant = torch.quantization.QuantStub()	# 입력을 양자화 하는 QuantStub()
        self.dequant = torch.quantization.DeQuantStub()	# 출력을 역양자화 하는 DeQuantStub()
        
        self.STAGE0_RBR_REPARAM = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.STAGE0_NONLINEARITY = nn.ReLU()
        #1
        self.STAGE1_0_RBR_REPARAM = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.STAGE1_0_NONLINEARITY = nn.ReLU()
        #2 
        self.STAGE2_0_RBR_REPARAM = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.STAGE2_0_NONLINEARITY = nn.ReLU()
        # 3
        self.STAGE2_1_RBR_REPARAM = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.STAGE2_1_NONLINEARITY = nn.ReLU()
        # 4
        self.STAGE3_0_RBR_REPARAM = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.STAGE3_0_NONLINEARITY = nn.ReLU()
        
        self.STAGE4_0_RBR_REPARAM = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.STAGE4_0_NONLINEARITY = nn.ReLU()
        # self.GAP19 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.GAP19 = nn.AdaptiveAvgPool2d(output_size=1)
        self.FLATTEN20 = nn.Flatten(start_dim=1, end_dim=-1)
        self.LINEAR = nn.Linear(in_features=64, out_features=10, bias=True)
        
    def forward(self, x):
        x = self.quant(x)
        x = self.STAGE0_RBR_REPARAM(x)
        x = self.STAGE0_NONLINEARITY(x)
        x = self.STAGE1_0_RBR_REPARAM(x)
        x = self.STAGE1_0_NONLINEARITY(x)
        x = self.STAGE2_0_RBR_REPARAM(x)
        x = self.STAGE2_0_NONLINEARITY(x)
        x = self.STAGE2_1_RBR_REPARAM(x)
        x = self.STAGE2_1_NONLINEARITY(x)
        x = self.STAGE3_0_RBR_REPARAM(x)
        x = self.STAGE3_0_NONLINEARITY(x)
        x = self.STAGE4_0_RBR_REPARAM(x)
        x = self.STAGE4_0_NONLINEARITY(x)
        x = self.GAP19(x)
        x = self.FLATTEN20(x)
        x = self.LINEAR(x)
        x = self.dequant(x)
        return x
    
    def get_shape(self, batch_size, input_shape):
        x = torch.randn(size=(batch_size, input_shape))
        x = self.STAGE0_RBR_REPARAM(x)
        x = self.STAGE0_NONLINEARITY(x)
        x = self.STAGE1_0_RBR_REPARAM(x)
        x = self.STAGE1_0_NONLINEARITY(x)
        x = self.STAGE2_0_RBR_REPARAM(x)
        x = self.STAGE2_0_NONLINEARITY(x)
        x = self.STAGE2_1_RBR_REPARAM(x)
        x = self.STAGE2_1_NONLINEARITY(x)
        x = self.STAGE3_0_RBR_REPARAM(x)
        x = self.STAGE3_0_NONLINEARITY(x)
        x = self.STAGE4_0_RBR_REPARAM(x)
        x = self.STAGE4_0_NONLINEARITY(x)
        
        return x.shape[1:]
