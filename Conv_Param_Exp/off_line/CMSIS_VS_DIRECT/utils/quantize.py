"""
#### Reference ###
### CMSIS NN quantization
https://community.arm.com/developer/ip-products/processors/b/processors-ip-blog/posts/deploying-convolutional-neural-network-on-cortex-m-with-cmsis-nn
"""
# Weight Qx.y * Activation Qx.y + Bias Qx.y -> Output Qx.y
import numpy as np
import torch
from fxpmath import Fxp
def FracBits(weight:np.ndarray, bits:int=8)->int:
    if isinstance(weight, torch.Tensor):
        weight = np.array(weight)
    elif isinstance(weight, Fxp):
        weight = weight.val
    
    #find number of integer bits to represent this range
    int_bits = int(np.ceil(np.log2(np.amax(weight))))
    frac_bits = bits- 1 - int_bits #remaining bits are fractional bits (1-bit for sign)
    return frac_bits

def Quantize_torch(weight:np.ndarray, frac_bits:int)->np.ndarray:
    if isinstance(weight, torch.Tensor):
        weight = np.array(weight)
    elif isinstance(weight, Fxp):
        weight = weight.val
    #floating point weights are scaled and rounded to [-128,127], which are used in 
    #the fixed-point operations on the actual hardware (i.e., microcontroller)
    # torch.clamp(torch.ceil(torch.round(weight*(2**frac_bits))), min=-128, max=127).type(torch.int8)
    quant_int8 = np.ceil(np.round(weight*(2**frac_bits)))
    np.clip(quant_int8, -128, 127, out=quant_int8)
    return quant_int8

def weight_quantization(weight:np.ndarray, bits:int=8)->tuple[np.ndarray, int]:
    shift_bits = FracBits(weight, bits=bits)
    weight_q = Quantize_torch(weight, shift_bits) #Q-format: Q7.0
    return weight_q, shift_bits

if __name__ == '__main__':
    a = np.random.rand(3, 2)
    res = weight_quantization(a)
    print(res)