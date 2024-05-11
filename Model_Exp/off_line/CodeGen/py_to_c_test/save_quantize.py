""" https://developer.arm.com/documentation/102591/2208
Converting a Neural Network for Arm Cortex-M with CMSIS-NN Guide

Bias Shift 계산:
Bias shift는 레이어의 계산 결과에 bias 값을 더할 때 필요한 쉬프트 값을 의미합니다. 주어진 식에 따라 bias shift는 다음과 같이 계산됩니다:
Bias shift = (fi + fw) - fb
여기서,
    fi: 입력의 소수점 비트 수
    fw: 가중치의 소수점 비트 수
    fb: bias의 소수점 비트 수

Out Shift 계산:
        Out shift는 레이어의 계산 결과를 최종 출력 포맷으로 조정할 때 필요한 쉬프트 값을 의미합니다. 주어진 식에 따라 out shift는 다음과 같이 계산됩니다:
Out shift = (fi + fw) - fo
여기서,
    fi: 입력의 소수점 비트 수
    fw: 가중치의 소수점 비트 수
    fo: 출력의 소수점 비트 수

즉, bias shift와 out shift를 계산하기 위해서는 입력, 가중치 및 출력의 소수점 비트 수를 알아야 합니다. 
이 값을 사용하여 주어진 식을 이용해 쉬프트 값을 계산하면 됩니다.
"""
from quantize import weight_quantization


import torch
import numpy as np
from torch import nn
from torch.functional import F
from collections import defaultdict
import os
def ddict():
    return defaultdict(ddict)
class CodeGenerator():
    def __init__(self, config:dict, root='./'):
        # #include "parameter.h"
        self.header = config['header'] # specific variable
        # #define CONV_WT {wt};
        self.body = config['body'] # weight array param
        self.foot = config['foot'] # declare variable code
        
        self.root = root
        
    def header_gen(self, filename='parameter.h'):
        with open(os.path.join(self.root, filename), 'w') as F:
            F.write("#ifndef _TEST_PARAMETER_H\n")
            F.write("#define _TEST_PARAMETER_H\n\n")
            for k, v in self.header.items():
                F.write(f"#define {k} {v}\n")
            F.write("\n#endif\n")

    def body_gen(self, filename=None):
        with open(os.path.join(self.root, filename), 'w') as F:
            F.write("#ifndef _TEST_INPUTS_H\n")
            F.write("#define _TEST_INPUTS_H\n\n")
            F.write("#include <stdint.h>\n")
            F.write('#include "parameter.h"\n')
            F.write('#include "weights.h"\n\n')
            F.write('#define DTCM __attribute__((section(".DTCM")))\n')
            for k, v in self.body.items():
                F.write(f"{v}\n")
            F.write("\n#endif\n")
            
    def foot_gen(self, filename):
        with open(os.path.join(self.root, filename), 'w') as F:
            F.write("#ifndef _TEST_WEIGHTS_H\n")
            F.write("#define _TEST_WEIGHTS_H\n\n")
            for k, v in self.foot.items():
                F.write(f"#define {k} {v}\n")
            F.write("\n#endif\n")

    def main_gen(self, filename):
        with open(os.path.join(self.root, filename), 'w') as F:
            F.writelines("""
#include <Arduino.h>
#include "arm_nnsupportfunctions.h"
#include "arm_math.h"

#include <arm_nnfunctions.h>
#include <stdlib.h>
#include "inputs.h"
#include "parameter.h"
#include "weights.h"
#include "modules.h"
#define ARM_MATH_LOOPUNROLL
#define ARM_MATH_DSP

void setup() {
    Serial.begin(115200);
    SCB_EnableICache();
    SCB_EnableDCache();
}

void loop() {
    printf("start execution\\n");

    memset(conv_out_nhwc, 0, CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W);
    memset(conv_out_hwnc, 0, CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W);
    memset(conv_buf, 0, CONV_OUT_CH);
    
    q7_t     *res1 = conv_out_nhwc;
    q7_t     *res2 = conv_out_hwnc;
    q7_t     *img_buffer = input_data;
    
    printf("---------------Naive Direct Convolution------------------\\n");
    unsigned long start = micros();
    arm_convolve_HWC_q7_RGB_Direct(img_buffer, IM_DIM_H, IM_IN_CH, conv_wt_nhwc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,
                            CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res1, CONV_OUT_DIM_H,
                             (q15_t *) col_buffer, NULL);
    unsigned long end = micros();
    
    printf("Naive Conv Time = %d\\n", end-start);
    printf("---------Custom Dot-product Convolution--------------\\n");
    start = micros();
    convolve_HWC_q7_RGB_Direct_HWNC(img_buffer, IM_DIM_H, IM_IN_CH, conv_wt_hwnc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,
                                CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res2, CONV_OUT_DIM_H,
                                (q31_t *) conv_buf, NULL);
    end = micros();
    printf("Time = %d\\n", end-start);
    if (validate_s8(res1, res2, CONV_OUT_CH * CONV_OUT_DIM_H * CONV_OUT_DIM_W)){
        printf("Custom Conv is correct\\n");
    }else{
        printf("Custom Conv is wrong\\n");
    }
    delay(3000);
}
"""
)
    def codegen(self, 
                path:str= './',
                header= True, 
                body= True,
                foot= True):
        self.root = path
        self.header_gen(filename = 'parameter.h.')
        # self.body_gen(filename = 'inputs.h')
        self.foot_gen(filename = 'weights.h')
        # self.main_gen(filename= 'main.c')
    

def list_bracket(lists):
    return "{" + ', '.join(map(str, lists)) + "}"

def reproducibility(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=np.inf) #extend numpy
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

#  
from enum import Enum, auto

class WT_Format(Enum):
    USE_NHWC = auto()
    USE_RGB = auto()
    USE_EVEN_CH_V1 = auto()
    USE_EVEN_CH_V2 = auto()
    
if __name__ == '__main__':
    config = ddict()
    DEPLOY_WT_FORMAT = WT_Format.USE_RGB
    config['header']['IM_IN_CH'] = 16
    config['header']['IM_DIM_H'] = 32
    config['header']['IM_DIM_W'] = 32
    
    config['header']['CONV_IM_CH'] = config['header']['IM_IN_CH']
    config['header']['CONV_OUT_CH'] = 32
    config['header']['CONV_KER_DIM_H'] = 3
    config['header']['CONV_KER_DIM_W'] = 3
    config['header']['CONV_STRIDE_H'] = 1
    config['header']['CONV_STRIDE_W'] = 1
    config['header']['CONV_PADDING_H'] = 1
    config['header']['CONV_PADDING_W'] = 1
    
    
    reproducibility(SEED=1234)
    img = torch.randn(size = (1, config['header']['IM_IN_CH'], 
                                    config['header']['IM_DIM_H'], 
                                    config['header']['IM_DIM_W']))
    
    conv2d = nn.Conv2d(in_channels= config['header']['CONV_IM_CH'], 
                        out_channels= config['header']['CONV_OUT_CH'], 
                        kernel_size=(config['header']['CONV_KER_DIM_H'], 
                                    config['header']['CONV_KER_DIM_W']),
                        stride=(config['header']['CONV_STRIDE_H'], 
                                config['header']['CONV_STRIDE_W']),
                        padding=(config['header']['CONV_PADDING_H'], 
                                config['header']['CONV_PADDING_W']),
                        bias=True, 
                        padding_mode='zeros')
    bias = conv2d.bias
    weight = conv2d.weight
    
    # Weight, Bias and Input quantization
    # If fi is the number of fractional bits for the input, fo for the output, fw for the weight and fb for the biases then:
    # The bias shift is : (fi + fw) - fb
    # The out shift is : (fi + fw) - fo
    q_Im_in, fi = weight_quantization(img)
    q_weight, fw = weight_quantization(weight)
    q_bias, fb = weight_quantization(bias)
    config['header']['CONV_WEIGHT_BIT'] = fw
    config['header']['CONV_BIAS_BIT'] = fb
    config['header']['INPUT_RSHIFT'] = fi
    
    Im_out_ref = F.conv2d(img, weight, bias,
                        stride= (config['header']['CONV_STRIDE_H'], 
                                config['header']['CONV_STRIDE_W']), 
                        padding= (config['header']['CONV_PADDING_H'],
                                config['header']['CONV_PADDING_W']))
    
    q_out, fo = weight_quantization(Im_out_ref)
    OUT_RSHIFT = (fi+fw) - fo
    BIAS_LSHIFT = (fi+fw) - fb
    config['header']['CONV_OUT_RSHIFT'] = OUT_RSHIFT
    config['header']['CONV_BIAS_LSHIFT'] = BIAS_LSHIFT
    
    q_bias = q_bias.detach().numpy().astype(np.int8).ravel().tolist()
    q_bias_ = list_bracket(q_bias)
    config['foot']['CONV_BIAS'] = q_bias_
    config['body']['CONV_BIAS'] = 'static q7_t conv_bias[CONV_OUT_CH] = CONV_BIAS;'
    
    q_Im_in = q_Im_in.detach().numpy()
    q_Im_in = np.transpose(q_Im_in, (0, 2, 3, 1)).astype(np.int8)
    # INPUT_MEAN = q_Im_in.mean(axis=(0,1,2)).astype(np.uint8).ravel().tolist()
    # INPUT_MEAN = list_bracket(INPUT_MEAN)
    
    q_Im_in = q_Im_in.ravel().tolist() #NCHW -> NHWC
    q_im_in = list_bracket(q_Im_in)
    config['foot']['IM_IN'] = q_im_in
    config['body']['IM_IN'] = 'static q7_t input_data[IM_IN_CH * IM_DIM_H * IM_DIM_W] = IM_IN;'
    
    # WEIGHT * INPUT + BIAS = OUTPUT
    # q_out = q_out.detach().numpy()
    _, Im_out_c, Im_out_h, Im_out_w = q_out.shape
    # q_out = np.transpose(q_out, (0, 2, 3, 1)).astype(np.int8).ravel().tolist() #NCHW -> NHWC
    # q_out = list_bracket(q_out)
    
    q_weight = q_weight.detach().numpy()
    NHWC_weight = np.transpose(q_weight, (0, 2, 3, 1)).astype(np.int8) #NCHW -> NHWC
    print(NHWC_weight.shape)
    # print(NHWC_weight)
    HWNC_weight = np.transpose(NHWC_weight, (1, 2, 0, 3)) #NHWC(0, 1, 2, 3) -> HWNC(1, 2, 0, 3)
    
    print(HWNC_weight.shape)
    # Settings for codegen

    config['foot']['NHWC_WEIGHT'] = list_bracket(NHWC_weight.ravel().tolist())
    config['body']['NHWC_WEIGHT'] = 'static q7_t conv_wt_nhwc[CONV_IM_CH * CONV_KER_DIM_H * CONV_KER_DIM_W * CONV_OUT_CH] = NHWC_WEIGHT;'

    
    assert not DEPLOY_WT_FORMAT in [i.value for i in WT_Format], "Not supported weight format"
    if DEPLOY_WT_FORMAT == WT_Format.USE_RGB: # RGB -> RRR GGG BBB (ODD)
        h, w, n, c = HWNC_weight.shape # [0, 3, 2, 5, 6, 9, 8, 11, 1, 4, 7, 10] -> [0, 2, 3, 5, 6, 8, 9, 11, (1, 4, 7, 10)]
        indice = np.array([np.array([0, 3, 2, 5, 6, 9, 8, 11, 1, 4, 7, 10]) + 12*i for i in range(n*c//12)])
        indice = indice.ravel()
        HWNC_weight = HWNC_weight.reshape(h, w, n*c)
        HWNC_weight = HWNC_weight[:, :, indice]
    
    elif DEPLOY_WT_FORMAT == WT_Format.USE_EVEN_CH_V1: # EVEN Channels
        HWNC_weight = np.concatenate([i for i in np.split(HWNC_weight, 
                                                          config['header']['CONV_IM_CH']//4, 
                                                          axis=3)], 
                                     axis=2)
        
    elif DEPLOY_WT_FORMAT== WT_Format.USE_EVEN_CH_V2: # EVEN Channels
        HWNC_weight = HWNC_weight.reshape(config['header']['CONV_KER_DIM_H'], 
                                        config['header']['CONV_KER_DIM_W'], 
                                        config['header']['CONV_OUT_CH'] *(config['header']['CONV_IM_CH']//4), 
                                        4)
       
    else:
        raise NotImplementedError
    
    config['foot']['HWNC_WEIGHT'] = list_bracket(HWNC_weight.ravel().tolist())
    config['body']['HWNC_WEIGHT'] = 'static q7_t conv_wt_hwnc[CONV_IM_CH * CONV_KER_DIM_H * CONV_KER_DIM_W * CONV_OUT_CH] = HWNC_WEIGHT;'
    
    config['body']['COL_BUFFER'] = 'static q7_t col_buffer[2 * CONV_KER_DIM_H * CONV_KER_DIM_W * CONV_OUT_CH * 2];'
    config['header']['CONV_OUT_DIM_H'] = Im_out_h
    config['header']['CONV_OUT_DIM_W'] = Im_out_w
    
    config['body']['CONV_OUT1'] = 'static q7_t conv_out_nhwc[CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W];'
    config['body']['CONV_OUT2'] = 'static DTCM q7_t conv_out_hwnc[CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W];'
    # for proposed direct conv only
    config['body']['CONV_BUF'] = 'static DTCM q31_t conv_buf[CONV_OUT_CH];'
    config['body']['CONV_OUT_BUF2'] = 'static DTCM q15_t conv_out_buf[CONV_IM_CH];'
    
    ####
    codegen = CodeGenerator(config=config)
    codegen.codegen(path='E:\\DirectConvolution_MCU\\lib\\Modules')