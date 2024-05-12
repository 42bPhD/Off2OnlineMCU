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
def setup(kwargs):
    tmp:list = []
    tmp.append('\nvoid setup() {')
    tmp.append('    Serial.begin(115200);')
    if kwargs['OPTIMIZING']['CACHE']:
        tmp.append('    SCB_EnableICache();')
        tmp.append('    SCB_EnableDCache();')
    else:
        tmp.append('    SCB_DisableDCache();')
        tmp.append('    SCB_DisableICache();')
    tmp.append('}')
    return tmp

def CCNT(kwargs=None):
    tmp:list = []
    tmp.append('\nvoid reset_cnt()')
    tmp.append('{')
    tmp.append('    CoreDebug->DEMCR |= 0x01000000;')
    tmp.append('    DWT->CYCCNT = 0; // reset the counter')
    tmp.append('    DWT->CTRL = 0;')
    tmp.append('}')
    tmp.append('void start_cnt()')
    tmp.append('{')
    tmp.append('    DWT->CTRL |= 0x00000001; // enable the counter')
    tmp.append('}')
    tmp.append('void stop_cnt()')
    tmp.append('{')
    tmp.append('    DWT->CTRL &= 0xFFFFFFFE; // disable the counter')
    tmp.append('}')
    tmp.append('unsigned int getCycles()')
    tmp.append('{')
    tmp.append('    return DWT->CYCCNT;')
    tmp.append('}')
    tmp.append('#if !defined DWT_LSR_Present_Msk')
    tmp.append('#define DWT_LSR_Present_Msk ITM_LSR_Present_Msk')
    tmp.append('#endif')
    tmp.append('#if !defined DWT_LSR_Access_Msk')
    tmp.append('#define DWT_LSR_Access_Msk ITM_LSR_Access_Msk')
    tmp.append('#endif')
    tmp.append('#define DWT_LAR_KEY 0xC5ACCE55')
    tmp.append('void dwt_access_enable(unsigned ena)')
    tmp.append('{')
    tmp.append('    uint32_t lsr = DWT->LSR;')
    tmp.append('    if ((lsr & DWT_LSR_Present_Msk) != 0)')
    tmp.append('    {')
    tmp.append('        if (ena)')
    tmp.append('        {')
    tmp.append('            if ((lsr & DWT_LSR_Access_Msk) != 0) // locked: access need unlock')
    tmp.append('            {')
    tmp.append('                DWT->LAR = DWT_LAR_KEY;')
    tmp.append('            }')
    tmp.append('        }')
    tmp.append('        else')
    tmp.append('        {')
    tmp.append('            if ((lsr & DWT_LSR_Access_Msk) == 0) // unlocked')
    tmp.append('            {')
    tmp.append('                DWT->LAR = 0;')
    tmp.append('            }')
    tmp.append('        }')
    tmp.append('    }')
    tmp.append('}')
    return tmp

def include(kwargs=None):
    tmp:list=[]
    tmp.append('#include <Arduino.h>')
    tmp.append('#include "arm_nnsupportfunctions.h"')
    tmp.append('#include "arm_math.h"')
    tmp.append('#include <arm_nnfunctions.h>')
    tmp.append('#include <stdlib.h>')
    tmp.append('#include "inputs.h"')
    tmp.append('#include "parameter.h"')
    tmp.append('#include "weights.h"')
    tmp.append('#include "modules.h"')
    tmp.append('#define ARM_MATH_LOOPUNROLL')
    tmp.append('#define ARM_MATH_DSP')
    return tmp

def gen_code(kwargs):
    tmp:list = []
    ####################################################################
    tmp.append('\nvoid loop() {')
    if kwargs['OPTIMIZING']['IM2COL']:
        tmp.append('    printf("start execution\\n");')
        tmp.append('    memset(conv_out_nhwc, 0, CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W);')
        tmp.append('    q7_t     *res1 = conv_out_nhwc;')
        tmp.append('    q7_t     *img_buffer = input_data;')
        tmp.append('    printf("---------------ARM CMSIS Im2Col Convolution------------------\\n");')
        tmp.append('    unsigned long start = micros();')
        if kwargs['OPTIMIZING']['WEIGHT_INTERLEAVING'] == 'RGB':
            tmp.append('    arm_convolve_HWC_q7_RGB_Im2Col(img_buffer, IM_DIM_H, IM_IN_CH, conv_wt_nhwc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,')
        else:
            tmp.append('    arm_convolve_HWC_q7_fast_Im2Col(img_buffer, IM_DIM_H, IM_IN_CH, conv_wt_nhwc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,')
        tmp.append('                            CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res1, CONV_OUT_DIM_H,')
        tmp.append('                            (q15_t *) col_buffer, NULL);')
        tmp.append('    unsigned long end = micros();')
        tmp.append('    printf("CMSIS Im2Col Conv Time = %d\\n", end-start);')
    ######################################################
    
    tmp.append('    printf("---------Direct Convolution--------------\\n");')
    tmp.append('    memset(conv_buf, 0, CONV_OUT_CH);')
    tmp.append('    memset(conv_out_hwnc, 0, CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W);')
    tmp.append('    q7_t     *res2 = conv_out_hwnc;')
    tmp.append('    start = micros();')
    if kwargs['OPTIMIZING']['WEIGHT_INTERLEAVING'] == 'RGB':
        tmp.append('    convolve_HWC_q7_RGB_Direct_HWNC_SIMD_fast(img_buffer, IM_DIM_H, IM_IN_CH, conv_wt_hwnc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,')
    else:
        tmp.append('    convolve_HWC_q7_Ch4_Direct_HWNC_SIMD_Optim_ch(img_buffer, IM_DIM_H, IM_IN_CH, conv_wt_hwnc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,')
    tmp.append('                                CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res2, CONV_OUT_DIM_H,')
    tmp.append('                                (q31_t *) conv_buf, NULL);')
    tmp.append('    end = micros();')
    tmp.append('    printf("Direct Conv Time = %d\\n", end-start);')
        
    tmp.append('    if (validate_s8(res1, res2, CONV_OUT_CH * CONV_OUT_DIM_H * CONV_OUT_DIM_W)){')
    tmp.append('        printf("Custom Conv is correct\\n");')
    tmp.append('    }else{')
    tmp.append('        printf("Custom Conv is wrong\\n");')
    tmp.append('    }')
    
    
    tmp.append('    delay(3000);')
    tmp.append('}')
    
    return tmp