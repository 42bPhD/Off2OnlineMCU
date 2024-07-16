
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

def CCNT(kwargs=None, delay_output=False):
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
    tmp.append('#include "symmetric/test_data.h"')
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
        elif kwargs['OPTIMIZING']['WEIGHT_INTERLEAVING'] == 'EVEN':
            tmp.append('    arm_convolve_HWC_q7_fast_Im2Col(img_buffer, IM_DIM_H, IM_IN_CH, conv_wt_nhwc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,')
        else:
            raise ValueError('Invalid weight interleaving')
        tmp.append('                            CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res1, CONV_OUT_DIM_H,')
        tmp.append('                            (q15_t *) col_buffer, NULL);')
        tmp.append('    unsigned long end = micros();')
        tmp.append('    printf("CMSIS Im2Col Conv Time = %d\\n", end-start);')
    tmp.append('    printf("---------TFLM Symmetric Im2Col Convolution--------------\\n");')
    tmp.append('    start = micros();')
    tmp.append('    arm_im2col_convolve_s8_opt(img_buffer, TFLM_CONV_W, TFLM_CONV_H, TFLM_IN_CH, TFLM_INPUT_BATCHES,')
    tmp.append('                                TFLM_weights, TFLM_OUT_CH, TFLM_FILTER_X, TFLM_FILTER_Y, TFLM_PAD_X, TFLM_PAD_Y,')
    tmp.append('                                TFLM_STRIDE_X, TFLM_STRIDE_Y, TFLM_biases, (q7_t *)TFLM_output_ref, TFLM_output_shift,')
    tmp.append('                                TFLM_output_mult, TFLM_OUTPUT_OFFSET, TFLM_INPUT_OFFSET, TFLM_OUT_ACTIVATION_MIN,')
    tmp.append('                                TFLM_OUT_ACTIVATION_MAX, TFLM_OUT_CONV_W, TFLM_OUT_CONV_H, (q15_t *) col_buffer);')
    tmp.append('    end = micros();')
    tmp.append('    printf("TFLM Im2Col Conv Time = %d\\n", end-start);')
    # tmp.append('    delay(3000);')
    tmp.append('}')
    
    return tmp