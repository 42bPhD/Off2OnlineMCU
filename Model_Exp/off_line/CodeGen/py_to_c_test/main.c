
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
    printf("start execution\n");

    memset(conv_out_nhwc, 0, CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W);
    memset(conv_out_hwnc, 0, CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W);
    memset(conv_buf, 0, CONV_OUT_CH);
    
    q7_t     *res1 = conv_out_nhwc;
    q7_t     *res2 = conv_out_hwnc;
    q7_t     *img_buffer = input_data;
    
    printf("---------------Naive Direct Convolution------------------\n");
    unsigned long start = micros();
    arm_convolve_HWC_q7_RGB_Direct(img_buffer, IM_DIM_H, IM_IN_CH, conv_wt_nhwc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,
                            CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res1, CONV_OUT_DIM_H,
                             (q15_t *) col_buffer, NULL);
    unsigned long end = micros();
    
    printf("Naive Conv Time = %d\n", end-start);
    printf("---------Custom Dot-product Convolution--------------\n");
    start = micros();
    convolve_HWC_q7_RGB_Direct_HWNC(img_buffer, IM_DIM_H, IM_IN_CH, conv_wt_hwnc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,
                                CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res2, CONV_OUT_DIM_H,
                                (q31_t *) conv_buf, NULL);
    end = micros();
    printf("Time = %d\n", end-start);
    if (validate_s8(res1, res2, CONV_OUT_CH * CONV_OUT_DIM_H * CONV_OUT_DIM_W)){
        printf("Custom Conv is correct\n");
    }else{
        printf("Custom Conv is wrong\n");
    }
    delay(3000);
}
