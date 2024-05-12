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
void reset_cnt()
{
    CoreDebug->DEMCR |= 0x01000000;
    DWT->CYCCNT = 0; // reset the counter
    DWT->CTRL = 0;
}
void start_cnt()
{
    DWT->CTRL |= 0x00000001; // enable the counter
}
void stop_cnt()
{
    DWT->CTRL &= 0xFFFFFFFE; // disable the counter
}
unsigned int getCycles()
{
    return DWT->CYCCNT;
}
#if !defined DWT_LSR_Present_Msk
#define DWT_LSR_Present_Msk ITM_LSR_Present_Msk
#endif
#if !defined DWT_LSR_Access_Msk
#define DWT_LSR_Access_Msk ITM_LSR_Access_Msk
#endif
#define DWT_LAR_KEY 0xC5ACCE55
void dwt_access_enable(unsigned ena)
{
    uint32_t lsr = DWT->LSR;
    if ((lsr & DWT_LSR_Present_Msk) != 0)
    {
        if (ena)
        {
            if ((lsr & DWT_LSR_Access_Msk) != 0) // locked: access need unlock
            {
                DWT->LAR = DWT_LAR_KEY;
            }
        }
        else
        {
            if ((lsr & DWT_LSR_Access_Msk) == 0) // unlocked
            {
                DWT->LAR = 0;
            }
        }
    }
}
void setup() {
    Serial.begin(115200);
    SCB_EnableICache();
    SCB_EnableDCache();
}
void loop() {
    printf("start execution\n");
    memset(conv_out_nhwc, 0, CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W);
    q7_t     *res1 = conv_out_nhwc;
    q7_t     *img_buffer = input_data;
    printf("---------------ARM CMSIS Im2Col Convolution------------------\n");
    unsigned long start = micros();
    arm_convolve_HWC_q7_RGB_Im2Col(img_buffer, IM_DIM_H, IM_IN_CH, conv_wt_nhwc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,
                            CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res1, CONV_OUT_DIM_H,
                            (q15_t *) col_buffer, NULL);
    unsigned long end = micros();
    printf("CMSIS Im2Col Conv Time = %d\n", end-start);
    printf("---------Direct Convolution--------------\n");
    memset(conv_buf, 0, CONV_OUT_CH);
    memset(conv_out_hwnc, 0, CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W);
    q7_t     *res2 = conv_out_hwnc;
    start = micros();
    convolve_HWC_q7_RGB_Direct_HWNC_SIMD_fast(img_buffer, IM_DIM_H, IM_IN_CH, conv_wt_hwnc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,
                                CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res2, CONV_OUT_DIM_H,
                                (q31_t *) conv_buf, NULL);
    end = micros();
    printf("Direct Conv Time = %d\n", end-start);
    if (validate_s8(res1, res2, CONV_OUT_CH * CONV_OUT_DIM_H * CONV_OUT_DIM_W)){
        printf("Custom Conv is correct\n");
    }else{
        printf("Custom Conv is wrong\n");
    }
    delay(3000);
}