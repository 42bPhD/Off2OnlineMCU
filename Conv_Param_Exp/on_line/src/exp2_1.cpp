
// #include <Arduino.h>
// #include "arm_nnsupportfunctions.h"
// #include "arm_math.h"
// #include <arm_nnfunctions.h>
// #include <stdlib.h>
// #include "inputs.h"
// #include "parameter.h"
// #include "weights.h"
// #include "modules.h"
// #include "memory_pack.h"
// #include "test_data.h"
// void reset_cnt()
// {
//     CoreDebug->DEMCR |= 0x01000000;
//     DWT->CYCCNT = 0; // reset the counter
//     DWT->CTRL = 0;
// }

// void start_cnt()
// {
//     DWT->CTRL |= 0x00000001; // enable the counter
// }

// void stop_cnt()
// {
//     DWT->CTRL &= 0xFFFFFFFE; // disable the counter
// }

// unsigned int getCycles()
// {
//     return DWT->CYCCNT;
// }
// #if !defined DWT_LSR_Present_Msk
// #define DWT_LSR_Present_Msk ITM_LSR_Present_Msk
// #endif
// #if !defined DWT_LSR_Access_Msk
// #define DWT_LSR_Access_Msk ITM_LSR_Access_Msk
// #endif
// #define DWT_LAR_KEY 0xC5ACCE55
// void dwt_access_enable(unsigned ena)
// {
//     uint32_t lsr = DWT->LSR;

//     if ((lsr & DWT_LSR_Present_Msk) != 0)
//     {
//         if (ena)
//         {
//             if ((lsr & DWT_LSR_Access_Msk) != 0) // locked: access need unlock
//             {
//                 DWT->LAR = DWT_LAR_KEY;
//             }
//         }
//         else
//         {
//             if ((lsr & DWT_LSR_Access_Msk) == 0) // unlocked
//             {
//                 DWT->LAR = 0;
//             }
//         }
//     }
// }

// void setup() {
//     Serial.begin(115200);
//     SCB_EnableICache();
//     SCB_EnableDCache();
// }

// void loop() {
//     printf("start execution\n");
//     q7_t     *img_buffer = input_data;
    
//     memset(conv_out_nhwc, 0, CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W);
//     q7_t     *res1 = conv_out_nhwc;
//     memset(conv_out_hwnc, 0, CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W);
//     q7_t     *res2 = conv_out_hwnc;

    
//     //unsigned long start = micros();
//     //This is for RGB
//     bool res;
    
//     printf("IM_DIM_H=%d, IM_IN_CH=%d, CONV_OUT_CH=%d, KernelSize=%d\n", IM_DIM_H, IM_IN_CH, CONV_OUT_CH, CONV_KER_DIM_H);
//     reset_cnt();
//     start_cnt();
//     res = arm_convolve_HWC_q7_RGB(img_buffer, IM_DIM_H, IM_IN_CH, conv_wt_nhwc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,
//                         CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res1, CONV_OUT_DIM_H,
//                          (q15_t *) col_buffer, NULL);
    


//     stop_cnt();
//     printf("CMSIS Instruction Per Cycle = %d, Don't Care=%d\n", getCycles(), res);
    
//     reset_cnt();
//     start_cnt();
//     // res = convolve_HWC_q7_Direct_HWNC_naive(img_buffer, IM_DIM_H, IM_IN_CH, conv_wt_hwnc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,
//     //                         CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res2, CONV_OUT_DIM_H,
//     //                         (q31_t *) conv_buf, NULL);
//     res = arm_im2col_convolve_s8_opt(img_buffer, TFLM_CONV_W, TFLM_CONV_H, TFLM_IN_CH, TFLM_INPUT_BATCHES, 
//                             TFLM_weights, TFLM_OUT_CH, TFLM_FILTER_X, TFLM_FILTER_Y, TFLM_PAD_X, TFLM_PAD_Y, 
//                             TFLM_STRIDE_X, TFLM_STRIDE_Y, TFLM_biases, (q7_t *)TFLM_output_ref, TFLM_output_shift,
//                             TFLM_output_mult, TFLM_OUTPUT_OFFSET, TFLM_INPUT_OFFSET, TFLM_OUT_ACTIVATION_MIN, 
//                             TFLM_OUT_ACTIVATION_MAX, TFLM_CONV_W, TFLM_CONV_H, (q15_t *) col_buffer);
//     stop_cnt();
//     printf("Proposed Instruction Per Cycle = %d, Don't Care= %d\n", getCycles(), res);
//     printf("-------------------------------------------------------\n");
    
    
//     printf("W=%d, H=%d, C_in=%d C_out=%d, KS=%d, P=%d, S=%d\n", IM_DIM_H, IM_DIM_H, IM_IN_CH, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H, CONV_STRIDE_H);
//     //convolve_HWC_q7_Ch4_Direct_HWNC_SIMD_only4ch
    
//     //convolve_HWC_q7_Ch4_Direct_HWNC_SIMD_only8ch
//     //convolve_HWC_q7_Ch4_Direct_HWNC_SIMD_Optim_ch
//     // arm_convolve_HWC_q7_RGB_Im2Col(img_buffer, IM_DIM[i], IM_IN_CH, conv_wt_nhwc, out_ch[j], CONV_STRIDE_H, CONV_PADDING_H,
//                 //             CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res1, CONV_OUT_DIM_H,
//                 //             (q15_t *) col_buffer, NULL);     
//     // arm_convolve_HWC_q7_fast_Im2Col(img_buffer, IM_DIM[i], IM_IN_CH, conv_wt_nhwc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,
//     //                         CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res1, CONV_OUT_DIM_H,
//     //                          (q15_t *) col_buffer, NULL);
//     // convolve_HWC_q7_Ch4_Direct_HWNC_SIMD_Optim_ch(img_buffer, IM_DIM[i], IM_IN_CH, conv_wt_hwnc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,
//     //                             CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res2, CONV_OUT_DIM_H,
//     //                             (q31_t *) conv_buf, NULL);
//     // convolve_HWC_q7_RGB_Direct_HWNC_SIMD_fast(img_buffer, IM_DIM[i], IM_IN_CH, conv_wt_hwnc, CONV_OUT_CH, CONV_KER_DIM_H, CONV_PADDING_H,
//     //                             CONV_STRIDE_H, conv_bias, CONV_BIAS_LSHIFT, CONV_OUT_RSHIFT, res1, CONV_OUT_DIM_H,
//     //                             (q31_t *) conv_buf, NULL);
    
//     // if (validate_s8(res1, res2, CONV_OUT_CH * CONV_OUT_DIM_H * CONV_OUT_DIM_W)){
//     //     printf("Custom Conv is correct\n");
//     // }else{
//     //     printf("Custom Conv is wrong\n");
//     // }

//     delay(3000);
//     printf("-----------------End of Comparision----------------------\n");
// }
