// #include <stdio.h>
// #include <Arduino.h>
// #include "arm_nnsupportfunctions.h"
// #include "arm_math.h"
// #include <stdlib.h>
// #include <arm_nnfunctions.h>

// // #include "arm_nnexamples_cifar10_parameter.h"
// // #include "arm_nnexamples_cifar10_weights.h"
// #include "arm_nnexamples_cifar10_inputs.h"
// #include "lenet_parameter.h"
// #include "lenet_weights.h"
// #include "modules.h"
// #include "memory_pack.h"
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

// static q7_t conv1_wt[CONV1_WT_SHAPE] = CONV1_WT;
// static q7_t conv1_bias[CONV1_BIAS_SHAPE] = CONV1_BIAS;

// static q7_t conv2_wt[CONV2_WT_SHAPE] = CONV2_WT;
// static q7_t conv2_bias[CONV2_OUT_CH] = CONV2_BIAS;

// static q7_t conv3_wt[CONV3_WT_SHAPE] = CONV3_WT;
// static q7_t conv3_bias[CONV3_OUT_CH] = CONV3_BIAS;

// static q7_t ip1_wt[FC1_WT_SHAPE] = FC1_WT;
// static q7_t ip1_bias[FC1_BIAS_SHAPE] = FC1_BIAS;

// /* Here the image_data should be the raw uint8 type RGB image in [RGB, RGB, RGB ... RGB] format */
// uint8_t image_data[CONV1_IM_CH * CONV1_IM_DIM * CONV1_IM_DIM] = IMG_DATA1;
// q7_t output_data[FC1_OUT];
// #define DTCM __attribute__((section(".DTCM")))
// static DTCM q7_t conv_buf[4 * CONV1_OUT_CH];
// //vector buffer: max(im2col buffer,average pool buffer, fully connected buffer)
// q7_t col_buffer[2 * MAX_CONV_BUFFER_SIZE];
// q7_t scratch_buffer[CONV1_OUT_CH * CONV1_OUT_CH * FC1_OUT * 4];
// q7_t scratch_buffer_tmp[CONV1_OUT_CH * CONV1_OUT_CH * FC1_OUT * 4];

// void setup()
// {
//     Serial.begin(115200);
//     SCB_EnableICache();
//     SCB_EnableDCache();
// }
// // #define PROPOSED_METHOD
// static q7_t conv1_wt_proposed[CONV1_WT_SHAPE] = CONV1_WT;
// static q7_t conv2_wt_proposed[CONV2_WT_SHAPE] = CONV2_WT;
// static q7_t conv3_wt_proposed[CONV3_WT_SHAPE] = CONV3_WT;


// void cifar10_main()
// {
//     /* start the execution */
//     memset(scratch_buffer, 0, CONV1_OUT_CH * CONV1_OUT_CH * 10 * 4);
//     q7_t *img_buffer1 = scratch_buffer;
//     q7_t *img_buffer2 = img_buffer1 + 32 * 32 * 32;

//     //memset(scratch_buffer_tmp, 0, CONV1_OUT_CH * CONV1_OUT_CH * 10 * 4);
//     //q7_t *img_buffer1_tmp = scratch_buffer_tmp;
//     //q7_t *img_buffer2_tmp = img_buffer1_tmp + 32 * 32 * 32;
//     printf("start execution\n");
    
//     /* input pre-processing */
//     int mean_data[3] = INPUT_MEAN_SHIFT;
//     unsigned int scale_data[3] = INPUT_RIGHT_SHIFT;
//     for (int i = 0; i < 32 * 32 * 3; i += 3)
//     {
//         img_buffer2[i] = (q7_t)__SSAT(((((int)image_data[i] - mean_data[0]) << 7) + (0x1 << (scale_data[0] - 1))) >> scale_data[0], 8);
//         img_buffer2[i + 1] = (q7_t)__SSAT(((((int)image_data[i + 1] - mean_data[1]) << 7) + (0x1 << (scale_data[1] - 1))) >> scale_data[1], 8);
//         img_buffer2[i + 2] = (q7_t)__SSAT(((((int)image_data[i + 2] - mean_data[2]) << 7) + (0x1 << (scale_data[2] - 1))) >> scale_data[2], 8);

//         // img_buffer2_tmp[i] = (q7_t)__SSAT(((((int)image_data[i] - mean_data[0]) << 7) + (0x1 << (scale_data[0] - 1))) >> scale_data[0], 8);
//         // img_buffer2_tmp[i + 1] = (q7_t)__SSAT(((((int)image_data[i + 1] - mean_data[1]) << 7) + (0x1 << (scale_data[1] - 1))) >> scale_data[1], 8);
//         // img_buffer2_tmp[i + 2] = (q7_t)__SSAT(((((int)image_data[i + 2] - mean_data[2]) << 7) + (0x1 << (scale_data[2] - 1))) >> scale_data[2], 8);
//     }
//     reset_cnt();
//     start_cnt();
//     //unsigned long s_time = millis();
    
//     #if defined(PROPOSED_METHOD)
//         convolve_HWC_q7_RGB_Direct_HWNC_SIMD_fast(img_buffer2, CONV1_IM_DIM, CONV1_IM_CH, conv1_wt_proposed, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING,
//                                 CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, img_buffer1, CONV1_OUT_DIM,
//                                 (q31_t *) conv_buf, NULL);
//     #else
//         arm_convolve_HWC_q7_RGB_Im2Col(img_buffer2, CONV1_IM_DIM, CONV1_IM_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING,
//                             CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, img_buffer1, CONV1_OUT_DIM,
//                             (q15_t *)col_buffer, NULL);
//     #endif
//     // unsigned long e_time = millis();
//     // printf("#############Conv-RGB Time = %d\n", e_time - s_time);
//     stop_cnt();
//     printf("#############Conv-RGB Conv1 Clock Cycle Count = %d###########\n", getCycles());
//     // if (validate_s8(img_buffer1, img_buffer1_tmp, CONV1_OUT_CH * CONV1_OUT_DIM * CONV1_OUT_DIM)){
//     //     printf("Custom Conv is correct\n");
//     // }else{
//     //     printf("Custom Conv is wrong\n");
//     // }
//     // return;
//     reset_cnt();
//     start_cnt();
//     //s_time = millis();
//     arm_relu_q7(img_buffer1, CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_OUT_CH);
//     // e_time = millis();
//     // printf("#############ReLU-Conv1 Time = %d\n", e_time - s_time);

//     stop_cnt();
//     printf("RELU Conv1 Clock Cycle Count = %d\n", getCycles());

//     reset_cnt();
//     start_cnt();
//     // s_time = millis();
//     arm_maxpool_q7_HWC(img_buffer1, CONV1_OUT_DIM, CONV1_OUT_CH, POOL1_KER_DIM,
//                        POOL1_PADDING, POOL1_STRIDE, POOL1_OUT_DIM, NULL, img_buffer2);
//     // e_time = millis();
//     // printf("#############MP-Conv1 Time = %d\n", e_time - s_time);
//     stop_cnt();
//     printf("MAXPOOL Conv1 Clock Cycle Count = %d\n", getCycles());

//     reset_cnt();
//     start_cnt();
//     //conv2 img_buffer2 -> img_buffer1
//     //s_time = millis();
//     #if defined(PROPOSED_METHOD)
//         convolve_HWC_q7_Ch4_Direct_HWNC_SIMD_Optim_ch(img_buffer2, CONV2_IM_DIM, CONV2_IM_CH, conv2_wt_proposed, CONV2_OUT_CH, CONV2_KER_DIM,
//                              CONV2_PADDING, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, img_buffer1,
//                              CONV2_OUT_DIM, (q31_t *)conv_buf, NULL);
//     #else
//         arm_convolve_HWC_q7_fast_Im2Col(img_buffer2, CONV2_IM_DIM, CONV2_IM_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM,
//                              CONV2_PADDING, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, img_buffer1,
//                              CONV2_OUT_DIM, (q15_t *)col_buffer, NULL);
//     #endif
//     // e_time = millis();
//     // printf("#############Ch4-Conv2 Time = %d\n", e_time - s_time);
//     stop_cnt();
//     printf("------------------CONV(EVEN) Conv2 Clock Cycle Count = %d-----------------\n", getCycles());

//     reset_cnt();
//     start_cnt();
//     //s_time = millis();
//     arm_relu_q7(img_buffer1, CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH);
//     // e_time = millis();
//     // printf("#############ReLU-Conv2 Time = %d\n", e_time - s_time);
//     stop_cnt();
//     printf("ReLU Conv2 Clock Cycle Count = %d\n", getCycles());
    
//     //pool2 img_buffer1 -> img_buffer2
//     reset_cnt();
//     start_cnt();
//     // s_time = millis();
//     arm_maxpool_q7_HWC(img_buffer1, CONV2_OUT_DIM, CONV2_OUT_CH, POOL2_KER_DIM,
//                        POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, conv_buf, img_buffer2);
//     // e_time = millis();
//     // printf("#############MP-Conv2 Time = %d\n", e_time - s_time);
//     stop_cnt();
//     printf("MAXPOOL Conv2 Clock Cycle Count = %d\n", getCycles());
    
//     reset_cnt();
//     start_cnt();
//     // conv3 img_buffer2 -> img_buffer1
//     //s_time  = millis();
//     #if defined(PROPOSED_METHOD)
//         convolve_HWC_q7_Ch4_Direct_HWNC_SIMD_Optim_ch(img_buffer2, CONV3_IM_DIM, CONV3_IM_CH, conv3_wt_proposed, CONV3_OUT_CH, CONV3_KER_DIM,
//                              CONV3_PADDING, CONV3_STRIDE, conv3_bias, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, img_buffer1,
//                              CONV3_OUT_DIM, (q31_t *)conv_buf, NULL);
//     #else
//         arm_convolve_HWC_q7_fast_Im2Col(img_buffer2, CONV3_IM_DIM, CONV3_IM_CH, conv3_wt, CONV3_OUT_CH, CONV3_KER_DIM,
//                                 CONV3_PADDING, CONV3_STRIDE, conv3_bias, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, img_buffer1,
//                                 CONV3_OUT_DIM, (q15_t *)col_buffer, NULL);
//     #endif
//     // e_time = millis();
//     // printf("#############Ch4-Conv3 Time = %d\n", e_time - s_time);
//     stop_cnt();
//     printf("-----------------Conv(EVEN) Conv3 Clock Cycle Count = %d----------------------\n", getCycles());

//     reset_cnt();
//     start_cnt();
//     //s_time = millis();
//     arm_relu_q7(img_buffer1, CONV3_OUT_DIM * CONV3_OUT_DIM * CONV3_OUT_CH);
//     // e_time = millis();
//     // printf("#############ReLU-Conv3 Time = %d\n", e_time - s_time);
//     stop_cnt();
//     printf("ReLU Conv3 Clock Cycle Count = %d\n", getCycles());

//     reset_cnt();
//     start_cnt();
//     //pool3 img_buffer-> img_buffer2
//     #if defined(PROPOSED_METHOD)
//         arm_maxpool_q7_HWC(img_buffer1, CONV3_OUT_DIM, CONV3_OUT_CH, POOL3_KER_DIM,
//                         POOL3_PADDING, POOL3_STRIDE, POOL3_OUT_DIM, conv_buf, img_buffer2);
//     #else
//         arm_maxpool_q7_HWC(img_buffer1, CONV3_OUT_DIM, CONV3_OUT_CH, POOL3_KER_DIM,
//                         POOL3_PADDING, POOL3_STRIDE, POOL3_OUT_DIM, col_buffer, img_buffer2);
//     #endif
//     // e_time = millis();
//     // printf("#############MP-Conv3 Time = %d\n", e_time - s_time);
//     stop_cnt();
//     printf("MAXPOOL Conv3 Clock Cycle Count = %d\n", getCycles());

//     reset_cnt();
//     start_cnt();

//     //ip1 img_buffer2 -> output_data
//     //s_time = millis();
//     arm_fully_connected_q7_opt(img_buffer2, ip1_wt, FC1_DIM, FC1_OUT, FC1_BIAS_LSHIFT, FC1_OUT_RSHIFT, ip1_bias,
//                                output_data, (q15_t *)img_buffer1);
//     // e_time = millis();
//     // printf("#############FCL Time = %d\n", e_time - s_time);
//     stop_cnt();
//     printf("FCL Clock Cycle Count = %d\n", getCycles());

//     reset_cnt();
//     start_cnt();
//     //s_time = millis();
//     arm_softmax_q7(output_data, 10, output_data);
//     // e_time = millis();
//     // printf("#############SoftMax Time = %d\n", e_time - s_time);
//     stop_cnt();
//     printf("SoftMax Clock Cycle Count = %d\n", getCycles());

//     // for (int i = 0; i < 10; i++)
//     // {
//     //     printf("%d: %d\n", i, output_data[i]);
//     // }
// }
// void loop()
// {
//     Serial.println("Computations done...");
    
// //    unsigned long start = millis();
//     cifar10_main();
//   //  unsigned long end = millis();
    
//     //printf("Cifar10 Models Time = %d\n", end - start);
    
//     delay(3000);
// }