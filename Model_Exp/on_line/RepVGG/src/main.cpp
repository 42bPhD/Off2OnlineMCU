#include <stdio.h>
#include <Arduino.h>
#include "arm_nnsupportfunctions.h"
#include "arm_math.h"
#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "weights.h"
#include "inputs.h"
// #include "arm_nnexamples_cifar10_parameter.h"
// #include "arm_nnexamples_cifar10_weights.h"
#include "modules.h"
#include "memory_pack.h"
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


#define PAD 1
#define STRIDE 2
#define INPUT_SIZE 96
#define INPUT_CH 3
#define OUTPUT_CH 16
#define KERNEL_SIZE 3
#define CONV1_OUT_DIM  48
#define CONV1_OUT_CH  16
#define CONV2_OUT_DIM  24
#define CONV2_OUT_CH  16
#define CONV3_OUT_DIM  12
#define CONV3_OUT_CH  32
#define CONV4_OUT_DIM  6
#define CONV4_OUT_CH  64
#define CONV5_OUT_DIM  3
#define CONV5_OUT_CH  128

static q7_t conv1_bias[CONV1_OUT_CH] = {0};
static q7_t conv2_weights[KERNEL_SIZE * KERNEL_SIZE * CONV1_OUT_CH * CONV2_OUT_CH] = {0};
static q7_t conv2_bias[CONV2_OUT_CH] = {0};
static q7_t conv3_weights[KERNEL_SIZE * KERNEL_SIZE * CONV2_OUT_CH * CONV3_OUT_CH] = {0};
static q7_t conv3_bias[CONV3_OUT_CH] = {0};
static q7_t conv4_weights[KERNEL_SIZE * KERNEL_SIZE * CONV3_OUT_CH * CONV4_OUT_CH] = {0};
static q7_t conv4_bias[CONV4_OUT_CH] = {0};
static q7_t conv5_weights[KERNEL_SIZE * KERNEL_SIZE * CONV4_OUT_CH * CONV5_OUT_CH] = {0};
static q7_t conv5_bias[CONV5_OUT_CH] = {0};
static q7_t fc1_weights[CONV5_OUT_CH * 2] = {0};
static q7_t col_buffer[2 * KERNEL_SIZE * KERNEL_SIZE * CONV1_OUT_CH] = {0};
static q7_t scratch_buffer[CONV1_OUT_CH * CONV1_OUT_DIM * CONV1_OUT_DIM] = {0};

void repvgg_main(){
    
    for (int i=0; i<CONV1_OUT_CH; i++){
        conv1_bias[i] = rand()%256;
    }
    
    for (int i=0; i<KERNEL_SIZE * KERNEL_SIZE * CONV1_OUT_CH * CONV2_OUT_CH; i++){
        conv2_weights[i] = rand()%256;
    }
    
    for (int i=0; i<CONV2_OUT_CH; i++){
        conv2_bias[i] = rand()%256;
    }
    
    for (int i=0; i<KERNEL_SIZE * KERNEL_SIZE * CONV2_OUT_CH * CONV3_OUT_CH; i++){
        conv3_weights[i] = rand()%256;
    }
    for (int i=0; i<CONV3_OUT_CH; i++){
        conv3_bias[i] = rand()%256;
    }
    
    for (int i=0; i<KERNEL_SIZE * KERNEL_SIZE * CONV3_OUT_CH * CONV4_OUT_CH; i++){
        conv4_weights[i] = rand()%256;
    }

    for (int i=0; i<CONV4_OUT_CH; i++){
        conv4_bias[i] = rand()%256;
    }
    
    for (int i=0; i<KERNEL_SIZE * KERNEL_SIZE * CONV4_OUT_CH * CONV5_OUT_CH; i++){
        conv5_weights[i] = rand()%256;
    }
    for (int i=0; i<CONV5_OUT_CH; i++){
        conv5_bias[i] = rand()%256;
    }
    
    for (int i=0; i<CONV5_OUT_CH * 2; i++){
        fc1_weights[i] = rand()%256;
    }
}
void setup()
{
    Serial.begin(115200);
    SCB_EnableICache();
    SCB_EnableDCache();
    repvgg_main();
}


// void cifar10_main()
// {
//     /* start the execution */
    
//     /**/
//     //memset(scratch_buffer_tmp, 0, CONV1_OUT_CH * CONV1_OUT_CH * 10 * 4);
//     //q7_t *img_buffer1_tmp = scratch_buffer_tmp;
//     //q7_t *img_buffer2_tmp = img_buffer1_tmp + 32 * 32 * 32;
//     printf("start execution\n");
    
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


void loop()
{
    Serial.println("Computations done...");
    
    reset_cnt();
    start_cnt();
    convolve_HWC_q7_RGB_Direct_HWNC_SIMD_fast(input_data, INPUT_SIZE, INPUT_CH, conv_wt_hwnc, CONV1_OUT_CH, KERNEL_SIZE, PAD,
                            STRIDE, conv1_bias, 0, 0, scratch_buffer, CONV1_OUT_DIM,
                            (q31_t *)col_buffer, NULL);
    stop_cnt();
    printf("Conv1 Clock Cycle Count = %d\n", getCycles());
    arm_relu_q7(scratch_buffer, CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_OUT_CH);
    reset_cnt();
    start_cnt();
    convolve_HWC_q7_Ch4_Direct_HWNC_SIMD_Optim_ch(scratch_buffer, CONV1_OUT_DIM, CONV1_OUT_CH, conv2_weights, CONV2_OUT_CH, KERNEL_SIZE,
                            PAD, STRIDE, conv2_bias, 0, 0, scratch_buffer, CONV2_OUT_DIM,
                            (q31_t *)col_buffer, NULL);
    stop_cnt();
    printf("Conv2 Clock Cycle Count = %d\n", getCycles());
    arm_relu_q7(scratch_buffer, CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH);
    reset_cnt();
    start_cnt();
    convolve_HWC_q7_Ch4_Direct_HWNC_SIMD_Optim_ch(scratch_buffer, CONV2_OUT_DIM, CONV2_OUT_CH, conv3_weights, CONV3_OUT_CH, KERNEL_SIZE,
                            PAD, STRIDE, conv3_bias, 0, 0, scratch_buffer, CONV3_OUT_DIM,
                            (q31_t *)col_buffer, NULL);
    stop_cnt();
    printf("Conv3 Clock Cycle Count = %d\n", getCycles());
    arm_relu_q7(scratch_buffer, CONV3_OUT_DIM * CONV3_OUT_DIM * CONV3_OUT_CH);
    reset_cnt();
    start_cnt();
    convolve_HWC_q7_Ch4_Direct_HWNC_SIMD_Optim_ch(scratch_buffer, CONV3_OUT_DIM, CONV3_OUT_CH, conv4_weights, CONV4_OUT_CH, KERNEL_SIZE,
                            PAD, STRIDE, conv4_bias, 0, 0, scratch_buffer, CONV4_OUT_DIM,
                            (q31_t *)col_buffer, NULL);
    stop_cnt();
    printf("Conv4 Clock Cycle Count = %d\n", getCycles());
    arm_relu_q7(scratch_buffer, CONV4_OUT_DIM * CONV4_OUT_DIM * CONV4_OUT_CH);
    reset_cnt();
    start_cnt();
    convolve_HWC_q7_Ch4_Direct_HWNC_SIMD_Optim_ch(scratch_buffer, CONV4_OUT_DIM, CONV4_OUT_CH, conv5_weights, CONV5_OUT_CH, KERNEL_SIZE,
                            PAD, STRIDE, conv5_bias, 0, 0, scratch_buffer, CONV5_OUT_DIM,
                            (q31_t *)col_buffer, NULL);
    stop_cnt();
    printf("Conv5 Clock Cycle Count = %d\n", getCycles());
    arm_relu_q7(scratch_buffer, CONV5_OUT_DIM * CONV5_OUT_DIM * CONV5_OUT_CH);
    
    arm_fully_connected_q7_opt(scratch_buffer, fc1_weights, CONV5_OUT_CH, 2, 0, 0, conv5_bias,
                                scratch_buffer, (q15_t *)col_buffer);
    
    arm_softmax_q7(scratch_buffer, 2, scratch_buffer);
    delay(3000);
}