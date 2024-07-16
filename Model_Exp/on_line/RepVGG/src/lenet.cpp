// #include <stdio.h>
// #include <Arduino.h>
// #include "arm_nnsupportfunctions.h"
// #include "arm_math.h"
// #include <stdlib.h>
// #include <arm_nnfunctions.h>

// // #include "arm_nnexamples_cifar10_parameter.h"
// // #include "arm_nnexamples_cifar10_weights.h"
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


// #define PAD 1
// #define STRIDE 2
// #define INPUT_SIZE 32
// #define INPUT_CH 3
// #define OUTPUT_CH 32
// #define KERNEL_SIZE 3
// #define CONV1_OUT_DIM  16
// #define CONV1_OUT_CH  32
// #define CONV2_OUT_DIM  16
// #define CONV2_OUT_CH  16
// #define CONV3_OUT_DIM  8
// #define CONV3_OUT_CH  32
// #define CONV4_OUT_DIM  4
// #define CONV4_OUT_CH  32
// #define CONV5_OUT_DIM  2
// #define CONV5_OUT_CH  64

// static q7_t input[INPUT_SIZE * INPUT_SIZE * INPUT_CH] = {0};
// static q7_t conv1_weights[KERNEL_SIZE * KERNEL_SIZE * INPUT_CH * CONV1_OUT_CH] = {0};
// static q7_t conv1_bias[CONV1_OUT_CH] = {0};
// static q7_t conv2_weights[KERNEL_SIZE * KERNEL_SIZE * CONV1_OUT_CH * CONV2_OUT_CH] = {0};
// static q7_t conv2_bias[CONV2_OUT_CH] = {0};
// static q7_t conv3_weights[KERNEL_SIZE * KERNEL_SIZE * CONV2_OUT_CH * CONV3_OUT_CH] = {0};
// static q7_t conv3_bias[CONV3_OUT_CH] = {0};
// static q7_t conv4_weights[KERNEL_SIZE * KERNEL_SIZE * CONV3_OUT_CH * CONV4_OUT_CH] = {0};
// static q7_t conv4_bias[CONV4_OUT_CH] = {0};
// static q7_t conv5_weights[KERNEL_SIZE * KERNEL_SIZE * CONV4_OUT_CH * CONV5_OUT_CH] = {0};
// static q7_t conv5_bias[CONV5_OUT_CH] = {0};
// static q7_t fc1_weights[CONV5_OUT_CH * 2] = {0};
// static q7_t col_buffer[2 * KERNEL_SIZE * KERNEL_SIZE * CONV1_OUT_CH] = {0};
// static q7_t scratch_buffer[CONV1_OUT_CH * CONV1_OUT_DIM * CONV1_OUT_DIM] = {0};

// void repvgg_main(){
    
//     for (int i=0; i<INPUT_SIZE*INPUT_SIZE*INPUT_CH; i++){
//         input[i] = rand()%256;
//     }
//     for (int i=0; i<KERNEL_SIZE * KERNEL_SIZE * INPUT_CH * CONV1_OUT_CH; i++){
//         conv1_weights[i] = rand()%256;
//     }
//     for (int i=0; i<CONV1_OUT_CH; i++){
//         conv1_bias[i] = rand()%256;
//     }
    
//     for (int i=0; i<KERNEL_SIZE * KERNEL_SIZE * CONV1_OUT_CH * CONV2_OUT_CH; i++){
//         conv2_weights[i] = rand()%256;
//     }
    
//     for (int i=0; i<CONV2_OUT_CH; i++){
//         conv2_bias[i] = rand()%256;
//     }
    
//     for (int i=0; i<KERNEL_SIZE * KERNEL_SIZE * CONV2_OUT_CH * CONV3_OUT_CH; i++){
//         conv3_weights[i] = rand()%256;
//     }
//     for (int i=0; i<CONV3_OUT_CH; i++){
//         conv3_bias[i] = rand()%256;
//     }
    
//     for (int i=0; i<KERNEL_SIZE * KERNEL_SIZE * CONV3_OUT_CH * CONV4_OUT_CH; i++){
//         conv4_weights[i] = rand()%256;
//     }

//     for (int i=0; i<CONV4_OUT_CH; i++){
//         conv4_bias[i] = rand()%256;
//     }
    
//     for (int i=0; i<KERNEL_SIZE * KERNEL_SIZE * CONV4_OUT_CH * CONV5_OUT_CH; i++){
//         conv5_weights[i] = rand()%256;
//     }
//     for (int i=0; i<CONV5_OUT_CH; i++){
//         conv5_bias[i] = rand()%256;
//     }
    
//     for (int i=0; i<CONV5_OUT_CH * 2; i++){
//         fc1_weights[i] = rand()%256;
//     }
// }
// void setup()
// {
//     Serial.begin(115200);
//     SCB_EnableICache();
//     SCB_EnableDCache();
//     repvgg_main();
// }



// void loop()
// {
//     Serial.println("Computations done...");
//     for (int i=0; i<INPUT_SIZE*INPUT_SIZE*INPUT_CH; i++){
//         input[i] = rand()%256;
//     }

//     reset_cnt();
//     start_cnt();
//     arm_convolve_HWC_q7_RGB_Im2Col(input, INPUT_SIZE, INPUT_CH, conv1_weights, CONV1_OUT_CH, KERNEL_SIZE, PAD,
//                             STRIDE, conv1_bias, 0, 0, scratch_buffer, CONV1_OUT_DIM,
//                             (q15_t *)col_buffer, NULL);
//     stop_cnt();
//     printf("Conv1 Clock Cycle Count = %d\n", getCycles());
//     arm_relu_q7(scratch_buffer, CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_OUT_CH);
//     reset_cnt();
//     start_cnt();
//     convolve_HWC_q7_Ch4_Direct_HWNC_SIMD_Optim_ch(scratch_buffer, CONV1_OUT_DIM, CONV1_OUT_CH, conv2_weights, CONV2_OUT_CH, KERNEL_SIZE,
//                             PAD, STRIDE, conv2_bias, 0, 0, scratch_buffer, CONV2_OUT_DIM,
//                             (q31_t *)col_buffer, NULL);
//     stop_cnt();
//     printf("Conv2 Clock Cycle Count = %d\n", getCycles());
//     arm_relu_q7(scratch_buffer, CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH);
//     reset_cnt();
//     start_cnt();
//     arm_convolve_HWC_q7_fast_Im2Col(scratch_buffer, CONV2_OUT_DIM, CONV2_OUT_CH, conv3_weights, CONV3_OUT_CH, KERNEL_SIZE,
//                             PAD, STRIDE, conv3_bias, 0, 0, scratch_buffer, CONV3_OUT_DIM,
//                             (q15_t *)col_buffer, NULL);
//     stop_cnt();
//     printf("Conv3 Clock Cycle Count = %d\n", getCycles());
//     arm_relu_q7(scratch_buffer, CONV3_OUT_DIM * CONV3_OUT_DIM * CONV3_OUT_CH);
//     reset_cnt();
//     start_cnt();
//     arm_convolve_HWC_q7_fast_Im2Col(scratch_buffer, CONV3_OUT_DIM, CONV3_OUT_CH, conv4_weights, CONV4_OUT_CH, KERNEL_SIZE,
//                             PAD, STRIDE, conv4_bias, 0, 0, scratch_buffer, CONV4_OUT_DIM,
//                             (q15_t *)col_buffer, NULL);
//     stop_cnt();
//     printf("Conv4 Clock Cycle Count = %d\n", getCycles());
//     arm_relu_q7(scratch_buffer, CONV4_OUT_DIM * CONV4_OUT_DIM * CONV4_OUT_CH);
//     reset_cnt();
//     start_cnt();
//     arm_convolve_HWC_q7_fast_Im2Col(scratch_buffer, CONV4_OUT_DIM, CONV4_OUT_CH, conv5_weights, CONV5_OUT_CH, KERNEL_SIZE,
//                             PAD, STRIDE, conv5_bias, 0, 0, scratch_buffer, CONV5_OUT_DIM,
//                             (q15_t *)col_buffer, NULL);
//     stop_cnt();
//     printf("Conv5 Clock Cycle Count = %d\n", getCycles());
//     arm_relu_q7(scratch_buffer, CONV5_OUT_DIM * CONV5_OUT_DIM * CONV5_OUT_CH);
    
//     arm_fully_connected_q7_opt(scratch_buffer, fc1_weights, CONV5_OUT_CH, 2, 0, 0, conv5_bias,
//                                 scratch_buffer, (q15_t *)col_buffer);
    
//     arm_softmax_q7(scratch_buffer, 2, scratch_buffer);
//     delay(3000);
// }