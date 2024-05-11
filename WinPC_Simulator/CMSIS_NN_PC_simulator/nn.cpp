//#pragma once
//#include "nn.h"
//#include "arm_nnfunctions.h"
//#include <iostream>
//#include <fstream>
//#include <memory>
//
//q7_t stage0_conv_out[STAGE0_CONV_OUT_CH*STAGE0_CONV_OUT_DIM*STAGE0_CONV_OUT_DIM];
//q7_t stage1_0_conv_out[STAGE1_0_CONV_OUT_CH*STAGE1_0_CONV_OUT_DIM*STAGE1_0_CONV_OUT_DIM];
//q7_t stage2_0_conv_out[STAGE2_0_CONV_OUT_CH*STAGE2_0_CONV_OUT_DIM*STAGE2_0_CONV_OUT_DIM];
//q7_t stage3_0_conv_out[STAGE3_0_CONV_OUT_CH*STAGE3_0_CONV_OUT_DIM*STAGE3_0_CONV_OUT_DIM];
//q7_t stage4_0_conv_out[STAGE4_0_CONV_OUT_CH*STAGE4_0_CONV_OUT_DIM*STAGE4_0_CONV_OUT_DIM];
//q7_t gap21_out[STAGE4_0_CONV_OUT_CH*GAP21_OUT_DIM*GAP21_OUT_DIM];
//q7_t linear_out[LINEAR_OUT];
//
//void save(const char* file, q7_t* out, size_t sz)
//{
//    std::ofstream fp(file, std::ios::binary);
//    fp.write(reinterpret_cast<char*>(out), sz);
//    fp.close();
//    //std::cout << "Saved " << file << std::endl;
//}
////
////
////uint32_t network(q7_t* input)
////{
////    arm_convolve_HWC_q7_RGB(input, CONV1_IM_DIM, CONV1_IM_CH, conv1_w, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING,
////        CONV1_STRIDE, conv1_b, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, conv1_out, CONV1_OUT_DIM,
////        (q15_t*)conv_buffer, NULL);
////
////    save("logs/conv1_out.raw", conv1_out, sizeof(conv1_out));
////    arm_maxpool_q7_HWC(conv1_out, POOL1_IM_DIM, POOL1_IM_CH, POOL1_KER_DIM, POOL1_PADDING, POOL1_STRIDE, POOL1_OUT_DIM, conv_buffer, pool1_out);
////    arm_relu_q7(pool1_out, POOL1_OUT_DIM * POOL1_OUT_DIM * CONV1_OUT_CH);
////
////    arm_convolve_HWC_q7_fast(pool1_out, CONV2_IM_DIM, CONV2_IM_CH, conv2_w, CONV2_OUT_CH, CONV2_KER_DIM,
////        CONV2_PADDING, CONV2_STRIDE, conv2_b, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, conv2_out,
////        CONV2_OUT_DIM, (q15_t*)conv_buffer, NULL);
////    save("logs/conv2_out.raw", conv2_out, sizeof(conv2_out));
////    arm_avepool_q7_HWC(conv2_out, POOL2_IM_DIM, POOL2_IM_CH, POOL2_KER_DIM, POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, conv_buffer, pool2_out);
////    arm_relu_q7(pool2_out, POOL2_OUT_DIM * POOL2_OUT_DIM * CONV2_OUT_CH);
////
////    arm_convolve_HWC_q7_fast(pool2_out, CONV3_IM_DIM, CONV3_IM_CH, conv3_w, CONV3_OUT_CH, CONV3_KER_DIM,
////        CONV3_PADDING, CONV3_STRIDE, conv3_b, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, conv3_out,
////        CONV3_OUT_DIM, (q15_t*)conv_buffer, NULL);
////    save("logs/conv3_out.raw", conv3_out, sizeof(conv3_out));
////    arm_avepool_q7_HWC(conv3_out, POOL3_IM_DIM, POOL3_IM_CH, POOL3_KER_DIM, POOL3_PADDING, POOL3_STRIDE, POOL3_OUT_DIM, conv_buffer, pool3_out);
////    arm_relu_q7(pool3_out, POOL3_OUT_DIM * POOL3_OUT_DIM * CONV3_OUT_CH);
////
////    arm_fully_connected_q7_opt(pool3_out, fc1_w, FC1_DIM, FC1_OUT, FC1_BIAS_LSHIFT, FC1_OUT_RSHIFT, fc1_b,
////        fc1_out, (q15_t*)fc_buffer);
////    save("logs/fc1_out.raw", fc1_out, sizeof(fc1_out));
////
////    arm_softmax_q7(fc1_out, FC1_OUT, y_out);
////    save("logs/y_out.raw", y_out, sizeof(y_out));
////
////    uint32_t index[1];
////    q7_t result[1];
////    uint32_t blockSize = sizeof(y_out);
////
////    arm_max_q7(y_out, blockSize, result, index);
////    //printf("Classified class %i\n", index[0]);
////
////    return index[0];
////}
//
//q7_t* run_nn(q7_t* input_data, q7_t* output_data, q7_t* buffer1, q7_t* buffer2, q7_t* col_buffer, q7_t* fc_buffer) {
//    arm_convolve_HWC_q7_RGB(input_data,
//                                STAGE0_CONV_IM_DIM,
//                                STAGE0_CONV_IM_CH,
//                                stage0_conv_wt,
//                                STAGE0_CONV_OUT_CH,
//                                STAGE0_CONV_KER_DIM,
//                                STAGE0_CONV_PADDING,
//                                STAGE0_CONV_STRIDE,
//                                stage0_conv_bias,
//                                STAGE0_CONV_BIAS_LSHIFT,
//                                STAGE0_CONV_OUT_RSHIFT,
//                                stage0_conv_out,
//                                STAGE0_CONV_OUT_DIM,
//                                (q15_t*)col_buffer,
//                                NULL
//                                );
//    save("e:/2_Quantization/deployment-with-CMSIS-NN/CMSIS_NN_PC_simulator/Deploy_Simulator/logs/stage0_conv_out.raw", stage0_conv_out, sizeof(stage0_conv_out));
//    arm_relu_q7(stage0_conv_out, STAGE0_CONV_OUT_DIM*STAGE0_CONV_OUT_DIM*STAGE0_CONV_OUT_CH);
//    arm_convolve_HWC_q7_fast(stage0_conv_out,
//                                        STAGE1_0_CONV_IM_DIM,
//                                        STAGE1_0_CONV_IM_CH,
//                                        stage1_0_conv_wt,
//                                        STAGE1_0_CONV_OUT_CH,
//                                        STAGE1_0_CONV_KER_DIM,
//                                        STAGE1_0_CONV_PADDING,
//                                        STAGE1_0_CONV_STRIDE,
//                                        stage1_0_conv_bias,
//                                        STAGE1_0_CONV_BIAS_LSHIFT,
//                                        STAGE1_0_CONV_OUT_RSHIFT,
//                                        stage1_0_conv_out,
//                                        STAGE1_0_CONV_OUT_DIM,
//                                        (q15_t*)col_buffer,
//                                        NULL);
//    save("e:/2_Quantization/deployment-with-CMSIS-NN/CMSIS_NN_PC_simulator/Deploy_Simulator/logs/stage1_0_conv_out.raw", stage1_0_conv_out, sizeof(stage1_0_conv_out));
//    arm_relu_q7(stage1_0_conv_out, STAGE1_0_CONV_OUT_DIM*STAGE1_0_CONV_OUT_DIM*STAGE1_0_CONV_OUT_CH);
//    arm_convolve_HWC_q7_fast(stage1_0_conv_out,
//                                        STAGE2_0_CONV_IM_DIM,
//                                        STAGE2_0_CONV_IM_CH,
//                                        stage2_0_conv_wt,
//                                        STAGE2_0_CONV_OUT_CH,
//                                        STAGE2_0_CONV_KER_DIM,
//                                        STAGE2_0_CONV_PADDING,
//                                        STAGE2_0_CONV_STRIDE,
//                                        stage2_0_conv_bias,
//                                        STAGE2_0_CONV_BIAS_LSHIFT,
//                                        STAGE2_0_CONV_OUT_RSHIFT,
//                                        stage2_0_conv_out,
//                                        STAGE2_0_CONV_OUT_DIM,
//                                        (q15_t*)col_buffer,
//                                        NULL);
//    save("e:/2_Quantization/deployment-with-CMSIS-NN/CMSIS_NN_PC_simulator/Deploy_Simulator/logs/stage2_0_conv_out.raw", stage2_0_conv_out, sizeof(stage2_0_conv_out));
//    arm_relu_q7(stage2_0_conv_out, STAGE2_0_CONV_OUT_DIM*STAGE2_0_CONV_OUT_DIM*STAGE2_0_CONV_OUT_CH);
//    arm_convolve_HWC_q7_fast(stage2_0_conv_out,
//                                        STAGE3_0_CONV_IM_DIM,
//                                        STAGE3_0_CONV_IM_CH,
//                                        stage3_0_conv_wt,
//                                        STAGE3_0_CONV_OUT_CH,
//                                        STAGE3_0_CONV_KER_DIM,
//                                        STAGE3_0_CONV_PADDING,
//                                        STAGE3_0_CONV_STRIDE,
//                                        stage3_0_conv_bias,
//                                        STAGE3_0_CONV_BIAS_LSHIFT,
//                                        STAGE3_0_CONV_OUT_RSHIFT,
//                                        stage3_0_conv_out,
//                                        STAGE3_0_CONV_OUT_DIM,
//                                        (q15_t*)col_buffer,
//                                        NULL);
//    save("e:/2_Quantization/deployment-with-CMSIS-NN/CMSIS_NN_PC_simulator/Deploy_Simulator/logs/stage3_0_conv_out.raw", stage3_0_conv_out, sizeof(stage3_0_conv_out));
//    arm_relu_q7(stage3_0_conv_out, STAGE3_0_CONV_OUT_DIM*STAGE3_0_CONV_OUT_DIM*STAGE3_0_CONV_OUT_CH);
//    arm_convolve_HWC_q7_fast(stage3_0_conv_out,
//                                        STAGE4_0_CONV_IM_DIM,
//                                        STAGE4_0_CONV_IM_CH,
//                                        stage4_0_conv_wt,
//                                        STAGE4_0_CONV_OUT_CH,
//                                        STAGE4_0_CONV_KER_DIM,
//                                        STAGE4_0_CONV_PADDING,
//                                        STAGE4_0_CONV_STRIDE,
//                                        stage4_0_conv_bias,
//                                        STAGE4_0_CONV_BIAS_LSHIFT,
//                                        STAGE4_0_CONV_OUT_RSHIFT,
//                                        stage4_0_conv_out,
//                                        STAGE4_0_CONV_OUT_DIM,
//                                        (q15_t*)col_buffer,
//                                        NULL);
//    save("e:/2_Quantization/deployment-with-CMSIS-NN/CMSIS_NN_PC_simulator/Deploy_Simulator/logs/stage4_0_conv_out.raw", stage4_0_conv_out, sizeof(stage4_0_conv_out));
//    arm_relu_q7(stage4_0_conv_out, STAGE4_0_CONV_OUT_DIM*STAGE4_0_CONV_OUT_DIM*STAGE4_0_CONV_OUT_CH);
//    arm_avepool_q7_HWC(stage4_0_conv_out,
//                                    GAP21_IM_DIM,
//                                    GAP21_IM_CH,
//                                    GAP21_KER_DIM,
//                                    GAP21_PADDING,
//                                    GAP21_STRIDE,
//                                    GAP21_OUT_DIM,
//                                    col_buffer,
//                                    gap21_out);
//    save("e:/2_Quantization/deployment-with-CMSIS-NN/CMSIS_NN_PC_simulator/Deploy_Simulator/logs/gap21_out.raw", gap21_out, sizeof(gap21_out));
//    arm_fully_connected_q7_opt(gap21_out,
//                                    linear_wt,
//                                    LINEAR_DIM,
//                                    LINEAR_OUT,
//                                    LINEAR_BIAS_LSHIFT,
//                                    LINEAR_OUT_RSHIFT,
//                                    linear_bias,
//                                    linear_out,
//                                    (q15_t*)fc_buffer
//                                    );
//    save("e:/2_Quantization/deployment-with-CMSIS-NN/CMSIS_NN_PC_simulator/Deploy_Simulator/logs/linear_out.raw", linear_out, sizeof(linear_out));
//    return linear_out;
//}
