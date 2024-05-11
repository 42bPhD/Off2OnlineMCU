//
//#pragma once
//#include "arm_nnfunctions.h"
//#include <stdint.h>
//#include <stdlib.h>
//#include <stdio.h>
//#include "inputs.h"
//#include "parameters.h"
//#include "weights.h"
//q7_t* run_nn(q7_t* input_data, q7_t* output_data, q7_t* buffer1, q7_t* buffer2,
//    q7_t* col_buffer, q7_t* fc_buffer);    
//static q7_t scratch_buffer[MAX_CONV_BUFFER_SIZE*4];
//static q7_t stage0_conv_wt[STAGE0_CONV_WT_SHAPE] = STAGE0_CONV_WT;
//static q7_t stage0_conv_bias[STAGE0_CONV_BIAS_SHAPE] = STAGE0_CONV_BIAS;
//static q7_t stage1_0_conv_wt[STAGE1_0_CONV_WT_SHAPE] = STAGE1_0_CONV_WT;
//static q7_t stage1_0_conv_bias[STAGE1_0_CONV_BIAS_SHAPE] = STAGE1_0_CONV_BIAS;
//static q7_t stage2_0_conv_wt[STAGE2_0_CONV_WT_SHAPE] = STAGE2_0_CONV_WT;
//static q7_t stage2_0_conv_bias[STAGE2_0_CONV_BIAS_SHAPE] = STAGE2_0_CONV_BIAS;
//static q7_t stage3_0_conv_wt[STAGE3_0_CONV_WT_SHAPE] = STAGE3_0_CONV_WT;
//static q7_t stage3_0_conv_bias[STAGE3_0_CONV_BIAS_SHAPE] = STAGE3_0_CONV_BIAS;
//static q7_t stage4_0_conv_wt[STAGE4_0_CONV_WT_SHAPE] = STAGE4_0_CONV_WT;
//static q7_t stage4_0_conv_bias[STAGE4_0_CONV_BIAS_SHAPE] = STAGE4_0_CONV_BIAS;
//static q7_t linear_wt[LINEAR_WT_SHAPE] = LINEAR_WT;
//static q7_t linear_bias[LINEAR_BIAS_SHAPE] = LINEAR_BIAS;
////uint32_t network(q7_t* input);
////static q7_t conv1_out[CONV1_OUT_CH * CONV1_OUT_DIM * CONV1_OUT_DIM];
////static q7_t pool1_out[CONV1_OUT_CH * POOL1_OUT_DIM * POOL1_OUT_DIM];
////static q7_t conv2_out[CONV2_OUT_CH * CONV2_OUT_DIM * CONV2_OUT_DIM];
////static q7_t pool2_out[CONV2_OUT_CH * POOL2_OUT_DIM * POOL2_OUT_DIM];
////static q7_t conv3_out[CONV3_OUT_CH * CONV3_OUT_DIM * CONV3_OUT_DIM];
////static q7_t pool3_out[CONV3_OUT_CH * POOL3_OUT_DIM * POOL3_OUT_DIM];
////static q7_t fc1_out[FC1_OUT];
////static q7_t y_out[FC1_OUT];
////static q7_t conv1_w[CONV1_WT_SHAPE] = CONV1_WT;
////static q7_t conv1_b[CONV1_BIAS_SHAPE] = CONV1_BIAS;
////static q7_t conv2_w[CONV2_WT_SHAPE] = CONV2_WT;
////static q7_t conv2_b[CONV2_BIAS_SHAPE] = CONV2_BIAS;
////static q7_t conv3_w[CONV3_WT_SHAPE] = CONV3_WT;
////static q7_t conv3_b[CONV3_BIAS_SHAPE] = CONV3_BIAS;
////static q7_t fc1_w[FC1_WT_SHAPE] = FC1_WT;
////static q7_t fc1_b[FC1_BIAS_SHAPE] = FC1_BIAS;
////static q7_t conv_buffer[MAX_CONV_BUFFER_SIZE];
////static q7_t fc_buffer[MAX_FC_BUFFER];