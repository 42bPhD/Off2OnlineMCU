#ifndef _TEST_INPUTS_H
#define _TEST_INPUTS_H

#include <stdint.h>
#include "parameter.h"
#include "weights.h"

#define DTCM __attribute__((section(".DTCM")))
static q7_t conv_bias[CONV_OUT_CH] = CONV_BIAS;
static q7_t input_data[IM_IN_CH * IM_DIM_H * IM_DIM_W] = IM_IN;
static q7_t conv_wt_nhwc[CONV_IM_CH * CONV_KER_DIM_H * CONV_KER_DIM_W * CONV_OUT_CH] = NHWC_WEIGHT;
static q7_t conv_wt_hwnc[CONV_IM_CH * CONV_KER_DIM_H * CONV_KER_DIM_W * CONV_OUT_CH] = HWNC_WEIGHT;
static q7_t col_buffer[2 * CONV_KER_DIM_H * CONV_KER_DIM_W * CONV_OUT_CH * 2];
static q7_t conv_out_nhwc[CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W];
static DTCM q7_t conv_out_hwnc[CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W];
static DTCM q31_t conv_buf[CONV_OUT_CH];
static DTCM q15_t conv_out_buf[CONV_IM_CH];

#endif
