#include "modules.h"
#include "arm_math_types.h"


void arm_relu6_s8(q7_t* data, uint16_t size);
void arm_max_q7(
    const q7_t* pSrc,
    uint32_t blockSize,
    q7_t* pResult,
    uint32_t* pIndex);
void AdaptiveAvgPool2d_q7_HWC(q7_t* Im_in,
    const uint16_t dim_im_in,
    const uint16_t ch_im_in,
    const uint16_t dim_im_out,
    q7_t* bufferA,
    q7_t* Im_out);

/**
 * @brief Converts the elements from a q7 vector to a q15 vector with an added offset
 * @param[in]    src        pointer to the q7 input vector
 * @param[out]   dst        pointer to the q15 output vector
 * @param[in]    block_size length of the input vector
 * @param[in]    offset     q7 offset to be added to each input vector element.
 *
 * \par Description:
 *
 * The equation used for the conversion process is:
 *
 * <pre>
 *  dst[n] = (q15_t) src[n] + offset;   0 <= n < block_size.
 * </pre>
 *
 */
void arm_q7_to_q15_with_offset(const q7_t* src, q15_t* dst, uint32_t block_size, q15_t offset);


arm_status arm_convolve_HWC_q7_RGB(const q7_t* Im_in,
    const uint16_t dim_im_in,
    const uint16_t ch_im_in,
    const q7_t* wt,
    const uint16_t ch_im_out,
    const uint16_t dim_kernel,
    const uint16_t padding,
    const uint16_t stride,
    const q7_t* bias,
    const uint16_t bias_shift,
    const uint16_t out_shift,
    q7_t* Im_out, const uint16_t dim_im_out, q15_t* bufferA, q7_t* bufferB);

arm_status arm_convolve_HWC_q7_fast(const q7_t* Im_in,
    const uint16_t dim_im_in,
    const uint16_t ch_im_in,
    const q7_t* wt,
    const uint16_t ch_im_out,
    const uint16_t dim_kernel,
    const uint16_t padding,
    const uint16_t stride,
    const q7_t* bias,
    const uint16_t bias_shift,
    const uint16_t out_shift,
    q7_t* Im_out,
    const uint16_t dim_im_out,
    q15_t* bufferA,
    q7_t* bufferB);

arm_status arm_convolve_1x1_HWC_q7_fast_nonsquare(const q7_t* Im_in,
    const uint16_t dim_im_in_x,
    const uint16_t dim_im_in_y,
    const uint16_t ch_im_in,
    const q7_t* wt,
    const uint16_t ch_im_out,
    const uint16_t dim_kernel_x,
    const uint16_t dim_kernel_y,
    const uint16_t padding_x,
    const uint16_t padding_y,
    const uint16_t stride_x,
    const uint16_t stride_y,
    const q7_t* bias,
    const uint16_t bias_shift,
    const uint16_t out_shift,
    q7_t* Im_out,
    const uint16_t dim_im_out_x,
    const uint16_t dim_im_out_y,
    q15_t* bufferA,
    q7_t* bufferB);

void arm_relu_q7(q7_t* data, uint16_t size);


arm_status arm_fully_connected_q7(const q7_t* pV,
    const q7_t* pM,
    const uint16_t dim_vec,
    const uint16_t num_of_rows,
    const uint16_t bias_shift,
    const uint16_t out_shift, const q7_t* bias, q7_t* pOut, q15_t* vec_buffer);

arm_status arm_fully_connected_q7_opt(const q7_t* pV,
    const q7_t* pM,
    const uint16_t dim_vec,
    const uint16_t num_of_rows,
    const uint16_t bias_shift,
    const uint16_t out_shift,
    const q7_t* bias,
    q7_t* pOut,
    q15_t* vec_buffer);

void arm_maxpool_q7_HWC(q7_t* Im_in,
    const uint16_t dim_im_in,
    const uint16_t ch_im_in,
    const uint16_t dim_kernel,
    const uint16_t padding,
    const uint16_t stride, const uint16_t dim_im_out, q7_t* bufferA, q7_t* Im_out);

void arm_avepool_q7_HWC(q7_t* Im_in,
    const uint16_t dim_im_in,
    const uint16_t ch_im_in,
    const uint16_t dim_kernel,
    const uint16_t padding,
    const uint16_t stride, const uint16_t dim_im_out, q7_t* bufferA, q7_t* Im_out);



void arm_softmax_q7(const q7_t* vec_in, const uint16_t dim_vec, q7_t* p_out);


arm_status arm_depthwise_separable_conv_HWC_q7(const q7_t* Im_in,
    const uint16_t dim_im_in,
    const uint16_t ch_im_in,
    const q7_t* wt,
    const uint16_t ch_im_out,
    const uint16_t dim_kernel,
    const uint16_t padding,
    const uint16_t stride,
    const q7_t* bias,
    const uint16_t bias_shift,
    const uint16_t out_shift,
    q7_t* Im_out,
    const uint16_t dim_im_out,
    q15_t* bufferA,
    q7_t* bufferB);


arm_status arm_depthwise_separable_conv_HWC_q7_nonsquare(const q7_t* Im_in,
    const uint16_t dim_im_in_x,
    const uint16_t dim_im_in_y,
    const uint16_t ch_im_in,
    const q7_t* wt,
    const uint16_t ch_im_out,
    const uint16_t dim_kernel_x,
    const uint16_t dim_kernel_y,
    const uint16_t padding_x,
    const uint16_t padding_y,
    const uint16_t stride_x,
    const uint16_t stride_y,
    const q7_t* bias,
    const uint16_t bias_shift,
    const uint16_t out_shift,
    q7_t* Im_out,
    const uint16_t dim_im_out_x,
    const uint16_t dim_im_out_y,
    q15_t* bufferA,
    q7_t* bufferB);