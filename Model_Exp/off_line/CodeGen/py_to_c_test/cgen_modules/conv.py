"""
 * Copyright (C) 2010-2021 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""
from string import Template


def arm_convolve_HWC_q7_RGB(Im_in, 
                            dim_im_in, 
                            ch_im_in, 
                            wt, 
                            ch_im_out, 
                            dim_kernel, 
                            padding, 
                            stride, 
                            bias, 
                            bias_shift, 
                            out_shift,
                            Im_out,
                            dim_im_out,
                            bufferA,
                            bufferB=None):
    """
    * This kernel is written exclusively for convolution with ch_im_in equals 3. 
    * This applies on the first layer of CNNs which has input image with RGB format.
    * Buffer size:
    * bufferA size: 2*ch_im_in*dim_kernel*dim_kernel
    * bufferB size: 0
    
    * Input dimension constraints:
    * ch_im_in equals 3

    Args:
        * @brief Q7 convolution function for RGB image
        * @param[in]       Im_in       pointer to input tensor
        * @param[in]       dim_im_in   input tensor dimention
        * @param[in]       ch_im_in    number of input tensor channels
        * @param[in]       wt          pointer to kernel weights
        * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
        * @param[in]       dim_kernel  filter kernel size
        * @param[in]       padding     padding sizes
        * @param[in]       stride      convolution stride
        * @param[in]       bias        pointer to bias
        * @param[in]       bias_shift  amount of left-shift for bias
        * @param[in]       out_shift   amount of right-shift for output
        * @param[in,out]   Im_out      pointer to output tensor
        * @param[in]       dim_im_out  output tensor dimension
        * @param[in,out]   bufferA     pointer to buffer space for input
        * @param[in,out]   bufferB     pointer to buffer space for output
        * @return     The function returns either
    """
    bufferB = bufferB if bufferB is not None else "NULL"
    return """arm_convolve_HWC_q7_RGB(${Im_in},${dim_im_in},${ch_im_in},
    ${wt},${ch_im_out},${dim_kernel},
                                ${padding},
                                ${stride},
                                ${bias},
                                ${bias_shift},
                                ${out_shift},
                                ${Im_out},
                                ${dim_im_out},
                                (q15_t*)${bufferA},
                                ${bufferB});""".format(Im_in=Im_in,
                                                            dim_im_in=dim_im_in,
                                                            ch_im_in=ch_im_in,
                                                            wt=wt,
                                                            ch_im_out=ch_im_out,
                                                            dim_kernel=dim_kernel,
                                                            padding=padding,
                                                            stride=stride,
                                                            bias=bias,
                                                            bias_shift=bias_shift,
                                                            out_shift=out_shift,
                                                            Im_out=Im_out,
                                                            dim_im_out=dim_im_out,
                                                            bufferA=bufferA,
                                                            bufferB=bufferB)

def arm_convolve_HWC_q7_fast(Im_in,
                             dim_im_in,
                             ch_im_in,
                             wt,
                             ch_im_out,
                             dim_kernel,
                             padding,
                             stride,
                             bias,
                             bias_shift,
                             out_shift,
                             Im_out,
                             dim_im_out,
                             bufferA,
                             bufferB=None):
    """
    * The im2col converts the Q7 tensor input into Q15 column, which is stored in bufferA. 
    * There is reordering happenning during this im2col process with arm_q7_to_q15_reordered_no_shift. 
    * For every four elements, the second and third elements are swapped.

    * The computation kernel arm_nn_mat_mult_kernel_q7_q15_reordered does the GEMM computation with the reordered columns.
    
    * To speed-up the determination of the padding condition, we split the computation into 3x3 parts, 
    *   i.e., {top, mid, bottom} X {left, mid, right}.
    * This reduces the total number of boundary condition checks and improves the data copying performance.

    * Buffer size:
    *   bufferA size: 2*ch_im_in*dim_kernel*dim_kernel
    *   bufferB size: 0
    *
    * Input dimension constraints:
    *   ch_im_in is multiple of 4    ( because of the SIMD32 read and swap )
    *   ch_im_out is multiple of 2    ( bacause 2x2 mat_mult kernel )
    
    Args:
     * @brief Fast Q7 convolution function
     * @param[in]       Im_in       pointer to input tensor
     * @param[in]       dim_im_in   input tensor dimention
     * @param[in]       ch_im_in    number of input tensor channels
     * @param[in]       wt          pointer to kernel weights
     * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
     * @param[in]       dim_kernel  filter kernel size
     * @param[in]       padding     padding sizes
     * @param[in]       stride      convolution stride
     * @param[in]       bias        pointer to bias
     * @param[in]       bias_shift  amount of left-shift for bias
     * @param[in]       out_shift   amount of right-shift for output
     * @param[in,out]   Im_out      pointer to output tensor
     * @param[in]       dim_im_out  output tensor dimension
     * @param[in,out]   bufferA     pointer to buffer space for input
     * @param[in,out]   bufferB     pointer to buffer space for output
     * @return     The function returns either
    """
    return Template("""arm_convolve_HWC_q7_fast(${Im_in},
                            ${dim_im_in},
                            ${ch_im_in},
                            ${wt},
                            ${ch_im_out},
                            ${dim_kernel},
                            ${padding},
                            ${stride},
                            ${bias},
                            ${bias_shift},
                            ${out_shift},
                            ${Im_out},
                            ${dim_im_out},
                            (q15_t*)${bufferA},
                            ${bufferB});
""").substitute(Im_in=Im_in,
                dim_im_in=dim_im_in,
                ch_im_in=ch_im_in,
                wt=wt,
                ch_im_out=ch_im_out,
                dim_kernel=dim_kernel,
                padding=padding,
                stride=stride,
                bias=bias,
                bias_shift=bias_shift,
                out_shift=out_shift,
                Im_out=Im_out,
                dim_im_out=dim_im_out,
                bufferA=bufferA,
                bufferB=bufferB if bufferB else "NULL")

def arm_convolve_HWC_q7_basic(Im_in,
                              dim_im_in,
                              ch_im_in,
                              wt,
                              ch_im_out,
                              dim_kernel,
                              padding,
                              stride,
                              bias,
                              bias_shift,
                              out_shift,
                              Im_out,
                              dim_im_out,
                              bufferA,
                              bufferB=None):
    """
    * Buffer size:
    *   bufferA size: 2*ch_im_in*dim_kernel*dim_kernel
    *   bufferB size: 0
    * This basic version is designed to work for any input tensor and weight dimension.

    
    Args:
        * @brief Basic Q7 convolution function
        * @param[in]       Im_in       pointer to input tensor
        * @param[in]       dim_im_in   input tensor dimention
        * @param[in]       ch_im_in    number of input tensor channels
        * @param[in]       wt          pointer to kernel weights
        * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
        * @param[in]       dim_kernel  filter kernel size
        * @param[in]       padding     padding sizes
        * @param[in]       stride      convolution stride
        * @param[in]       bias        pointer to bias
        * @param[in]       bias_shift  amount of left-shift for bias
        * @param[in]       out_shift   amount of right-shift for output
        * @param[in,out]   Im_out      pointer to output tensor
        * @param[in]       dim_im_out  output tensor dimension
        * @param[in,out]   bufferA     pointer to buffer space for input
        * @param[in,out]   bufferB     pointer to buffer space for output
        * @return     The function returns either
    """
    return Template("""arm_convolve_HWC_q7_basic(${Im_in},
                            ${dim_im_in},
                            ${ch_im_in},
                            ${wt},
                            ${ch_im_out},
                            ${dim_kernel},
                            ${padding},
                            ${stride},
                            ${bias},
                            ${bias_shift},
                            ${out_shift},
                            ${Im_out},
                            ${dim_im_out},
                            (q15_t*)${bufferA},
                            ${bufferB});
                            """).substitute(Im_in=Im_in,
                            dim_im_in=dim_im_in,
                            ch_im_in=ch_im_in,
                            wt=wt,
                            ch_im_out=ch_im_out,
                            dim_kernel=dim_kernel,
                            padding=padding,
                            stride=stride,
                            bias=bias,
                            bias_shift=bias_shift,
                            out_shift=out_shift,
                            Im_out=Im_out,
                            dim_im_out=dim_im_out,
                            bufferA=bufferA,
                            bufferB=bufferB if bufferB else "NULL")

def arm_depthwise_separable_conv_HWC_q7(Im_in,
                                        dim_im_in,
                                        ch_im_in,
                                        wt,
                                        ch_im_out,
                                        dim_kernel,
                                        padding,
                                        stride,
                                        bias,
                                        bias_shift,
                                        out_shift,
                                        Im_out,
                                        dim_im_out,
                                        bufferA,
                                        bufferB=None):
    """       
        * Buffer size:
        * bufferA size: 2*ch_im_in*dim_kernel*dim_kernel
        * bufferB size: 0
        
        * ch_im_in equals ch_im_out
        
        * Implementation:
        * There are 3 nested loop here:
        * Inner loop: calculate each output value with MAC instruction over an accumulator
        * Mid   loop: loop over different output channel
        * Outer loop: loop over different output (x, y)
        
        * @brief Q7 depthwise separable convolution function
        * @param[in]       Im_in       pointer to input tensor
        * @param[in]       dim_im_in   input tensor dimension
        * @param[in]       ch_im_in    number of input tensor channels
        * @param[in]       wt          pointer to kernel weights
        * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
        * @param[in]       dim_kernel  filter kernel size
        * @param[in]       padding     padding sizes
        * @param[in]       stride      convolution stride
        * @param[in]       bias        pointer to bias
        * @param[in]       bias_shift  amount of left-shift for bias
        * @param[in]       out_shift   amount of right-shift for output
        * @param[in,out]   Im_out      pointer to output tensor
        * @param[in]       dim_im_out  output tensor dimension
        * @param[in,out]   bufferA     pointer to buffer space for input
        * @param[in,out]   bufferB     pointer to buffer space for output
        * @return     The function returns either
    """
    return Template("""arm_depthwise_separable_conv_HWC_q7(${Im_in},
                            ${dim_im_in},
                            ${ch_im_in},
                            ${wt},
                            ${ch_im_out},
                            ${dim_kernel},
                            ${padding},
                            ${stride},
                            ${bias},
                            ${bias_shift},
                            ${out_shift},
                            ${Im_out},
                            ${dim_im_out},
                            (q15_t*)${bufferA},
                            ${bufferB});
                            """).substitute(Im_in=Im_in,
                            dim_im_in=dim_im_in,
                            ch_im_in=ch_im_in,
                            wt=wt,
                            ch_im_out=ch_im_out,
                            dim_kernel=dim_kernel,
                            padding=padding,
                            stride=stride,
                            bias=bias,
                            bias_shift=bias_shift,
                            out_shift=out_shift,
                            Im_out=Im_out,
                            dim_im_out=dim_im_out,
                            bufferA=bufferA,
                            bufferB=bufferB if bufferB else "NULL")