"""
/* ----------------------------------------------------------------------
  * Project:      CMSIS NN Library
  * Title:        arm_pool_q7_HWC.c
  * Description:  Pooling function implementations

  * $Date:        20. July 2021
  * $Revision:    V.1.1.1

  * Target Processor:  Cortex-M cores
 -------------------------------------------------------------------- */
"""

from string import Template

def arm_maxpool_q7_HWC(Im_in,
                       dim_im_in,
                       ch_im_in,
                       dim_kernel,
                       padding,
                       stride,
                       dim_im_out,
                       bufferA,
                       Im_out):
    """
        * @brief Q7 max pooling function
        * The pooling function is implemented as split x-pooling then y-pooling.
        
        * This pooling function is input-destructive. 
        * Input data is undefined after calling this function.
        
        Args:
        * @param[in, out]  Im_in       pointer to input tensor
        * @param[in]       dim_im_in   input tensor dimention
        * @param[in]       ch_im_in    number of input tensor channels
        * @param[in]       dim_kernel  filter kernel size
        * @param[in]       padding     padding sizes
        * @param[in]       stride      convolution stride
        * @param[in]       dim_im_out  output tensor dimension
        * @param[in,out]   bufferA     Not used
        * @param[in,out]   Im_out      pointer to output tensor

    """
    return Template("""
        arm_maxpool_q7_HWC(${Im_in}, 
                            ${dim_im_in}, 
                            ${ch_im_in}, 
                            ${dim_kernel}, 
                            ${padding}, 
                            ${stride}, 
                            ${dim_im_out}, 
                            ${bufferA}, 
                            ${Im_out});""").substitute(Im_in=Im_in,
                                                    dim_im_in=dim_im_in,
                                                    ch_im_in=ch_im_in,
                                                    dim_kernel=dim_kernel,
                                                    padding=padding,
                                                    stride=stride,
                                                    dim_im_out=dim_im_out,
                                                    bufferA=bufferA,
                                                    Im_out=Im_out)

def arm_avepool_q7_HWC(Im_in,
                       dim_im_in,
                       ch_im_in,
                       dim_kernel,
                       padding,
                       stride,
                       dim_im_out,
                       bufferA,
                       Im_out):
    """
        * @brief Q7 average pooling function
        * Buffer size:
 
        * bufferA size:  2*dim_im_out*ch_im_in
        * The pooling function is implemented as split x-pooling then y-pooling.
        
        * This pooling function is input-destructive. Input data is undefined
        * after calling this function.
        
        Args:
        * @param[in, out]  Im_in       pointer to input tensor
        * @param[in]       dim_im_in   input tensor dimention
        * @param[in]       ch_im_in    number of input tensor channels
        * @param[in]       dim_kernel  filter kernel size
        * @param[in]       padding     padding sizes
        * @param[in]       stride      convolution stride
        * @param[in]       dim_im_out  output tensor dimension
        * @param[in,out]   bufferA     Not used
        * @param[in,out]   Im_out      pointer to output tensor

    """
    return Template("""
        arm_avepool_q7_HWC(${Im_in}, 
                            ${dim_im_in}, 
                            ${ch_im_in}, 
                            ${dim_kernel}, 
                            ${padding}, 
                            ${stride}, 
                            ${dim_im_out}, 
                            ${bufferA}, 
                            ${Im_out});""").substitute(Im_in=Im_in,
                                                    dim_im_in=dim_im_in,
                                                    ch_im_in=ch_im_in,
                                                    dim_kernel=dim_kernel,
                                                    padding=padding,
                                                    stride=stride,
                                                    dim_im_out=dim_im_out,
                                                    bufferA=bufferA,
                                                    Im_out=Im_out)