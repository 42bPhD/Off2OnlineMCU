from string import Template
"""
/* ----------------------------------------------------------------------
  * Project:      CMSIS NN Library
  * Title:        arm_relu_q7.c
  * Description:  Q7 version of ReLU
  * $Date:        20. July 2021
  * $Revision:    V.1.1.3
  * Target Processor:  Cortex-M cores
-------------------------------------------------------------------- */

"""

def arm_max_q7(pSrc, blockSize):
    """
    * @brief  Q7 vector maximum value
    * @param[in]       pSrc points to the input buffer
    * @param[in]       blockSize length of the input vector
    * @return maximum value
    *
    * <b>Scaling and Overflow Behavior:</b>
    * \par
    * The function uses a 32-bit internal accumulator.
    * The behavior of this function is undefined if the input is not stored in the same output type as the accumulator.
    * The function is an instance of q15_to_q7 (with additional accuracy loss).
    * The accumulator maintains full precision of the intermediate multiplication results but provides only a single guard bit.
    * There is no saturation on intermediate additions.
    * If the accumulator overflows it wraps around and distorts the result.
    * In order to avoid overflows completely the input signal must be scaled down by log2(blockSize) bits, as a total of blockSize additions are performed internally.
    * After division by blockSize the result has to be saturated to fit into q7 type.
    """
    return Template("""
        arm_max_q7(${pSrc}, ${blockSize});""").substitute(pSrc=pSrc,
                                                          blockSize=blockSize)

def arm_relu_q7(pSrc, blockSize):
    """
    * @brief  ReLU function for Q7 type
    * @param[in]  pSrc       points to the input buffer
    * @param[in]  blockSize  number of samples in the input buffer
    * @return none
    
    """
    return Template("""
        arm_relu_q7(${pSrc}, ${blockSize});""").substitute(pSrc=pSrc, blockSize=blockSize)

def arm_relu6_s8(data, size):
    return Template("""
        arm_relu6_s8(${data}, ${size});""").substitute(data=data, size=size)