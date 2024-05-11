#include "arm_math_types.h"
#include "arm_nnfunctions.h"

void arm_q7_to_q15_with_offset(const q7_t* src, q15_t* dst, uint32_t block_size, q15_t offset)
{
    int block_cnt;

#if defined(ARM_MATH_MVEI)

    int16x8_t source;
    const int16x8_t source_offset = vdupq_n_s16(offset);
    block_cnt = block_size / 8;

    while (block_cnt > 0)
    {
        source = vldrbq_s16(src);
        source = vaddq_s16(source, source_offset);
        vstrhq_s16(dst, source);
        dst += 8;
        src += 8;
        block_cnt--;
    }

    block_cnt = block_size & 0x7;

#elif defined(ARM_MATH_DSP)
    /* Run the below code for cores that support SIMD instructions  */
    q31_t in_q7x4;
    q31_t in_q15x2_1;
    q31_t in_q15x2_2;
    q31_t out_q15x2_1;
    q31_t out_q15x2_2;

    /*loop unrolling */
    block_cnt = block_size >> 2;

    /* First part of the processing with loop unrolling.  Compute 4 outputs at a time. */
    const q31_t offset_q15x2 = __PKHBT(offset, offset, 16);
    while (block_cnt > 0)
    {
        /* convert from q7 to q15 and then store the results in the destination buffer */
        in_q7x4 = arm_nn_read_q7x4_ia(&src);

        /* Extract and sign extend each of the four q7 values to q15 */
        in_q15x2_1 = __SXTAB16(offset_q15x2, __ROR(in_q7x4, 8));
        in_q15x2_2 = __SXTAB16(offset_q15x2, in_q7x4);

        out_q15x2_2 = __PKHTB(in_q15x2_1, in_q15x2_2, 16);
        out_q15x2_1 = __PKHBT(in_q15x2_2, in_q15x2_1, 16);

        arm_nn_write_q15x2_ia(&dst, out_q15x2_1);
        arm_nn_write_q15x2_ia(&dst, out_q15x2_2);

        block_cnt--;
    }
    /* Handle left over samples */
    block_cnt = block_size % 0x4;

#else
    /* Run the below code for Cortex-M0 */
    /* Loop over block_size number of values */
    block_cnt = block_size;
#endif

    while (block_cnt > 0)
    {
        *dst++ = (q15_t)*src++ + offset;

        /* Decrement the loop counter */
        block_cnt--;
    }
}