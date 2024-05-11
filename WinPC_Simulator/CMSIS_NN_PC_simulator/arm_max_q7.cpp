#include <stdlib.h>
#include "modules.h"

void arm_max_q7(
    const q7_t* pSrc,
    uint32_t blockSize,
    q7_t* pResult,
    uint32_t* pIndex)
{
    q7_t maxVal, out;                              /* Temporary variables to store the output value. */
    uint32_t blkCnt, outIndex;                     /* Loop counter */

    /* Initialise index value to zero. */
    outIndex = 0U;
    /* Load first input value that act as reference value for comparision */
    out = *pSrc++;

    /* Initialize blkCnt with number of samples */
    blkCnt = (blockSize - 1U);

    while (blkCnt > 0U)
    {
        /* Initialize maxVal to the next consecutive values one by one */
        maxVal = *pSrc++;

        /* compare for the maximum value */
        if (out < maxVal)
        {
            /* Update the maximum value and it's index */
            out = maxVal;
            outIndex = blockSize - blkCnt;
        }

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Store the maximum value and it's index into destination pointers */
    *pResult = out;
    *pIndex = outIndex;
}