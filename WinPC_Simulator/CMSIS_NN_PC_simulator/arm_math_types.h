#ifndef _ARM_MATH_TYPES_H_
#define _ARM_MATH_TYPES_H_

typedef enum
{
    ARM_MATH_SUCCESS = 0,        /**< No error */
    ARM_MATH_ARGUMENT_ERROR = -1,        /**< One or more arguments are incorrect */
    ARM_MATH_LENGTH_ERROR = -2,        /**< Length of data buffer is incorrect */
    ARM_MATH_SIZE_MISMATCH = -3,        /**< Size of matrices is not compatible with the operation */
    ARM_MATH_NANINF = -4,        /**< Not-a-number (NaN) or infinity is generated */
    ARM_MATH_SINGULAR = -5,        /**< Input matrix is singular and cannot be inverted */
    ARM_MATH_TEST_FAILURE = -6,        /**< Test Failed */
    ARM_MATH_DECOMPOSITION_FAILURE = -7         /**< Decomposition Failed */
}arm_status;

#endif