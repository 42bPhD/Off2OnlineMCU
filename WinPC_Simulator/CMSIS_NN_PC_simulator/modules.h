#pragma once
#ifndef __CMSIS_H__
#define __CMSIS_H__
#include <iostream>
#include <chrono>
#include <cstring>
typedef int16_t q15_t;
typedef int8_t q7_t;
typedef int32_t q31_t;
typedef int64_t q63_t;
typedef float float32_t;

typedef struct {
    union {
        uint32_t f0r3r;
        uint16_t f0r_f3r[2];
        q7_t f03[4];
    };
    union {
        uint32_t f2b5b;
        uint16_t f2b_f5b[2];
        q7_t f25[4];
    };
    union {
        uint32_t f6r9r;
        uint16_t f6r_f9r[2];
        q7_t f69[4];
    };
    union {
        uint32_t f8b11b;
        uint16_t f8b_f11b[2];
        q7_t f811[4];
    };
    union {
        uint32_t f147_10g;
        q7_t f1_4_7_10g[4];
    };

}nn_pack_t;


//#define NN_ROUND(out_shift) ((0x1 << out_shift) >> 1)
#define NN_ROUND(out_shift) 0

#define __PKHBT(ARG1,ARG2,ARG3)          ( ((((uint32_t)(ARG1))          ) & 0x0000FFFFUL) |  \
                                           ((((uint32_t)(ARG2)) << (ARG3)) & 0xFFFF0000UL)  )

#define __PKHTB(ARG1,ARG2,ARG3)          ( ((((uint32_t)(ARG1))          ) & 0xFFFF0000UL) |  \
                                           ((((uint32_t)(ARG2)) >> (ARG3)) & 0x0000FFFFUL)  )
#define __SIMD32_TYPE int32_t
#define __SIMD32(addr)        (*(__SIMD32_TYPE **) & (addr))
#define __SIMD32_CONST(addr)  ( (__SIMD32_TYPE * )   (addr))
#define _SIMD32_OFFSET(addr)  (*(__SIMD32_TYPE * )   (addr))
#define __SIMD64(addr)        (*(      int64_t **) & (addr))
#define INDEX_MASK         0x0000003F
/* SIMD replacement */

static q31_t read_q7x4(
    q7_t const* pQ7)
{
    q31_t val;
    memcpy(&val, pQ7, 4);
    return (val);
}

/**
  @brief         Read 4 Q7 from Q7 pointer and increment pointer afterwards.
  @param[in]     pQ7       points to input value
  @return        Q31 value
 */
#define read_q7x4_ia(pQ7) read_q7x4((*(pQ7) += 4) - 4)

 /**
   @brief         Read 2 Q15 from Q15 pointer.
   @param[in]     pQ15      points to input value
   @return        Q31 value
  */
static q31_t read_q15x2(
    q15_t* pQ15)
{
    q31_t val;

    memcpy(&val, pQ15, 4);

    return (val);
}

/**
  @brief         Read 2 Q15 from Q15 pointer and increment pointer afterwards.
  @param[in]     pQ15      points to input value
  @return        Q31 value
 */
static q31_t read_q15x2_ia(
    q15_t** pQ15)
{
    q31_t val;

    memcpy(&val, *pQ15, 4);
    *pQ15 += 2;

    return (val);
}

/**
  @brief         Read 2 Q15 from Q15 pointer and decrement pointer afterwards.
  @param[in]     pQ15      points to input value
  @return        Q31 value
 */
static q31_t read_q15x2_da(
    q15_t** pQ15)
{
    q31_t val;

    memcpy(&val, *pQ15, 4);
    *pQ15 -= 2;

    return (val);
}

/**
  @brief         Write 2 Q15 to Q15 pointer and increment pointer afterwards.
  @param[in]     pQ15      points to input value
  @param[in]     value     Q31 value
  @return        none
 */
static void write_q15x2_ia(
    q15_t** pQ15,
    q31_t    value)
{
    q31_t val = value;

    memcpy(*pQ15, &val, 4);
    *pQ15 += 2;
}

/**
  @brief         Write 2 Q15 to Q15 pointer.
  @param[in]     pQ15      points to input value
  @param[in]     value     Q31 value
  @return        none
 */
static void write_q15x2(
    q15_t* pQ15,
    q31_t   value)
{
    q31_t val = value;

    memcpy(pQ15, &val, 4);
}



/**
  @brief         Read 4 Q7 from Q7 pointer and decrement pointer afterwards.
  @param[in]     pQ7       points to input value
  @return        Q31 value
 */
static q31_t read_q7x4_da(
    q7_t** pQ7)
{
    q31_t val;

    memcpy(&val, *pQ7, 4);
    *pQ7 -= 4;

    return (val);
}

/**
  @brief         Write 4 Q7 to Q7 pointer and increment pointer afterwards.
  @param[in]     pQ7       points to input value
  @param[in]     value     Q31 value
  @return        none
 */
static void write_q7x4_ia(
    q7_t** pQ7,
    q31_t   value)
{
    q31_t val = value;

    memcpy(*pQ7, &val, 4);
    *pQ7 += 4;
}

/*

Normally those kind of definitions are in a compiler file
in Core or Core_A.

But for MSVC compiler it is a bit special. The goal is very specific
to CMSIS-DSP and only to allow the use of this library from other
systems like Python or Matlab.

MSVC is not going to be used to cross-compile to ARM. So, having a MSVC
compiler file in Core or Core_A would not make sense.

*/

static uint8_t __CLZ(uint32_t data)
{
    if (data == 0U) { return 32U; }

    uint32_t count = 0U;
    uint32_t mask = 0x80000000U;

    while ((data & mask) == 0U)
    {
        count += 1U;
        mask = mask >> 1U;
    }
    return count;
}

static int32_t __SSAT(int32_t val, uint32_t sat)
{
    if ((sat >= 1U) && (sat <= 32U))
    {
        const int32_t max = (int32_t)((1U << (sat - 1U)) - 1U);
        const int32_t min = -1 - max;
        if (val > max)
        {
            return max;
        }
        else if (val < min)
        {
            return min;
        }
    }
    return val;
}
//static int __SSAT(int a, const int bit) {
//    int clip_value = a;
//    if (a > (0x1 << (bit - 1)) - 1) {
//        clip_value = (0x1 << (bit - 1)) - 1; //127
//    }
//    else if (a < -(0x1 << (bit - 1))) {
//        clip_value = -(0x1 << (bit - 1)); //-128
//    }
//    return clip_value;
//};

static uint32_t __USAT(int32_t val, uint32_t sat)
{
    if (sat <= 31U)
    {
        const uint32_t max = ((1U << sat) - 1U);
        if (val > (int32_t)max)
        {
            return max;
        }
        else if (val < 0)
        {
            return 0U;
        }
    }
    return (uint32_t)val;
}
/**
  @brief         Read 4 q7 from q7 pointer and post increment pointer.
  @param[in]     in_q7       Pointer to pointer that holds address of input.
  @return        q31 value
 */
static q31_t arm_nn_read_q7x4_ia(const q7_t** in_q7)
{
    q31_t val = 0;
    memcpy(&val, *in_q7, 4);
    *in_q7 += 4;

    return (val);
}


/**
 * @brief Clips Q63 to Q31 values.
 */
static q31_t clip_q63_to_q31(
    q63_t x)
{
    return ((q31_t)(x >> 32) != ((q31_t)x >> 31)) ?
        ((0x7FFFFFFF ^ ((q31_t)(x >> 63)))) : (q31_t)x;
}

/**
 * @brief Clips Q63 to Q15 values.
 */
static q15_t clip_q63_to_q15(
    q63_t x)
{
    return ((q31_t)(x >> 32) != ((q31_t)x >> 31)) ?
        ((0x7FFF ^ ((q15_t)(x >> 63)))) : (q15_t)(x >> 15);
}

/**
 * @brief Clips Q31 to Q7 values.
 */
static q7_t clip_q31_to_q7(
    q31_t x)
{
    return ((q31_t)(x >> 24) != ((q31_t)x >> 23)) ?
        ((0x7F ^ ((q7_t)(x >> 31)))) : (q7_t)x;
}

/**
 * @brief Clips Q31 to Q15 values.
 */
static q15_t clip_q31_to_q15(
    q31_t x)
{
    return ((q31_t)(x >> 16) != ((q31_t)x >> 15)) ?
        ((0x7FFF ^ ((q15_t)(x >> 31)))) : (q15_t)x;
}

/**
 * @brief Multiplies 32 X 64 and returns 32 bit result in 2.30 format.
 */
static q63_t mult32x64(
    q63_t x,
    q31_t y)
{
    return ((((q63_t)(x & 0x00000000FFFFFFFF) * y) >> 32) +
        (((q63_t)(x >> 32) * y)));
}

/**
 * @brief Function to Calculates 1/in (reciprocal) value of Q31 Data type.
 */
static uint32_t arm_recip_q31(
    q31_t in,
    q31_t* dst,
    const q31_t* pRecipTable)
{
    q31_t out;
    uint32_t tempVal;
    uint32_t index, i;
    uint32_t signBits;

    if (in > 0)
    {
        signBits = ((uint32_t)(__CLZ(in) - 1));
    }
    else
    {
        signBits = ((uint32_t)(__CLZ(-in) - 1));
    }

    /* Convert input sample to 1.31 format */
    in = (in << signBits);

    /* calculation of index for initial approximated Val */
    index = (uint32_t)(in >> 24);
    index = (index & INDEX_MASK);

    /* 1.31 with exp 1 */
    out = pRecipTable[index];

    /* calculation of reciprocal value */
    /* running approximation for two iterations */
    for (i = 0U; i < 2U; i++)
    {
        tempVal = (uint32_t)(((q63_t)in * out) >> 31);
        tempVal = 0x7FFFFFFFu - tempVal;
        /*      1.31 with exp 1 */
        /* out = (q31_t) (((q63_t) out * tempVal) >> 30); */
        out = clip_q63_to_q31(((q63_t)out * tempVal) >> 30);
    }

    /* write output */
    *dst = out;

    /* return num of signbits of out = 1/in value */
    return (signBits + 1U);
}


/**
 * @brief Function to Calculates 1/in (reciprocal) value of Q15 Data type.
 */
static uint32_t arm_recip_q15(
    q15_t in,
    q15_t* dst,
    const q15_t* pRecipTable)
{
    q15_t out = 0;
    uint32_t tempVal = 0;
    uint32_t index = 0, i = 0;
    uint32_t signBits = 0;

    if (in > 0)
    {
        signBits = ((uint32_t)(__CLZ(in) - 17));
    }
    else
    {
        signBits = ((uint32_t)(__CLZ(-in) - 17));
    }

    /* Convert input sample to 1.15 format */
    in = (in << signBits);

    /* calculation of index for initial approximated Val */
    index = (uint32_t)(in >> 8);
    index = (index & INDEX_MASK);

    /*      1.15 with exp 1  */
    out = pRecipTable[index];

    /* calculation of reciprocal value */
    /* running approximation for two iterations */
    for (i = 0U; i < 2U; i++)
    {
        tempVal = (uint32_t)(((q31_t)in * out) >> 15);
        tempVal = 0x7FFFu - tempVal;
        /*      1.15 with exp 1 */
        out = (q15_t)(((q31_t)out * tempVal) >> 14);
        /* out = clip_q31_to_q15(((q31_t) out * tempVal) >> 14); */
    }

    /* write output */
    *dst = out;

    /* return num of signbits of out = 1/in value */
    return (signBits + 1);
}

/**
 * @brief Integer exponentiation
 * @param[in]    x           value
 * @param[in]    nb          integer exponent >= 1
 * @return x^nb
 *
 */
static inline float32_t arm_exponent_f32(float32_t x, int nb)
{
    float32_t r = x;
    nb--;
    while (nb > 0)
    {
        r = r * x;
        nb--;
    }
    return(r);
}

/*
 * @brief C custom defined intrinsic functions
 */


 /*
  * @brief C custom defined QADD8
  */
static uint32_t __QADD8(
    uint32_t x,
    uint32_t y)
{
    q31_t r, s, t, u;

    r = __SSAT(((((q31_t)x << 24) >> 24) + (((q31_t)y << 24) >> 24)), 8) & (int32_t)0x000000FF;
    s = __SSAT(((((q31_t)x << 16) >> 24) + (((q31_t)y << 16) >> 24)), 8) & (int32_t)0x000000FF;
    t = __SSAT(((((q31_t)x << 8) >> 24) + (((q31_t)y << 8) >> 24)), 8) & (int32_t)0x000000FF;
    u = __SSAT(((((q31_t)x) >> 24) + (((q31_t)y) >> 24)), 8) & (int32_t)0x000000FF;

    return ((uint32_t)((u << 24) | (t << 16) | (s << 8) | (r)));
}

/*
 * @brief C custom defined QSUB8
 */
static uint32_t __QSUB8(
    uint32_t x,
    uint32_t y)
{
    q31_t r, s, t, u;

    r = __SSAT(((((q31_t)x << 24) >> 24) - (((q31_t)y << 24) >> 24)), 8) & (int32_t)0x000000FF;
    s = __SSAT(((((q31_t)x << 16) >> 24) - (((q31_t)y << 16) >> 24)), 8) & (int32_t)0x000000FF;
    t = __SSAT(((((q31_t)x << 8) >> 24) - (((q31_t)y << 8) >> 24)), 8) & (int32_t)0x000000FF;
    u = __SSAT(((((q31_t)x) >> 24) - (((q31_t)y) >> 24)), 8) & (int32_t)0x000000FF;

    return ((uint32_t)((u << 24) | (t << 16) | (s << 8) | (r)));
}


/*
 * @brief C custom defined QADD16
 */
static uint32_t __QADD16(
    uint32_t x,
    uint32_t y)
{
    /*  q31_t r,     s;  without initialisation 'arm_offset_q15 test' fails  but 'intrinsic' tests pass! for armCC */
    q31_t r = 0, s = 0;

    r = __SSAT(((((q31_t)x << 16) >> 16) + (((q31_t)y << 16) >> 16)), 16) & (int32_t)0x0000FFFF;
    s = __SSAT(((((q31_t)x) >> 16) + (((q31_t)y) >> 16)), 16) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r)));
}


/*
 * @brief C custom defined SHADD16
 */
static uint32_t __SHADD16(
    uint32_t x,
    uint32_t y)
{
    q31_t r, s;

    r = (((((q31_t)x << 16) >> 16) + (((q31_t)y << 16) >> 16)) >> 1) & (int32_t)0x0000FFFF;
    s = (((((q31_t)x) >> 16) + (((q31_t)y) >> 16)) >> 1) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r)));
}


/*
 * @brief C custom defined QSUB16
 */
static uint32_t __QSUB16(
    uint32_t x,
    uint32_t y)
{
    q31_t r, s;

    r = __SSAT(((((q31_t)x << 16) >> 16) - (((q31_t)y << 16) >> 16)), 16) & (int32_t)0x0000FFFF;
    s = __SSAT(((((q31_t)x) >> 16) - (((q31_t)y) >> 16)), 16) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r)));
}


/*
 * @brief C custom defined SHSUB16
 */
static uint32_t __SHSUB16(
    uint32_t x,
    uint32_t y)
{
    q31_t r, s;

    r = (((((q31_t)x << 16) >> 16) - (((q31_t)y << 16) >> 16)) >> 1) & (int32_t)0x0000FFFF;
    s = (((((q31_t)x) >> 16) - (((q31_t)y) >> 16)) >> 1) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r)));
}


/*
 * @brief C custom defined QASX
 */
static uint32_t __QASX(
    uint32_t x,
    uint32_t y)
{
    q31_t r, s;

    r = __SSAT(((((q31_t)x << 16) >> 16) - (((q31_t)y) >> 16)), 16) & (int32_t)0x0000FFFF;
    s = __SSAT(((((q31_t)x) >> 16) + (((q31_t)y << 16) >> 16)), 16) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r)));
}


/*
 * @brief C custom defined SHASX
 */
static uint32_t __SHASX(
    uint32_t x,
    uint32_t y)
{
    q31_t r, s;

    r = (((((q31_t)x << 16) >> 16) - (((q31_t)y) >> 16)) >> 1) & (int32_t)0x0000FFFF;
    s = (((((q31_t)x) >> 16) + (((q31_t)y << 16) >> 16)) >> 1) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r)));
}


/*
 * @brief C custom defined QSAX
 */
static uint32_t __QSAX(
    uint32_t x,
    uint32_t y)
{
    q31_t r, s;

    r = __SSAT(((((q31_t)x << 16) >> 16) + (((q31_t)y) >> 16)), 16) & (int32_t)0x0000FFFF;
    s = __SSAT(((((q31_t)x) >> 16) - (((q31_t)y << 16) >> 16)), 16) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r)));
}


/*
 * @brief C custom defined SHSAX
 */
static uint32_t __SHSAX(
    uint32_t x,
    uint32_t y)
{
    q31_t r, s;

    r = (((((q31_t)x << 16) >> 16) + (((q31_t)y) >> 16)) >> 1) & (int32_t)0x0000FFFF;
    s = (((((q31_t)x) >> 16) - (((q31_t)y << 16) >> 16)) >> 1) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r)));
}


/*
 * @brief C custom defined SMUSDX
 */
static uint32_t __SMUSDX(
    uint32_t x,
    uint32_t y)
{
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y) >> 16)) -
        ((((q31_t)x) >> 16) * (((q31_t)y << 16) >> 16))));
}

/*
 * @brief C custom defined SMUADX
 */
static uint32_t __SMUADX(
    uint32_t x,
    uint32_t y)
{
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y) >> 16)) +
        ((((q31_t)x) >> 16) * (((q31_t)y << 16) >> 16))));
}


/*
 * @brief C custom defined QADD
 */
static int32_t __QADD(
    int32_t x,
    int32_t y)
{
    return ((int32_t)(clip_q63_to_q31((q63_t)x + (q31_t)y)));
}


/*
 * @brief C custom defined QSUB
 */
static int32_t __QSUB(
    int32_t x,
    int32_t y)
{
    return ((int32_t)(clip_q63_to_q31((q63_t)x - (q31_t)y)));
}


/*
 * @brief C custom defined SMLAD
 */
static uint32_t __SMLAD(
    uint32_t x,
    uint32_t y,
    uint32_t sum)
{
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) +
        ((((q31_t)x) >> 16) * (((q31_t)y) >> 16)) +
        (((q31_t)sum))));
}


/*
 * @brief C custom defined SMLADX
 */
static uint32_t __SMLADX(
    uint32_t x,
    uint32_t y,
    uint32_t sum)
{
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y) >> 16)) +
        ((((q31_t)x) >> 16) * (((q31_t)y << 16) >> 16)) +
        (((q31_t)sum))));
}


/*
 * @brief C custom defined SMLSDX
 */
static uint32_t __SMLSDX(
    uint32_t x,
    uint32_t y,
    uint32_t sum)
{
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y) >> 16)) -
        ((((q31_t)x) >> 16) * (((q31_t)y << 16) >> 16)) +
        (((q31_t)sum))));
}


/*
 * @brief C custom defined SMLALD
 */
static uint64_t __SMLALD(
    uint32_t x,
    uint32_t y,
    uint64_t sum)
{
    /*  return (sum + ((q15_t) (x >> 16) * (q15_t) (y >> 16)) + ((q15_t) x * (q15_t) y)); */
    return ((uint64_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) +
        ((((q31_t)x) >> 16) * (((q31_t)y) >> 16)) +
        (((q63_t)sum))));
}


/*
 * @brief C custom defined SMLALDX
 */
static uint64_t __SMLALDX(
    uint32_t x,
    uint32_t y,
    uint64_t sum)
{
    /*  return (sum + ((q15_t) (x >> 16) * (q15_t) y)) + ((q15_t) x * (q15_t) (y >> 16)); */
    return ((uint64_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y) >> 16)) +
        ((((q31_t)x) >> 16) * (((q31_t)y << 16) >> 16)) +
        (((q63_t)sum))));
}


/*
 * @brief C custom defined SMUAD
 */
static uint32_t __SMUAD(
    uint32_t x,
    uint32_t y)
{
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) +
        ((((q31_t)x) >> 16) * (((q31_t)y) >> 16))));
}


/*
 * @brief C custom defined SMUSD
 */
static uint32_t __SMUSD(
    uint32_t x,
    uint32_t y)
{
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) -
        ((((q31_t)x) >> 16) * (((q31_t)y) >> 16))));
}


/*
 * @brief C custom defined SXTB16
 */
static uint32_t __SXTB16(
    uint32_t x)
{
    return ((uint32_t)(((((q31_t)x << 24) >> 24) & (q31_t)0x0000FFFF) |
        ((((q31_t)x << 8) >> 8) & (q31_t)0xFFFF0000)));
}
static uint32_t __ROR(uint32_t op1, uint32_t op2)
{
    op2 %= 32U;
    if (op2 == 0U)
    {
        return op1;
    }
    return (op1 >> op2) | (op1 << (32U - op2));
}
static uint32_t __SXTB16_RORn(
    uint32_t x, uint32_t rotate)
{

    uint32_t result = __ROR(x, rotate);
    //
    result = __SXTB16(result);
    return result;
}
/*
 * @brief C custom defined SMMLA
 */
static int32_t __SMMLA(
    int32_t x,
    int32_t y,
    int32_t sum)
{
    return (sum + (int32_t)(((int64_t)x * y) >> 32));
}
/**
 * @brief read and expand one q7 word into two q15 words
 */
static q31_t arm_nn_read_q15x2_ia(q15_t** in_q15)
{
    q31_t val;

    memcpy(&val, *in_q15, 4);
    *in_q15 += 2;

    return (val);
}
static const q7_t* read_and_pad(const q7_t* source, q31_t* out1, q31_t* out2)
{
    q31_t inA = arm_nn_read_q7x4_ia(&source);
    q31_t inAbuf1 = __SXTB16_RORn((uint32_t)inA, 8);
    q31_t inAbuf2 = __SXTB16(inA);

    *out2 = (int32_t)(__PKHTB(inAbuf1, inAbuf2, 16));
    *out1 = (int32_t)(__PKHBT(inAbuf2, inAbuf1, 16));

    return source;
}

static uint16_t arm_nn_read_q7x2_ia(uint8_t** in_q7)
{
    uint16_t val = 0;

    memcpy(&val, *in_q7, 2);
    *in_q7 += 2;

    return (val);
}


static uint32_t read_U8x(
    uint8_t** pU8, int size)
{
    uint32_t val = 0;
    memcpy(&val, *pU8, size);
    *pU8 += size;

    return (val);
}
static uint32_t read_q7x3(
    q7_t* pQ7)
{
    uint32_t val = 0;
    memcpy(&val, pQ7, 3);

    return (val);
}
static uint32_t arm_nn_read_q7x3_ia(const q7_t** in_q7)
{
    uint32_t val = 0;
    memcpy(&val, *in_q7, 3);
    *in_q7 += 3;

    return (val);
}
static const q7_t* read_q7x3_packing(const q7_t* source,
    uint32_t* out1, // rg
    q7_t* out2) // b
{
    uint32_t inA = arm_nn_read_q7x3_ia(&source);
    *out1 = __SXTB16(inA); // 3(24), 1(-2)   //rb
    *out2 = (q7_t)__SXTB16_RORn(inA, 8);
    return source;
}
static const q7_t* read_q7x4_ia_pack(const q7_t* source, q31_t* out1, q31_t* out2)
{
    q31_t inA = arm_nn_read_q7x4_ia(&source);
    *out2 = __SXTB16_RORn((uint32_t)inA, 8);
    *out1 = __SXTB16(inA);

    return source;
}

//static const q7_t* read_q7x3_packing(const q7_t* source,
//    uint32_t* out1, // rb
//    uint32_t* out2)// g
//{
//    uint32_t inA = arm_nn_read_q7x3_ia(&source);
//
//   
//    *out1 = __SXTB16(inA); // 3(24), 1(-2)   //rb
//    *out2 = (q7_t)__SXTB16_RORn(inA, 8) & 0x0000FFFF;
//    return source;
//}



static int validate_s8(q7_t* act, const q7_t* ref, int size)
{
    int test_passed = true;
    int count = 0;
    int total = 0;

    for (int i = 0; i < size; ++i)
    {
        total++;
        if (act[i] != ref[i])
        {
            count++;
            printf("ERROR at pos %d: Act: %d Ref: %d\r\n", i, act[i], ref[i]);
            //printf("Error index[%d] = %d\n", i, act[i] - ref[i]);
            test_passed = false;
        }
        else {
            //printf("Correct value at pos %d: Act: %d Ref: %d\r\n", i, act[i], ref[i]);
        }
    }

    if (!test_passed)
    {
        printf("%d of %d failed\r\n", count, total);
    }
    return test_passed;
}


#endif // __CMSIS_H__
