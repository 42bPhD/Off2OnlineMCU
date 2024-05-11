#!/usr/bin/env bash


CMSIS_PATH="/e/2_Quantization/deployment-with-CMSIS-NN"

# NN and DSP source files
NN_SOURCES=$(find ${CMSIS_PATH}/CMSIS_5/CMSIS/NN/Source -name "*.c")
DSP_SOURCES="${CMSIS_PATH}/CMSIS_5/CMSIS/DSP/Source/StatisticsFunctions/arm_max_q7.c"
NN_H="${CMSIS_PATH}/cfiles/parameters.h"

gcc -g \
    -I "${CMSIS_PATH}/CMSIS_5/CMSIS/Core/Include" \
    -I "${CMSIS_PATH}/CMSIS_5/CMSIS/DSP/Include" \
    -I "${CMSIS_PATH}/CMSIS_5/CMSIS/NN/Include" \
    -D __ARM_ARCH_8M_BASE__ \
    $NN_SOURCES \
    $NN_H \
    "${CMSIS_PATH}/cfiles/main.c" -o main

#Execute main.exe
./main.exe