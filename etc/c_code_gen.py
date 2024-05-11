import pickle
import torch


import torch
import numpy as np
from torch import nn
from torch.functional import F
from collections import defaultdict
import os
from pprint import pprint

def ddict():
    return defaultdict(ddict)

def list_bracket(lists):
    return "{" + ', '.join(map(str, lists)) + "}"

if __name__ == '__main__':
    with open('cfiles/q_params.pkl', 'rb') as f:
        q_params = pickle.load(f)
    with open('cfiles/params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    from models.model_q import MCU_VGGRep, MCU_VGGRepC1
    
    # weights = MCU_VGGRep()
    weights = MCU_VGGRepC1()
    weights.load_state_dict(torch.load('./weights/mcu_vggrepc1.pth'))
    print(weights)
    for name, param in weights.named_modules():
        if 'GAP' in name:
            print(weights.get_shape(1, (3, 96, 96)))
            print(name, param)

    # exit()
    # print(weights)
    
    previous_module = ddict()
    current_module = ddict()
    
    current_module['IM_IN'] = 'input_data'
    current_module['BUFFER'] = 'buffer1'
    tmp_out_variable_buffer = ""
    code_buffer = "#pragma once"
    # code_buffer += 'void setup() {\n'
    # code_buffer += '    Serial.begin(115200);\n'
    # code_buffer += '    SCB_EnableICache();\n'
    # code_buffer += '    SCB_EnableDCache();\n'
    # code_buffer += '}\n'
    code_buffer += """
#include "nn.h"
#include "arm_nnfunctions.h"
#include <iostream>
#include <fstream>
#include <memory>

%s
void save(const char* file, q7_t* out, size_t sz)
{
    std::ofstream fp(file, std::ios::binary);
    fp.write(reinterpret_cast<char*>(out), sz);
    fp.close();
    //std::cout << "Saved " << file << std::endl;
}
q7_t* run_nn(q7_t* input_data, q7_t* output_data, q7_t* buffer1, q7_t* buffer2, q7_t* col_buffer, q7_t* fc_buffer) {
"""
    
    variable_buffer = """
#pragma once
#include "arm_nnfunctions.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "inputs.h"
#include "parameters.h"
#include "weights.h"
q7_t* run_nn(q7_t* input_data, q7_t* output_data, q7_t* buffer1, q7_t* buffer2,
    q7_t* col_buffer, q7_t* fc_buffer);    
"""
    
    # variable_buffer += '#include "modules.h"\n'
    vs_prj_path = "e:/2_Quantization/deployment-with-CMSIS-NN/CMSIS_NN_PC_simulator/Deploy_Simulator/logs"
    variable_buffer += 'static q7_t scratch_buffer[MAX_CONV_BUFFER_SIZE*4];\n'
    for idx, (name, param) in enumerate(weights.named_modules()):
        if idx == 0:
            current_module['IM_IN'] = 'input_data'
            continue
        else:
            current_module['BUFFER'] = f'{name.lower()}_out'
        # if idx == len(dict(weights.named_modules())) - 1:
        #     # current_module['BUFFER'] = 'output_data'
        #     current_module['BUFFER'] = name.lower() + '_out'
        
        if isinstance(param, nn.Conv2d):
            # print(param.weight.shape)
            # print(param.bias.shape)
            WEIGHT = f'{name.lower()}_wt'
            BIAS = f'{name.lower()}_bias'
            
            variable_buffer += f"static q7_t {WEIGHT}[{name}_WT_SHAPE] = {name}_WT;\n"
            variable_buffer += f"static q7_t {BIAS}[{name}_BIAS_SHAPE] = {name}_BIAS;\n"
            KERNEL_SIZE_H, KERNEL_SIZE_W = (f'{name}_KER_DIM', f'{name}_KER_DIM') if param.kernel_size[0] == param.kernel_size[1] else (f'{name}_KER_DIM_X', f'{name}_KER_DIM_Y')
            STRIDE_H, STRIDE_W = (f'{name}_STRIDE', f'{name}_STRIDE') if param.stride[0] == param.stride[1] else (f'{name}_STRIDE_X', f'{name}_STRIDE_Y')
            PADDING_H, PADDING_W = (f'{name}_PADDING', f'{name}_PADDING') if param.padding[0] == param.padding[1] else (f'{name}_PADDING_X', f'{name}_PADDING_Y')
            IN_CHANNELS, OUT_CHANNELS = f'{name}_IM_CH', f'{name}_OUT_CH'
            IM_DIM, OUT_DIM = f'{name}_IM_DIM', f'{name}_OUT_DIM'
            
            if param.in_channels == 3 and idx == 1:
                code_buffer += f"""    arm_convolve_HWC_q7_RGB({current_module['IM_IN']},
                                {IM_DIM},
                                {IN_CHANNELS},
                                {WEIGHT},
                                {OUT_CHANNELS},
                                {KERNEL_SIZE_H},
                                {PADDING_H},
                                {STRIDE_H},
                                {BIAS},
                                {name}_BIAS_LSHIFT,
                                {name}_OUT_RSHIFT,
                                {current_module['BUFFER']},
                                {OUT_DIM},
                                (q15_t*)col_buffer,
                                NULL
                                );\n"""
                code_buffer += f"""    save("{vs_prj_path}/{current_module['BUFFER']}.raw", {current_module['BUFFER']}, sizeof({current_module['BUFFER']}));\n"""
                # current_module['IM_IN'] = 'buffer2'
                tmp_out_variable_buffer += f"q7_t {name.lower()}_out[{OUT_CHANNELS}*{OUT_DIM}*{OUT_DIM}];\n"
            elif 'DEPTHWISE' in name:
                # print('depthwise conv')
                code_buffer += f"""    arm_depthwise_separable_conv_HWC_q7({current_module['IM_IN']},
                                        {IM_DIM},
                                        {IN_CHANNELS},
                                        {WEIGHT},
                                        {OUT_CHANNELS},
                                        {KERNEL_SIZE_H},
                                        {PADDING_H},
                                        {STRIDE_H},
                                        {BIAS},
                                        {name}_BIAS_LSHIFT,
                                        {name}_OUT_RSHIFT,
                                        {current_module['BUFFER']},
                                        {OUT_DIM},
                                        (q15_t*)col_buffer,
                                        NULL);\n"""
                'arm_depthwise_separable_conv_HWC_q7_nonsquare'
                code_buffer += f"""    save("{vs_prj_path}/{current_module['BUFFER']}.raw", {current_module['BUFFER']}, sizeof({current_module['BUFFER']}));\n"""
                tmp_out_variable_buffer += f"q7_t {current_module['BUFFER']}[{OUT_CHANNELS}*{OUT_DIM}*{OUT_DIM}];\n"
            elif param.kernel_size[0] == 1 and param.kernel_size[0] == 1:
                # print('1x1 conv')
                code_buffer += f"""    arm_convolve_1x1_HWC_q7_fast_nonsquare({current_module['IM_IN']},
                                            {IM_DIM},
                                            {IM_DIM},
                                            {IN_CHANNELS},
                                            {WEIGHT},
                                            {OUT_CHANNELS},
                                            {KERNEL_SIZE_H},
                                            {KERNEL_SIZE_W},
                                            {PADDING_H},
                                            {PADDING_W},
                                            {STRIDE_H},
                                            {STRIDE_W},
                                            {BIAS},
                                            {name}_BIAS_LSHIFT,
                                            {name}_OUT_RSHIFT,
                                            {current_module['BUFFER']},
                                            {OUT_DIM},
                                            {OUT_DIM},
                                            (q15_t*)col_buffer,
                                            NULL);\n"""
                code_buffer += f"""    save("{vs_prj_path}/{current_module['BUFFER']}.raw", {current_module['BUFFER']}, sizeof({current_module['BUFFER']}));\n"""
                tmp_out_variable_buffer += f"q7_t {current_module['BUFFER']}{OUT_CHANNELS}*{OUT_DIM}*{OUT_DIM}];\n"
            else:
                code_buffer += f"""    arm_convolve_HWC_q7_fast({current_module['IM_IN']},
                                        {IM_DIM},
                                        {IN_CHANNELS},
                                        {WEIGHT},
                                        {OUT_CHANNELS},
                                        {KERNEL_SIZE_H},
                                        {PADDING_H},
                                        {STRIDE_H},
                                        {BIAS},
                                        {name}_BIAS_LSHIFT,
                                        {name}_OUT_RSHIFT,
                                        {current_module['BUFFER']},
                                        {OUT_DIM},
                                        (q15_t*)col_buffer,
                                        NULL);\n"""
                
                code_buffer += f"""    save("{vs_prj_path}/{current_module['BUFFER']}.raw", {current_module['BUFFER']}, sizeof({current_module['BUFFER']}));\n"""
                tmp_out_variable_buffer += f"q7_t {current_module['BUFFER']}[{OUT_CHANNELS}*{OUT_DIM}*{OUT_DIM}];\n"
                # print('Fast conv')
                '''
                arm_convolve_HWC_q7_basic
                arm_convolve_HWC_q7_basic_nonsquare
                arm_convolve_HWC_q7_fast_nonsquare
                '''
                
                
        elif isinstance(param, nn.Linear):
            'arm_fully_connected_q7'
        
            IN_DIM, OUT_DIM = f'{name}_DIM', f'{name}_OUT'
            
            WEIGHT = f'{name.lower()}_wt'
            BIAS = f'{name.lower()}_bias'
            
            variable_buffer += f"static q7_t {name.lower()}_wt[{name}_WT_SHAPE] = {name}_WT;\n"
            variable_buffer += f"static q7_t {name.lower()}_bias[{name}_BIAS_SHAPE] = {name}_BIAS;\n"
            code_buffer += f"""    arm_fully_connected_q7_opt({current_module['IM_IN']},
                                    {WEIGHT},
                                    {IN_DIM},
                                    {OUT_DIM},
                                    {name}_BIAS_LSHIFT,
                                    {name}_OUT_RSHIFT,
                                    {BIAS},
                                    {current_module['BUFFER']},
                                    (q15_t*)fc_buffer
                                    );\n"""
            code_buffer += f"""    save("{vs_prj_path}/{current_module['BUFFER']}.raw", {current_module['BUFFER']}, sizeof({current_module['BUFFER']}));\n"""
            tmp_out_variable_buffer += f"q7_t {current_module['BUFFER']}[{OUT_DIM}];\n"
        elif isinstance(param, nn.ReLU):
            'arm_relu_q7'
            'arm_relu6_s8'
            code_buffer += f"""    arm_relu_q7({current_module['IM_IN']}, {OUT_DIM}*{OUT_DIM}*{OUT_CHANNELS});\n"""
            continue
            # previous_module['IM_IN'], current_module['IM_IN'] = current_module['IM_IN'], previous_module['IM_IN']
            # previous_module['BUFFER'], current_module['BUFFER'] = current_module['BUFFER'], previous_module['BUFFER']
        elif isinstance(param, nn.MaxPool2d):
            '''
            arm_maxpool_q7_HWC
            The pooling function is implemented as split x-pooling then y-pooling.
            This pooling function is input-destructive. Input data is undefined after calling this function.
            '''
            
            code_buffer += f"""    arm_maxpool_q7_HWC({current_module['IM_IN']},
                                    {name}_IM_DIM,
                                    {name}_IM_CH,
                                    {name}_KER_DIM,
                                    {name}_PADDING,
                                    {name}_STRIDE,
                                    {name}_OUT_DIM,
                                    {current_module['BUFFER']},
                                    NULL);\n"""
            code_buffer += f"""    save("{vs_prj_path}/{current_module['BUFFER']}.raw", {current_module['BUFFER']}, sizeof({current_module['BUFFER']}));\n"""
        elif isinstance(param, nn.AdaptiveAvgPool2d):
            code_buffer += f"""    arm_avepool_q7_HWC({current_module['IM_IN']},
                                    {name}_IM_DIM,
                                    {name}_IM_CH,
                                    {name}_KER_DIM,
                                    {name}_PADDING,
                                    {name}_STRIDE,
                                    {name}_OUT_DIM,
                                    col_buffer,
                                    {current_module['BUFFER']});\n"""
            tmp_out_variable_buffer += f"q7_t {current_module['BUFFER']}[{OUT_CHANNELS}*{name}_OUT_DIM*{name}_OUT_DIM];\n"
            code_buffer += f"""    save("{vs_prj_path}/{current_module['BUFFER']}.raw", {current_module['BUFFER']}, sizeof({current_module['BUFFER']}));\n"""
        elif isinstance(param, nn.AvgPool2d):
            code_buffer += f"""    arm_avepool_q7_HWC({current_module['IM_IN']},
                                    {name}_IM_DIM,
                                    {name}_IM_CH,
                                    {name}_KER_DIM,
                                    {name}_PADDING,
                                    {name}_STRIDE,
                                    {name}_OUT_DIM,
                                    col_buffer,
                                    {current_module['BUFFER']});\n"""
            tmp_out_variable_buffer += f"q7_t {current_module['BUFFER']}[{OUT_CHANNELS}*{name}_OUT_DIM*{name}_OUT_DIM];\n"
            code_buffer += f"""    save("{vs_prj_path}/{current_module['BUFFER']}.raw", {current_module['BUFFER']}, sizeof({current_module['BUFFER']}));\n"""
        elif isinstance(param, nn.Softmax):
            code_buffer += f"""    arm_softmax_q7({current_module['IM_IN']}, {OUT_DIM}, {current_module['BUFFER']});\n"""
        elif isinstance(param, nn.Flatten):
            continue
        else:
            raise NotImplementedError(f'Not implemented for {param}')
        current_module['IM_IN'], current_module['BUFFER'] = current_module['BUFFER'], current_module['IM_IN']
    else:
        code_buffer += f"""    return {current_module['IM_IN']};\n"""
        
    code_buffer += '}'
    
    
    path = "E:/2_Quantization/deployment-with-CMSIS-NN/CMSIS_NN_PC_simulator/Deploy_Simulator"
    
    with open(os.path.join(path, 'nn.cpp'), 'w') as F:
        code_buffer = code_buffer%tmp_out_variable_buffer
        F.write(code_buffer)
    with open(os.path.join(path, 'nn.h'), 'w') as F:
        F.write(variable_buffer)
    # for idx, (name, param) in enumerate(weights.named_parameters()):
    #     print(idx, name)
    #     print()
