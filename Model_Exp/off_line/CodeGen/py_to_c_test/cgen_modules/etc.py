from string import Template
from collections import defaultdict

def list_bracket(lists:list):
	return "{" + ', '.join(map(str, lists)) + "}"

def static_q7_t(name:str):
    return f"static q7_t {name} = {name.upper()};"

def parameter_macro(name:str, value:str):
	return f'#define {name.upper()} {value}'

def ddict():
    return defaultdict(ddict)

def codegen_nn_cpp(comp_graphs:str):
    """
        Implementing the compuational graph to the run_nn function
    """
    return Template(
"""
#pragma once
#include "nn.h"
#include "arm_nnfunctions.h"
#include <iostream>
#include <fstream>
#include <memory>

q7_t* run_nn(q7_t* input_data, q7_t* output_data, q7_t* buffer1, q7_t* buffer2, q7_t* col_buffer, q7_t* fc_buffer) {
    ${comp_graphs}
    
    return output_data;
} 
""").substitute(comp_graphs=comp_graphs)

def codegen_nn_header(variable_buffer:str):
    """
        Implementing the run_nn function and the input data
    """
    return Template("""
#pragma once
#include "arm_nnfunctions.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "inputs.h"
#include "parameters.h"
#include "weights.h"
q7_t* run_nn(q7_t* input_data, q7_t* output_data, q7_t* buffer1, q7_t* buffer2, q7_t* col_buffer, q7_t* fc_buffer);
${variable_buffer}
""").substitute(variable_buffer=variable_buffer)
    
def save_activation(path, data, sizeof_data=None):
    
    return Template("""
    save(${path}, ${data}, sizeof(${sizeof_data}));
""").substitute(path=path, data=data, sizeof_data=sizeof_data if sizeof_data else data)

def load_activation(path):
    return Template("""
    load(${path});
""").substitute(path=path)


def save_main(raw_path, comp_graphs):
    return Template("""
#define _CRT_SECURE_NO_WARNINGS
#include "nn.h"
#include "inputs.h"
#include <iostream>
#include "modules.h"
#include <fstream>
#include <memory>
using namespace std;
q7_t* load(const char* file)
{
	size_t sz;
	q7_t* in;
	//printf("Loading %s\n", file);
	FILE* fp = fopen(file, "rb");
	// assert(fp);
	fseek(fp, 0, SEEK_END);
	sz = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	in = (q7_t*)malloc(sz); // Cast the return value of malloc to q7_t*
	fread(in, 1, sz, fp);
	fclose(fp);
	return in;
}
int main() {
	//q7_t input[96 * 96 * 3] = INPUT_DATA_0_7;
	q7_t* input = load(${raw_path});
	q7_t* buffer1 = scratch_buffer;
	q7_t* buffer2 = buffer1 + 96 * 96 * 3;
	q7_t test_output[2];
	//memset(test_output, 0, sizeof(test_output));
	q7_t col_buffer[MAX_CONV_BUFFER_SIZE*2*2];
	q7_t fc_buffer[MAX_FC_BUFFER*2];
	
	
	run_nn(input, test_output, buffer1, buffer2, col_buffer, fc_buffer);
	uint32_t index;
	//index = network(input);
	//printf("Real %d\n", INPUT_Y_DATA_0_0);
	
	//cout << "total_time: " << (double)(finish_t - begin_t) / CLOCKS_PER_SEC << "s" << endl;
	//printf("GT: %d\n", INPUT_Y_DATA_0_7);

	//for (int i = 0; i < 2; i++) {
	//	printf("Class: %d, Predict probability: %d\n", i, test_output[i]);
	//}
	
	return EXIT_SUCCESS;
}
""").safe_substitute(raw_path = raw_path, comp_graphs = comp_graphs)
    
"""
	arm_add_q7로 overflow가 있을 수 있음.(추후 확인예정)
	arm_q7_to_q15로 변환 후 arm_add_q15로 계산, arm_q15_to_q7로 변환

"""

def weight_bias_generation(name:str):
    WEIGHT = f'{name.lower()}_wt'
    BIAS = f'{name.lower()}_bias'
    
    variable_buffer = ""
    variable_buffer += "static q7_t ${WEIGHT}[${name}_WT_SHAPE] = ${name}_WT;\n"
    variable_buffer += "static q7_t ${BIAS}[${name}_BIAS_SHAPE] = ${name}_BIAS;\n"
    return Template("""
    static q7_t ${WEIGHT}[${name}_WT_SHAPE] = ${name}_WT;

""").substitute(weight=WEIGHT, name=name, BIAS=BIAS)