#define _CRT_SECURE_NO_WARNINGS
#include <stdint.h>
#include <stdio.h>
#include "nn.h"
#include <iostream>
#include "modules.h"
#include <fstream>
#include <memory>
using namespace std;
//q7_t* load(const char* file)
//{
//	size_t sz;
//	q7_t* in;
//	//printf("Loading %s\n", file);
//	FILE* fp = fopen(file, "rb");
//	// assert(fp);
//	fseek(fp, 0, SEEK_END);
//	sz = ftell(fp);
//	fseek(fp, 0, SEEK_SET);
//	in = (q7_t*)malloc(sz); // Cast the return value of malloc to q7_t*
//	fread(in, 1, sz, fp);
//	fclose(fp);
//	return in;
//}

int main() {
	/*
	//q7_t input[96 * 96 * 3] = INPUT_DATA_0_7;
	q7_t* input = load("E:/2_Quantization/deployment-with-CMSIS-NN/CMSIS_NN_PC_simulator/Deploy_Simulator/logs/input.raw");
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
	*/
	return EXIT_SUCCESS;

}