import os
import sys
import math
import argparse
import numpy as np

from packaging import version
from abc import ABC, abstractmethod
import torch  # TensorFlow를 PyTorch로 대체
import torch.nn as nn  # PyTorch의 nn 모듈 import

DEFAULT_TESTDATA_SET = 'TFLM'
OUTDIR = '../../on_line/lib/Modules/symmetric/'
PREGEN = 'PregeneratedData/'

LICENSE = '''// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// This source code is licensed under the Apache-2.0 license found in the
// LICENSE file in the root directory of this source tree.
// ----------------------------------------------------------------------------'''

def parse_args():
    parser = argparse.ArgumentParser(description="Generate input and refererence output data for unittests."
                                     "It can regenerate all or load all data/input or only parts of it, "
                                     "which may be useful when debugging.")
    parser.add_argument('--regenerate-weights', action='store_true', help="Regenerate and store new weights")
    parser.add_argument('--regenerate-input', action='store_true', help="Regenerate and store new input")
    parser.add_argument('--regenerate-biases', action='store_true', help="Regenerate and store new biases")
    parser.add_argument('-a', '--regenerate-all', action='store_true', help="Regenerate and store all data")
    # parser.add_argument('--dataset', type=str, default=DEFAULT_TESTDATA_SET, help="Name of generated test set")
    # parser.add_argument('-t', '--type', type=str, default='conv', choices=['conv', 'pooling'], help='Type of test.')

    args = parser.parse_args()
    return args


class TestSettings(ABC):

    INT_MAX = 32767
    INT_MIN = -32767

    def __init__(self, args, in_ch, out_ch, x_in, y_in, w_x, w_y, stride_x, stride_y, pad, randmin, randmax,
                 outminrange=-128, outmaxrange=127, batches=1):

        self.minrange = -128
        self.maxrange = 127

        # Randomization interval
        self.mins = randmin
        self.maxs = randmax

        self.input_ch = in_ch
        self.output_ch = out_ch
        self.x_input = x_in
        self.y_input = y_in
        self.filter_x = w_x
        self.filter_y = w_y
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.batches = batches

        self.has_padding = pad

        self.scaling_factors = []

        minrange = randmin - 1
        maxrange = randmax + 1

        (self.input_scale, self.input_zero_point) = self.derive_scale_and_zeropoint_from_min_max(minrange, maxrange)
        (self.output_scale, self.output_zero_point) = self.derive_scale_and_zeropoint_from_min_max(outminrange,
                                                                                                   outmaxrange)
        self.generated_header_files = []
        self.pregenerated_data_dir = PREGEN
        self.testdataset = DEFAULT_TESTDATA_SET
        
        self.kernel_table_file = os.path.join(self.pregenerated_data_dir, self.testdataset, 'kernel.txt')
        self.inputs_table_file = os.path.join(self.pregenerated_data_dir, self.testdataset, 'input.txt')
        self.bias_table_file = os.path.join(self.pregenerated_data_dir, self.testdataset, 'bias.txt')
        self.parameters_file = os.path.join(self.pregenerated_data_dir, self.testdataset, 'params.txt')

        self.set_output_dims_and_padding()

        self.regenerate_new_weights = args.regenerate_weights
        self.regenerate_new_input = args.regenerate_input
        self.regenerate_new_bias = args.regenerate_biases
        if not os.path.exists(self.parameters_file) or args.regenerate_all:
            self.regenerate_new_bias = True
            self.regenerate_new_weights = True
            self.regenerate_new_input = True
            self.save_parameters()
        else:
            self.load_parameters()
        # self.headers_dir = os.path.join(OUTDIR, self.testdataset)
        self.headers_dir = os.path.join(OUTDIR)

    def clamp_int8(self, result):
        int8_min = self.minrange
        int8_max = self.maxrange
        result = np.where(result < int8_min, int8_min, result)
        result = np.where(result > int8_max, int8_max, result)
        return result

    def derive_scale_and_zeropoint_from_min_max(self, minrange, maxrange):
        scale = (maxrange - minrange) / ((self.maxrange * 1.0) - self.minrange)
        zeropoint = self.minrange + int(-minrange / scale + 0.5)
        zeropoint = max(-128, min(zeropoint, 128))
        return (scale, zeropoint)

    def save_multiple_dim_array_in_txt(self, file, data):
        header = ','.join(map(str, data.shape))
        np.savetxt(file, data.reshape(-1, data.shape[-1]), header=header, delimiter=',')

    def load_multiple_dim_array_from_txt(self, file):
        with open(file) as f:
            shape = list(map(int, next(f)[1:].split(',')))
            data = np.genfromtxt(f, delimiter=',').reshape(shape)
        return data.astype(np.float32)

    def save_parameters(self):
        regendir = os.path.dirname(self.parameters_file)
        if not os.path.exists(regendir):
            os.makedirs(regendir)
        params = np.array([self.input_ch, self.output_ch, self.x_input, self.y_input, self.filter_x, self.filter_y,
                           self.stride_x, self.stride_y, self.pad_x, self.pad_y, self.batches, self.has_padding])
        np.savetxt(self.parameters_file, params, fmt='%i')

    def load_parameters(self):
        params = np.loadtxt(self.parameters_file).astype(int)
        (self.input_ch, self.output_ch, self.x_input, self.y_input, self.filter_x, self.filter_y,
         self.stride_x, self.stride_y, self.pad_x, self.pad_y, self.batches, self.has_padding) = (map(lambda x: x, params))

    def convert_tensor_np(self, tensor_in, converter):
        w = tensor_in.numpy()
        shape = w.shape
        w = w.ravel()
        fw = converter(w)
        fw.shape = shape
        return torch.tensor(fw)

    def convert_tensor(self, tensor_in, converter):
        w = tensor_in.numpy()
        shape = w.shape
        w = w.ravel()
        normal = np.array(w)
        float_normal = []

        for i in normal:
            float_normal.append(converter(i))

        np_float_array = np.asarray(float_normal)
        np_float_array.shape = shape

        return torch.tensor(np_float_array)

    def get_randomized_data(self, dims, npfile, regenerate, decimals=0):
        if not os.path.exists(npfile) or regenerate:
            regendir = os.path.dirname(npfile)
            if not os.path.exists(regendir):
                os.makedirs(regendir)
            if decimals == 0:
                data = torch.randint(self.mins, self.maxs, dims, dtype=torch.int32).float()  
            else:
                data = torch.FloatTensor(*dims).uniform_(self.mins, self.maxs)
                data = torch.tensor(np.around(data.numpy(), decimals))
            print("Saving data to {}".format(npfile))
            self.save_multiple_dim_array_in_txt(npfile, data.numpy())
        else:
            print("Loading data from {}".format(npfile))
            data = torch.tensor(self.load_multiple_dim_array_from_txt(npfile))  
        return data
       
    def write_c_header_wrapper(self):
        filename = "test_data.h"
        filepath = os.path.join(self.headers_dir, filename)
        print("Generating C header wrapper {}...".format(filepath))
        with open(filepath, 'w+') as f:
            f.write("{}\n\n".format(LICENSE))
            while len(self.generated_header_files) > 0:
                f.write('#include "{}"\n'.format(self.generated_header_files.pop()))

    def write_c_config_header(self):
        filename = "config_data.h"
        self.generated_header_files.append(filename)
        filepath = os.path.join(self.headers_dir, filename)
        prefix = self.testdataset.upper()
        print("Writing C header with config data {}...".format(filepath))
        with open(filepath, "w+") as f:
            f.write("{}\n".format(LICENSE))
            f.write("#pragma once\n")
            f.write("#define {}_OUT_CH {}\n".format(prefix, self.output_ch))
            f.write("#define {}_IN_CH {}\n".format(prefix, self.input_ch))
            f.write("#define {}_CONV_W {}\n".format(prefix, self.x_input))
            f.write("#define {}_CONV_H {}\n".format(prefix, self.y_input))
            f.write("#define {}_FILTER_X {}\n".format(prefix, self.filter_x))
            f.write("#define {}_FILTER_Y {}\n".format(prefix, self.filter_y))
            f.write("#define {}_STRIDE_X {}\n".format(prefix, self.stride_x))
            f.write("#define {}_STRIDE_Y {}\n".format(prefix, self.stride_y))
            f.write("#define {}_PAD_X {}\n".format(prefix, self.pad_x))
            f.write("#define {}_PAD_Y {}\n".format(prefix, self.pad_y))
            f.write("#define {}_OUT_CONV_W {}\n".format(prefix, self.x_output))
            f.write("#define {}_OUT_CONV_H {}\n".format(prefix, self.y_output))
            f.write("#define {}_DST_SIZE {}\n".format(prefix, self.x_output * self.y_output * self.output_ch))
            f.write("#define {}_INPUT_SIZE {}\n".format(prefix, self.x_input * self.y_input * self.input_ch))
            f.write("#define {}_INPUT_OFFSET {}\n".format(prefix, -self.input_zero_point))
            f.write("#define {}_OUTPUT_OFFSET {}\n".format(prefix, self.output_zero_point))
            f.write("#define {}_OUT_ACTIVATION_MIN {}\n".format(prefix, self.minrange))
            f.write("#define {}_OUT_ACTIVATION_MAX {}\n".format(prefix, self.maxrange))
            f.write("#define {}_INPUT_BATCHES {}\n".format(prefix, self.batches))
    
    def generate_c_array(self, name:str, array:torch.Tensor, datatype="q7_t", const="const "):
        if not os.path.exists(self.headers_dir):
            os.makedirs(self.headers_dir)

        w = None
        if isinstance(array, list):
            w = array
            size = len(array)
        else:
            w = array.numpy()
            w = w.ravel()
            size = torch.numel(array)

        filename = name + "_data.h"
        filepath = os.path.join(self.headers_dir, filename)

        self.generated_header_files.append(filename)
        print("Generating C header {}...".format(filepath))
        with open(filepath, "w+") as f:
            f.write("{}\n".format(LICENSE))
            f.write("#pragma once\n")
            f.write("#include <stdint.h>\n\n")
            f.write(const + datatype + " " + self.testdataset + '_' + name + "[%d] =\n{\n" % size)
            for i in range(size - 1):
                f.write("  %d,\n" % w[i])
            f.write("  %d\n" % w[size - 1])
            f.write("};\n")

    def set_output_dims_and_padding(self):
        if self.has_padding:
            self.x_output = math.ceil(float(self.x_input) / float(self.stride_x))
            self.y_output = math.ceil(float(self.y_input) / float(self.stride_y))
            self.padding = 'SAME'
            pad_along_width = max((self.x_output - 1) * self.stride_x + self.filter_x - self.x_input, 0)
            pad_along_height = max((self.y_output - 1) * self.stride_y + self.filter_y - self.y_input, 0)
            pad_top = pad_along_height // 2
            pad_left = pad_along_width // 2
            self.pad_x = pad_left
            self.pad_y = pad_top
        else:
            self.x_output = math.ceil(float(self.x_input - self.filter_x + 1) / float(self.stride_x))
            self.y_output = math.ceil(float(self.y_input - self.filter_y + 1) / float(self.stride_y))
            self.padding = 'VALID'
            self.pad_x = 0
            self.pad_y = 0

    @abstractmethod
    def generate_data(self, input_data=None, weights=None, biases=None):
        ''' Must be overriden '''


class ConvSettings(TestSettings):
    def __init__(self, args, in_ch=1, out_ch=1, x_in=7, y_in=7, w_x=3, w_y=3, stride_x=2, stride_y=2,
                 pad=True, randmin=-7, randmax=7, outminrange=-128, outmaxrange=127, batches=1):
        super().__init__(args, in_ch, out_ch, x_in, y_in, w_x, w_y, stride_x, stride_y, pad, randmin, randmax,
                         outminrange, outmaxrange, batches)

    def quantize_bias(self, nparray):
        num_channels = self.output_ch
        quantized_values = []
        values = np.array(nparray)

        def quantize_float_to_int(value, scale):
            quantized = round(value / scale)
            if quantized > self.INT_MAX:
                quantized = self.INT_MAX
            elif quantized < self.INT_MIN:
                quantized = self.INT_MIN
            return quantized

        for x in range(num_channels):
            quantized_values.append(quantize_float_to_int(values[x], self.scaling_factors[x]*self.input_scale))

        return np.asarray(quantized_values)

    def reshape_kernel(self, kernel:torch.Tensor):
        # PyTorch의 메모리 레이아웃에 맞게 수정
        # Output channel, Input channel//Group_ch, Filter height, Filter width
        kernel = torch.reshape(kernel, [self.output_ch, self.input_ch, self.filter_y, self.filter_x])
        return kernel

    def quantize_input(self, value):
        result = np.round(value / self.input_scale) + self.input_zero_point
        return self.clamp_int8(result)

    def quantize_output(self, value):
        result = round(value / self.output_scale) + self.output_zero_point
        return self.clamp_int8(result)
        
    def quantize_filter(self, nparray:np.ndarray):
        quantized_values = []
        channel_count = self.output_ch
        input_size = self.filter_y * self.filter_x * self.input_ch * self.output_ch
        per_channel_size = input_size // channel_count
        values = np.array(nparray)
        stride = 1
        channel_stride = per_channel_size

        for channel in range(channel_count):
            fmin = 0
            fmax = 0
            for i in range(per_channel_size):
                idx = channel * channel_stride + i * stride
                fmin = min(fmin, values[idx])
                fmax = max(fmax, values[idx])

            self.scaling_factors.append(max(abs(fmin), abs(fmax)) / self.maxrange)

            for x in range(per_channel_size):
                chs = channel * channel_stride + x * stride
                quantized_value = round(round(values[chs]) / self.scaling_factors[channel])

                # Clamp
                quantized_value = min(127, max(-127, quantized_value))
                quantized_values.append(quantized_value)

        return np.asarray(quantized_values)
    
    def generate_quantize_per_channel_multiplier(self):
        num_channels = self.output_ch
        per_channel_multiplier = []
        per_channel_shift = []

        if len(self.scaling_factors) != num_channels:
            raise RuntimeError("Missing scaling factors")

        def quantize_scale(scale):
            significand, shift = math.frexp(scale)
            significand_q31 = round(significand * (1 << 31))
            return significand_q31, shift

        for i in range(num_channels):
            effective_output_scale = self.input_scale * self.scaling_factors[i] / self.output_scale
            (quantized_multiplier, shift) = quantize_scale(effective_output_scale)

            per_channel_multiplier.append(quantized_multiplier)
            per_channel_shift.append(shift)

        self.generate_c_array("output_mult", per_channel_multiplier, datatype='int32_t')
        self.generate_c_array("output_shift", per_channel_shift, datatype='int32_t')

    
    def convolution(self, indata, weights, bias=None):
        indata = indata.float()
        weights = weights.float()
        bias = bias.float() if bias is not None else None
        
        # PyTorch의 conv2d 함수 사용
        if self.has_padding:
            out = nn.functional.conv2d(indata, weights, stride=(self.stride_x, self.stride_y), padding=(self.pad_x, self.pad_y))
        else:
            out = nn.functional.conv2d(indata, weights, stride=(self.stride_x, self.stride_y))
        
        if torch.tensor([self.batches, self.y_output, self.x_output, self.output_ch]).tolist() != \
           list(out.size()):
            raise RuntimeError("Shape mismatch, need to regenerate data?")

        if bias is not None:
            out = out + bias.view(1, -1, 1, 1)  # PyTorch에서 bias 더하기
        out = torch.clamp(out, self.minrange, self.maxrange)  # PyTorch clamp 함수 사용
        return out

    def generate_data(self, input_data=None, weights=None, biases=None):
        if input_data is not None:
            input_data = torch.reshape(torch.tensor(input_data),
                                       [self.batches, self.input_ch, self.y_input, self.x_input])
        else:
            # NHWC
            input_data = self.get_randomized_data([self.batches, self.input_ch, self.y_input, self.x_input],
                                                               self.inputs_table_file,
                                                               regenerate=self.regenerate_new_input).clone().detach()
            # NHWC -> NCHW
        if weights is not None:
            weights = torch.reshape(torch.tensor(weights), 
                                    [self.output_ch, self.input_ch, self.filter_y, self.filter_x])
        else:
            # NHWC
            weights = self.get_randomized_data([self.output_ch, self.input_ch, self.filter_y, self.filter_x],
                                                self.kernel_table_file,
                                                regenerate=self.regenerate_new_weights).clone().detach()
            # NHWC -> NCHW
        if biases is not None:
            biases = torch.reshape(torch.tensor(biases), [self.output_ch])
        else:
            # N 차원 배열 생성
            biases = self.get_randomized_data([self.output_ch],
                                            self.bias_table_file,
                                            regenerate=self.regenerate_new_bias).clone().detach()

        # Generate conv reference
        conv = self.convolution(input_data, weights, biases)
        
        # Quantize and write to C headers
        ## CMSIS Format에 맞도록 C코드로 변환
        # 차원 순서 변경: (batches, input_ch, height, width) -> (batches, height, width, input_ch)
        self.generate_c_array("input", self.convert_tensor(input_data.permute(0, 2, 3, 1), self.quantize_input)) 
        # 차원 순서 변경: (output_ch, input_ch, filter_y, filter_x) -> (output_ch, filter_y, filter_x, input_ch)
        self.generate_c_array("weights", self.convert_tensor_np(weights.permute(0, 2, 3, 1), self.quantize_filter))  
        self.generate_c_array("biases", self.convert_tensor_np(biases, self.quantize_bias), "int32_t")
        self.generate_quantize_per_channel_multiplier()
        self.generate_c_array("output_ref", self.convert_tensor(conv, self.quantize_output))
        self.write_c_config_header()
        self.write_c_header_wrapper()

if __name__ == '__main__':
    args = parse_args()
    generator = ConvSettings(args, in_ch=3, out_ch=32, x_in=32, y_in=32, w_x=5, w_y=5, stride_x=1, stride_y=1,
                                pad=True, randmin=1, randmax=4, outminrange=-126, outmaxrange=127)
    generator.generate_data()
    