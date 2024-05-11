import os
from subprocess import call
import copy 
import numpy as np
import torch
from torch import (
    nn as nn,
    quantization
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from .fully_connected_opt_weight_generation import \
    convert_to_x4_q7_weights
from torch.utils.data import DataLoader

from torch.ao.nn.quantized.modules.conv import Conv2d as QConv2d
from torch.ao.nn.quantized.modules.linear import Linear as QLinear
# TODO:
# + Change interface detection to named modules DONE
# + Work with children modules DONE ----->!!!!!!!!THIS IMPOSES THAT NO GROUPING OF MODULES CAN BE DONE (SEQUENTIAL OR LIST) BECAUSE THEN THEY DO NOT APPEAR AS CHILDREN!!!!!!
# + Change buffering from individual buffering to global buffering: one buffer (duplicated) for input/output and another for column buffer
#   + The input/output buffer has to be the greatest of inputs output sizes 
#   + The col buffer has to be the greatest of column transformations for conv, pool and fc
#       + CONV: 2*ch_im_in*dim_kernel*dim_kernel
#       + POOL: 2*dim_im_out*ch_im_in
#       + FC: dim_vec
import json
from fxpmath import Fxp
from collections import OrderedDict
class CMSISConverter:
    def __init__(
            self,
            root,
            model,
            weight_file_name,
            parameter_file_name,
            linear_features,
            weight_bits=8,
            compilation_config=None,
        ):

        # TODO: defined by user should be
        self.root = root
        self.io_folder = os.path.join(self.root, "logs")
        os.makedirs(self.io_folder, exist_ok=True)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.model = copy.deepcopy(model)
        self.model_qq = copy.deepcopy(model)
        self.parameter_file_name = os.path.join(self.root, parameter_file_name)
        self.weight_file_name = os.path.join(self.root, weight_file_name)

        with open(self.parameter_file_name, "w") as w_file:
            w_file.write("#pragma once\n")
        with open(self.weight_file_name, "w") as w_file:
            w_file.write("#pragma once\n")

        self.compilation = compilation_config
        # define storage for maximum buffers in CMSIS
        self.max_col_buffer = 0
        self.max_fc_buffer = 0

        self.weight_bits = weight_bits
        # TODO: the image shape befor efully connected is only known because
        # there is a function the model that gets it. Should be independent of tht function
        self.conv_linear_interface_shape = linear_features

        # for storing all convolution, pooling and linear params
        self.params = {}
        self.q_params = {}
        self.connectivity = {}
        self.param_prefix_name = None
        self.last_layer_name = None
        self.fxp_q_params = {}
        # storing inputs and ouputs quantized
        self.logging = {}
        
    
    def convert_model(self, loader):
        #! Convert!!
        # Observering the model
        self.generate_intermediate_values(loader)
        # PADDING, STRIDE, KERNEL, IN_CH, OUT_CH, IM_DIM, OUT_DIM
        self.save_params_model()
        
        # Refine the weight Q-bit
        self.refine_model_weights(loader, search_range=0)
        # Refine the bias Q-bit
        self.refine_model_weights(loader, bias=True, search_range=0)
        # Refine the activation Q-bit
        self.refine_activations(loader, search_range=0)
        # Reassign the Q-params and shifts
        self.reassign_q_params_n_shifts()
        
        # Write the Q-params and shifts
        self.write_shifts_n_params()
        # Convert the weights
        self.convert_weights()

    def module_name_adapter(self, name):
        return name.replace(".", "_").upper()

    def quantize_input(self, inp):
        if inp.is_cuda:
            inp = inp.cpu()
        qtensor = self.extract_q_quantize_tensor(inp).permute(1, 2, 0).numpy().astype(np.int8)
        try:
            qtensor.tofile(os.path.join(self.io_folder, "input.raw"))
        except:
            from time import sleep
            sleep(1)
            qtensor.tofile(os.path.join(self.io_folder, "input.raw"))
        return qtensor
    
    def compute_fractional_bits(self, min_value, max_value):
        # max_value = torch.abs(torch.max(max_value)) - torch.abs(torch.max(max_value)/pow(2, self.weight_bits)) # allow very small saturation.
        # min_value = torch.abs(torch.min(min_value)) - torch.abs(torch.min(min_value)/pow(2, self.weight_bits))
        return int(torch.ceil(torch.log2(torch.max(torch.abs(max_value), torch.abs(min_value)))).item())
    
    def compute_fractional_bits_tensor(self, weight, maximum_bit=32):
        # return Integer bits
        return min((self.weight_bits - 1- self.compute_fractional_bits(torch.min(weight), torch.max(weight))),maximum_bit)

    def extract_q_quantize_tensor(self, weight, method='minmax', percentile=0.995):
        # weight = torch.clamp(weight, 
        #                      min=torch.quantile(weight, q=1-percentile), 
        #                      max=torch.quantile(weight, q=percentile)) # clip outlier
        if method == 'minmax':
            q_frac = self.compute_fractional_bits_tensor(weight)
        elif method == 'kld':
            q_frac = self.find_dec_bits_kld(weight)
        else:
            raise ValueError("Method not supported")
        return self.quantize_tensor(weight, q_frac)

    
    def find_dec_bits_kld(self, weight, bit_width=8, scan_times=4, maximum_bit=16):
        from scipy import stats
        """
        # saturation shift, using KLD method (Kullback-Leibler divergence)
        # Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
        :param data: The data for looking for quantisation
        :param bit_width: the bitwidth of the data
        :param scan_times: the times to try the best kld (normally the second is the best.)
        :return: dec bit width for this data
        """
        # do a regular non-saturated quantisation
        max_val = torch.max(weight)
        min_val = torch.min(weight)
        abs_max = torch.max(torch.abs(max_val), torch.abs(min_val))
        int_bits = int(torch.ceil(torch.log2(max(torch.abs(max_val), torch.abs(min_val)))).item())
        dec_bits = (bit_width-1) - int_bits

        # now looking for the best quantisation using KLD method
        small_var = 1e-5
        bins = torch.arange(-abs_max, abs_max, abs_max / 2048 * 2)
        q_bins = torch.arange(-abs_max, abs_max, abs_max / 256 * 2)
        flat_hist = torch.tensor(np.histogram(torch.flatten(weight).numpy(), bins=bins)[0])
        kl_loss = []
        kl_shifts = []
        for shift in range(scan_times):
            t = 2 ** (dec_bits  + shift)  # 2-based threshold
            act = torch.round(torch.flatten(weight) * t)
            act = act / t
            act = torch.clip(act, -128 / t, 127 / t)
            act = torch.tensor(np.histogram(act.numpy(), bins=q_bins)[0])
            act_hist = torch.zeros(2047)
            chunk = int(2048 / 256)
            for i in range(int(255)):
                none_zero = torch.count_nonzero(flat_hist[i * chunk:(i + 1) * chunk])
                if none_zero == 0:
                    continue
                for j in range(chunk):
                    act_hist[i * chunk + j] = act[i] / none_zero if flat_hist[i * chunk + j] != 0 else 0
            flat_hist[flat_hist == 0] = small_var
            act_hist[act_hist == 0] = small_var
            kl = stats.entropy(flat_hist, act_hist)
            kl_loss.append(kl)
            kl_shifts.append(dec_bits + shift)

        # now get the least loss from the scaned kld shift
        dec_bits = kl_shifts[np.argmin(kl_loss).item()]  # set the dec_bit to the KLD results
        return min(dec_bits, maximum_bit)
    
    def quantize_tensor(self, weight, q_frac):
        # return torch.clamp(torch.ceil(weight * (2 ** q_frac)).type(torch.int8), 
        #                    min=-(2 ** (self.weight_bits - 1)), 
        #                    max=(2 ** (self.weight_bits - 1) - 1))
        return torch.ceil(weight * (2 ** q_frac)).type(torch.int8)


    def generate_intermediate_values(self, loader):
        self.model = self.model.to(self.device)
        from torch.ao.quantization import QConfig, HistogramObserver, prepare
        self.model.qconfig = QConfig(
            activation=HistogramObserver.with_args(
                quant_min=-127,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
                reduce_range=True),
            weight=HistogramObserver.with_args(
                quant_min=-127,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
                reduce_range=True)
            )
        self.model = prepare(self.model, inplace=False)
        # self.model.qconfig = quantization.QConfig(
        #     activation=quantization.HistogramObserver.with_args(
        #         dtype=torch.qint8,
        #         qscheme=torch.per_tensor_symmetric),
        #     weight=quantization.HistogramObserver.with_args(
        #         dtype=torch.qint8,
        #         qscheme=torch.per_tensor_symmetric))
        # self.model = quantization.prepare(
        #     self.model,
        #     prehook=quantization.HistogramObserver.with_args(
        #         dtype=torch.qint8,
        #         qscheme=torch.per_tensor_symmetric))
        # quantization.convert(self.model, inplace=True)
        from tqdm import tqdm
        self.model = torch.quantization.convert(self.model, inplace=False)
        # self.model_qq = copy.deepcopy(self.model)
        register_hooks(self.model)
        for sample, _ in tqdm(loader, total=len(loader)):
            sample = sample.to(self.device) if not sample.is_cuda else sample
            _ = self.model(sample)
        print(self.model)
        
    def save_params_model(self):
        previous_module = "input"
        from torch.ao.quantization import QuantStub, DeQuantStub
        parents = {name: module for name, module in self.model.named_children()}
        for name, module in self.model.named_children():
            if isinstance(module, (QuantStub, DeQuantStub)):
                continue
            print(name, module)
            self.param_prefix_name = name
            if isinstance(module, (nn.Conv2d, QConv2d)):
                self.save_params_conv(module)
            if isinstance(module, (nn.Linear, QLinear)):
                self.save_params_linear(module)
            if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, 
                                   nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d)):
                self.save_params_pool(module, previous_module, parents)
            if isinstance(module, (nn.Conv2d, nn.Linear, QConv2d, QLinear)):
                self.connectivity[self.param_prefix_name] = previous_module
                previous_module = name
                self.q_params[self.param_prefix_name] = {}
                self.save_qparams_module(module)
        else:
            self.last_layer_name = name.lower()
        print("Hello~~")

    def return_act_inp_bits(self, module):
        input = module.input.dequantize()
        output = module.output.dequantize()
        
        act_bits = self.weight_bits - 1 - self.compute_fractional_bits(input.min(), input.max())
        inp_bits = self.weight_bits - 1 - self.compute_fractional_bits(output.min(), output.max())
        return act_bits, inp_bits

    def save_qparams_module(self, module):
        act_bits, inp_bits = self.return_act_inp_bits(module)
        self.compute_output_bias_shifts(module.weight, module.bias, act_bits, inp_bits)

    def compute_output_bias_shifts(self, weight, bias, activation_bits, input_bits):
        q_weight = self.compute_fractional_bits_tensor(weight)
        q_bias = self.compute_fractional_bits_tensor(bias)

        self.q_params[self.param_prefix_name]["BIAS_LSHIFT"] = (input_bits + q_weight - q_bias)
        self.q_params[self.param_prefix_name]["OUT_RSHIFT"] = (input_bits + q_weight - activation_bits)
        self.q_params[self.param_prefix_name]["WEIGHT_Q"] = q_weight
        self.q_params[self.param_prefix_name]["BIAS_Q"] = q_bias
        self.q_params[self.param_prefix_name]["INPUT_Q"] = input_bits
        self.q_params[self.param_prefix_name]["OUT_Q"] = activation_bits

    def save_params_conv(self, module):
        self.params[self.param_prefix_name.upper() + "_IM_CH"] = module.in_channels
        self.params[self.param_prefix_name.upper() + "_OUT_CH"] = module.out_channels

        # kernel has to be squared
        if isinstance(module.kernel_size, tuple):
            kernel = module.kernel_size[0]
        else:
            kernel = module.kernel_size
        self.params[self.param_prefix_name.upper() + "_KER_DIM"] = kernel

        if isinstance(module.padding, tuple):
            padding = module.padding[0]
        else:
            padding = module.padding
        self.params[self.param_prefix_name.upper() + "_PADDING"] = padding

        if isinstance(module.stride, tuple):
            stride = module.stride[0]
        else:
            stride = module.stride
        self.params[self.param_prefix_name.upper() + "_STRIDE"] = stride

        self.params[self.param_prefix_name.upper() + "_IM_DIM"] = module.input_shape[-1]
        self.params[self.param_prefix_name.upper() + "_OUT_DIM"] = module.output_shape[-1]

        col_buffer = 2 * module.in_channels * kernel * kernel
        if self.max_col_buffer < col_buffer:
            self.max_col_buffer = col_buffer
            self.params["MAX_CONV_BUFFER_SIZE"] = self.max_col_buffer

    def save_params_pool(self, module, previous_module, parents):
        if isinstance(module, (nn.AdaptiveAvgPool2d)):
            input_size = parents[previous_module].output_shape[-1]
            output_size = module.output_size
            stride = input_size//output_size
            # in_ch = parents[previous_module].in_channels
            kernel = parents[previous_module].kernel_size
            self.params[self.param_prefix_name.upper() + "_KER_DIM"] = input_size - (output_size - 1) * stride
            self.params[self.param_prefix_name.upper() + "_PADDING"] = 0
            self.params[self.param_prefix_name.upper() + "_STRIDE"] = stride
            out_ch = parents[previous_module].out_channels #previous out channel to be avg module input channel
            self.params[self.param_prefix_name.upper() + "_IM_DIM"] = input_size
            self.params[self.param_prefix_name.upper() + "_IM_CH"] = out_ch
            self.params[self.param_prefix_name.upper() + "_OUT_DIM"] = output_size
            print("??")
            return
        # kernel has to be squared
        self.params[self.param_prefix_name.upper() + "_IM_CH"] = module.input_shape[1] #_IM_CH
        if isinstance(module.kernel_size, tuple):
            kernel = module.kernel_size[0]
            self.params[self.param_prefix_name.upper() + "_KER_DIM_H"] = module.kernel_size[0]
            self.params[self.param_prefix_name.upper() + "_KER_DIM_W"] = module.kernel_size[1]
        else:
            kernel = module.kernel_size
            self.params[self.param_prefix_name.upper() + "_KER_DIM"] = kernel

        if isinstance(module.padding, tuple):
            padding = module.padding[0]
            self.params[self.param_prefix_name.upper() + "_PADDING_H"] =  module.padding[0]
            self.params[self.param_prefix_name.upper() + "_PADDING_W"] =  module.padding[1]
        else:
            padding = module.padding
            self.params[self.param_prefix_name.upper() + "_PADDING"] = padding

        if isinstance(module.stride, tuple):
            stride = module.stride[0]
            self.params[self.param_prefix_name.upper() + "_STRIDE_H"] = module.stride[0]
            self.params[self.param_prefix_name.upper() + "_STRIDE_W"] = module.stride[1]
        else:
            stride = module.stride
            self.params[self.param_prefix_name.upper() + "_STRIDE"] = stride
        
        self.params[self.param_prefix_name.upper() + "_IM_DIM"] = module.input_shape[-1]
        self.params[self.param_prefix_name.upper() + "_OUT_DIM"] = module.output_shape[-1]

    def save_params_linear(self, module):
        self.params[self.param_prefix_name.upper() + "_OUT"] = module.out_features
        self.params[self.param_prefix_name.upper() + "_DIM"] = torch.prod(
            torch.tensor(module.input_shape[-1:])).item()

        if self.max_fc_buffer < self.params[self.param_prefix_name.upper() + "_DIM"]:
            self.max_fc_buffer = self.params[self.param_prefix_name.upper() + "_DIM"]
            self.params["MAX_FC_BUFFER"] = self.max_fc_buffer

#################################################################################################3
    def refine_model_weights(self, loader, bias=False, search_range=1):
        if bias:
            index_q = "BIAS_Q"
            index = ".bias"
        else:
            index_q = "WEIGHT_Q"
            index = ".weight"
        model_usage = copy.deepcopy(self.model_qq)
        model_usage = model_usage.to(self.device)
        best_accuracy = evaluate(model_usage, loader, self.device)
        for key in tqdm(self.q_params.keys(), desc="Refining " + index):
            q_ = self.q_params[key][index_q]
            # Search new Q-bits for the weights
            for new_q in range(q_ + search_range, self.weight_bits):
                model_usage.state_dict()[key + index].copy_(self.convert_saturate_deconvert(
                                                model_usage.state_dict()[key + index], new_q))
                new_accuracy = evaluate(model_usage, loader, self.device)
                model_usage = copy.deepcopy(self.model_qq)
                model_usage = model_usage.to(self.device)
                if new_accuracy > best_accuracy:
                    print("\nNew accuracy: ", new_accuracy, 
                          "Best accuracy: ", best_accuracy, 
                          "New Q: ", new_q, "Best Q: ", self.q_params[key][index_q])
                    best_accuracy = new_accuracy
                    self.q_params[key][index_q] = new_q
            self.model_qq.state_dict()[key + index].copy_(self.convert_saturate_deconvert(
                    self.model_qq.state_dict()[key + index], self.q_params[key][index_q]))

    def convert_saturate_deconvert(self, matrix, q):
        matrix = self.quantize_tensor(matrix, q)
        matrix[matrix > 126] = 127
        matrix[matrix < -127] = -128
        return self.dequantize_tensor(matrix, q)

    def dequantize_tensor(self, matrix, q):
        return matrix.type(torch.float32)/2**q

    def refine_activations(self, loader, search_range=3):
        for key in tqdm(self.q_params.keys(), desc="Refining activations"):
            q_ = self.q_params[key]["OUT_Q"]
            model_usage = copy.deepcopy(self.model_qq)
            model_usage = model_usage.to(self.device)
            best_accuracy = self.evaluate_modules(model_usage, loader, key, q_)
            for new_q in range(q_ + search_range, self.weight_bits):
                if new_q < 0:
                    continue
                new_accuracy = self.evaluate_modules(model_usage, loader, key, new_q)
                model_usage = copy.deepcopy(self.model_qq)
                model_usage = model_usage.to(self.device)
                if new_accuracy > best_accuracy:
                    print("\nNew accuracy: ", new_accuracy, 
                          "Best accuracy: ", best_accuracy, 
                          "New Q: ", new_q, "Best Q: ", self.q_params[key]["OUT_Q"])
                    best_accuracy = new_accuracy
                    self.q_params[key]["OUT_Q"] = new_q

    def evaluate_modules(self, model, loader, key, q_):
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = torch.squeeze(labels).to(self.device)
                out = inputs
                for name, module in model.named_children():
                    try:
                        out = module(out)
                    except:
                        pass
                    if name == key:
                        out = self.convert_saturate_deconvert(out, q_)
                _, preds = torch.max(out, -1)
                total += inputs.shape[0]
                correct += (preds == labels).sum().item()
        return correct/total

    def reassign_q_params_n_shifts(self):
        for module in self.model.named_children():
            self.param_prefix_name = module[0]
            if isinstance(module[1], (nn.Conv2d, nn.Linear)):
                self.change_q_params()

    def change_q_params(self):
        if self.connectivity[self.param_prefix_name] in self.q_params.keys():
            self.q_params[self.param_prefix_name]["INPUT_Q"] = self.q_params[
                self.connectivity[self.param_prefix_name]]["OUT_Q"]

        self.q_params[self.param_prefix_name]["BIAS_LSHIFT"] = (
            self.q_params[self.param_prefix_name]["INPUT_Q"] +
            self.q_params[self.param_prefix_name]["WEIGHT_Q"] -
            self.q_params[self.param_prefix_name]["BIAS_Q"]
        )
        self.q_params[self.param_prefix_name]["OUT_RSHIFT"] = (
            self.q_params[self.param_prefix_name]["INPUT_Q"] +
            self.q_params[self.param_prefix_name]["WEIGHT_Q"] -
            self.q_params[self.param_prefix_name]["OUT_Q"]
        )

    def write_shifts_n_params(self):
        with open(self.parameter_file_name, "w+") as w_file:
            for i, j in self.params.items():
                w_file.write("#define " + i + " " + str(j) + "\n")
            for i, j in self.q_params.items():
                for k, l in j.items():
                    w_file.write("#define " + i.upper() + "_" + k + " " + str(l) + "\n")

    def convert_weights(self):
        self.model.to('cpu')
        for module in self.model.named_children():
            self.param_prefix_name = module[0]
            if isinstance(module[1], nn.Conv2d):
                for param in module[1].named_parameters():
                    self.convert_conv_weight(param[0], param[1])
            if isinstance(module[1], nn.Linear):
                for param in module[1].named_parameters():
                    self.convert_linear_weight(module[0], param[0], param[1])
            if isinstance(module[1], (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                for param in module[1].named_parameters():
                    self.convert_pool_weight(param[0], param[1])

    def choose_bias_weight(self, tensor_name, weight):
        if tensor_name == "bias":
            name = self.param_prefix_name.upper() + "_BIAS"
            qweight = self.quantize_tensor(weight, self.q_params[self.param_prefix_name]["BIAS_Q"])
        if tensor_name == "weight":
            name = self.param_prefix_name.upper() + "_WT"
            qweight = self.quantize_tensor(weight, self.q_params[self.param_prefix_name]["WEIGHT_Q"])
        return name, qweight

    def convert_pool_weight(self, tensor_name, weight):
        name, qweight = self.choose_bias_weight(tensor_name, weight)
        self.write_weights(name, qweight.numpy().astype(np.int8))
    
    def convert_linear_weight(self, module_name, tensor_name, weight):
        name, qweight = self.choose_bias_weight(tensor_name, weight)
        if tensor_name == "bias":
            self.write_weights(name, qweight.numpy().astype(np.int8))
        if tensor_name == "weight":
            original_shape = qweight.shape
            if self.last_layer_name in module_name.lower():
                # Last Conv -> Linear Layer Decompose to Tensor size.
                # self.conv_linear_interface_shape -> 256, 1, 1
                # original_shape -> 2, 256
                trans_weight = (
                    qweight.reshape(
                        original_shape[0],
                        *tuple(
                            torch.tensor(self.conv_linear_interface_shape)
                            .numpy()
                            .tolist()
                        ),# 2, 256, 1, 1
                    )
                    .permute(0, 2, 3, 1) # 2, 1, 1, 256
                    .reshape(original_shape) # 2, 256
                )

                weight = convert_to_x4_q7_weights(
                    trans_weight.reshape( # 2, 256
                        original_shape[0], original_shape[1], 1, 1
                    )
                    .numpy()
                    .astype(np.int8)
                )
            else:
                weight = convert_to_x4_q7_weights(
                    qweight.reshape(
                        original_shape[0], original_shape[1], 1, 1
                    )
                    .numpy()
                    .astype(np.int8)
                )
            self.write_weights(name, weight)
        
    def convert_conv_weight(self, tensor_name, weight):
        name, qweight = self.choose_bias_weight(tensor_name, weight)
        if tensor_name == "bias":
            self.write_weights(name, qweight.numpy().astype(np.int8))
        if tensor_name == "weight":
            self.write_weights(
                    name, qweight.permute(0, 2, 3, 1).numpy().astype(np.int8)
                )

    def write_weights(self, name, weight):
        with open(self.weight_file_name, "a") as w_file:
            w_file.write("#define " + name + " {")
            weight.tofile(w_file, sep=",")
            w_file.write("}\n")
            w_file.write("#define " + name + "_SHAPE ")
            w_file.write(str(np.prod(weight.shape)))
            w_file.write("\n")

    def compile(self):
        call(self.compilation, cwd=self.root, shell=True)

    def execute(self, exec_path):
        call(exec_path, cwd=self.root)
        # TODO: this implies that the executable produces this file
        return np.fromfile(os.path.join(self.io_folder, f"{self.last_layer_name}_out.raw"), dtype=np.int8)
    
    def evaluate_cmsis(self, exec_path:str, loader:DataLoader):
        correct = 0
        total = 0
        self.compile()
        from collections import defaultdict
        from sklearn.metrics import classification_report
        cfm = defaultdict(list)
        for input_batch, label_batch in tqdm(loader, total=len(loader)):
            for inp, label in zip(input_batch, label_batch):
                self.quantize_input(inp) # Save input tensor(Quantization format) for Inference
                out = self.execute(exec_path) # Load y_out.raw Inference using C++ code 
                pred = np.argmax(out)
                cfm['y_pred'].append(pred)
                cfm['y_true'].append(label.item())
                correct += pred == label.item()
                total += 1
                break
        print(f"Confusion Matrix: {classification_report(y_true=cfm['y_true'], y_pred=cfm['y_pred'])}")
        print(f"Test accuracy for CMSIS model {correct / total}")

###################################################################################################

    def write_logging(self):
        for i, j in self.logging.items():
            j.tofile(
                os.path.join(self.io_folder, str(i).lower() + "_torch.raw")
            )

    def register_logging(self):
        for module in self.model.named_children():
            if isinstance(module[1], (nn.Conv2d, nn.Linear)):
                self.param_prefix_name = module[0]
                self.logging[self.param_prefix_name + "_OUT"] = \
                    self.quantize_tensor(
                        module[1].output,
                        self.q_params[self.param_prefix_name]["OUT_Q"]).numpy()
        self.write_logging()

    def sample_inference_checker(self, exec_path, inp, draw=False):
        self.compile()
        self.quantize_input(inp[0])
        out = self.execute(exec_path)
        out_torch = self.model(inp)[0]
        self.register_logging()
        self.draw_model_comparison(draw)

    def draw_model_comparison(self, draw=False):
        for module in self.model.named_children():
            self.param_prefix_name = module[0]
            if isinstance(module[1], (nn.Conv2d, nn.Linear)):
                if draw:
                    self.draw_activation(
                        os.path.join(
                            self.io_folder,
                            self.param_prefix_name.lower() + "_out_torch.raw"),
                        os.path.join(
                            self.io_folder,
                            self.param_prefix_name.lower() + "_out.raw")
                    )

    def draw_activation(self, torch_activation_name, cmsis_activation_name):
        torch_activation = np.sort(np.fromfile(torch_activation_name, dtype=np.int8))
        cmsis_activation = np.sort(np.fromfile(cmsis_activation_name, dtype=np.int8))
        label = torch_activation_name.split("_")[0]
        plt.plot(torch_activation, label="PyTorch " + self.param_prefix_name.lower(), c='k')
        # plt.hist(torch_activation, bins=16, alpha=0.5, edgecolor='black', color='k', label="PyTorch " + label)
        plt.plot(cmsis_activation, label="CMSIS-NN " + self.param_prefix_name.lower(), c='r')
        # plt.hist(cmsis_activation, bins=16, alpha=0.5,  edgecolor='black', c='r', label="CMSIS-NN " + label, )
        plt.legend()
        plt.savefig(torch_activation_name.split(".")[0] + ".png")
        plt.clf()
        # plt.show()
    
    def sample_inference_diff(self, batch_x, batch_y, data_len=10, draw=False):
        def list_bracket(lists):
            return "{" + ', '.join(map(str, lists)) + "}"
        import os
        out_torch = self.model(batch_x)
        # with open(os.path.join(path, 'inputs.h'), 'w') as f:
        from collections import defaultdict
        CONFIG= defaultdict(str)
        CONFIG['c'] = {}
        CONFIG['c']['path'] = "E:/2_Quantization/deployment-with-CMSIS-NN/CMSIS_NN_PC_simulator/Deploy_Simulator"
        CONFIG['c']['file'] = "inputs.h"
        
        for idx, (inp, label) in enumerate(zip(tqdm(batch_x), batch_y)):
            if idx >= data_len:
                break
            if inp.is_cuda:
                inp = inp.cpu()
                label = label.cpu()
            qtensor = self.extract_q_quantize_tensor(inp).permute(1, 2, 0).numpy().astype(np.int8)
            qtensor.tofile(os.path.join(self.io_folder, "input.raw"))
            # CONFIG['c']['content'] = f"#define INPUT_DATA_SHAPE_{idx} {list_bracket(qtensor.shape)}\n"
            CONFIG['c']['content'] += f"#define INPUT_Y_DATA_{idx} {label.item()}\n"
            CONFIG['c']['content'] += f"#define INPUT_X_DATA_{idx} {list_bracket(qtensor.tolist())}\n"
            CONFIG['result'] += f"Index: {idx}, Expected: {label.item()} | Predicted: {torch.argmax(out_torch[idx]).item()}\n"
            
            for module in self.model.named_children():
                if isinstance(module[1], (nn.Conv2d, nn.Linear)):
                    self.param_prefix_name = module[0]
                    self.logging[self.param_prefix_name + "_OUT"] = \
                        self.quantize_tensor(
                            module[1].output,
                            self.q_params[self.param_prefix_name]["OUT_Q"]).numpy()
            
            for i, j in self.logging.items():
                j.tofile(os.path.join(self.io_folder, f"{str(i).lower()}_torch.raw"))
    
def hook_save_params(module, input, output):
    setattr(module, "input_shape", input[0].shape)
    setattr(module, "output_shape", output[0].shape)
    setattr(module, "input", input[0][0])
    setattr(module, "output", output[0])


def register_hooks(model:nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, (QConv2d, QLinear, nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            module.register_forward_hook(hook_save_params)



# def draw_activation(torch_activation_name, cmsis_activation_name):
#     torch_activation = np.sort(np.fromfile(torch_activation_name, dtype=np.int8))
#     cmsis_activation = np.sort(np.fromfile(cmsis_activation_name, dtype=np.int8))
#     label = torch_activation_name.split("_")[0]
#     plt.plot(torch_activation, label="PyTorch " + label, c='k')
#     plt.plot(cmsis_activation, label="CMSIS-NN " + label, c='r')
#     plt.legend()
#     plt.savefig(torch_activation_name.split(".")[0] + ".png")
#     plt.clf()
#     # plt.show()

def evaluate(model, loader, device):
    model.eval().to(device)
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = torch.squeeze(labels).to(device)
            out = model(inputs)
            _, preds = torch.max(out, -1)
            total += inputs.shape[0]
            correct += (preds == labels).sum().item()
    return correct/total

