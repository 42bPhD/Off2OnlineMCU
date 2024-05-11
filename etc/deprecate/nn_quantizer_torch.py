# Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.
# 
# SPDX-License-Identifier: Apache-2.0
# 
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# NN-Quantizer for Caffe models
# Include <Caffe installation path>/python in PYTHONPATH environment variable 
# import caffe
# from caffe.proto import caffe_pb2
import os
import torch
import pickle
import copy
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
# import torch.functional as F
from torch.functional import F 
import toml
from hook import net_info, ddict


def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return net_info(m)
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output


class torch_Quantizer(object):
    """\
    Quantize a trained caffe model to 8-bits 
    """
    def __init__(self,
                model:nn.Module,
                Qconfig:str=''):
        self.config = Qconfig
        self.model = model
        self.hook_m = copy.deepcopy(model)
        self.accuracy_layer=config['target_accuracy_layer']
        self.epochs = config['batch_size']
        self.weight_file = config['weight_file']
        
        self.quant_weight_file=""

        self.start_layer=[]
        self.end_layer=[]
        self.layer=[]
        self.layer_shape={}

        self.num_ops={}
        self.num_wts={}
        self.wt_int_bits={}
        self.wt_dec_bits={}
        self.bias_int_bits={}
        self.bias_dec_bits={}
        self.act_int_bits={}
        self.act_dec_bits={}
        self.bias_lshift={}
        self.act_rshift={}
        
        self.model_info = ddict()
        self.data_loader:DataLoader = None
        
        

    def save_quant_params(self, model_info_file):
        with open(model_info_file, 'w') as f:
            toml.dump(self, f)

    def load_quant_params(self, model_info_file):
        #Model Parameter
        model_par=pickle.load(open(model_info_file,'rb'))
        with open(model_info_file, 'r') as f:
            model_par = toml.load(f)
        self.model_file = model_par['model_file']
        #Weight file (pth)
        self.weight_file=model_par['weight_file']
        
        # Quantized weight file (h5)
        self.quant_weight_file=model_par['quant_weight_file']
        # Convolutional Layer
        
        self.layer = model_par['layer']
        
        self.num_ops=model_par['num_ops']
        self.num_wts=model_par['num_wts']
        
        self.wt_int_bits=model_par['wt_int_bits']
        self.wt_dec_bits=model_par['wt_dec_bits']
        self.bias_int_bits=model_par['bias_int_bits']
        self.bias_dec_bits=model_par['bias_dec_bits']
        self.act_int_bits=model_par['act_int_bits']
        self.act_dec_bits=model_par['act_dec_bits']
        
        self.bias_lshift=model_par['bias_lshift']
        self.act_rshift=model_par['act_rshift']
        
        self.accuracy_layer=model_par['accuracy_layer']
        self.epochs=model_par['epoch']
    
    @torch.no_grad()
    def run_full_network(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        net = copy.deepcopy(self.model)
        self.model.load_state_dict(net)
        self.model.to(self.device)
        test_loss = 0
        correct = 0 
        for data, target in self.data_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(self.data_loader.dataset)
        accuracy = 100. * correct / len(self.data_loader.dataset)
        
        print("Full precision accuracy: %.2f%%" %(accuracy))
        return accuracy

    def run_quantized_network(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        net = torch.load(self.model_file)
        self.model.load_state_dict(net)
        self.model.to(self.device)
        acc = np.zeros(self.epochs)
        with torch.no_grad():
            for i in range(self.epochs):
                out = self.model.forward()
                acc[i] = out[self.accuracy_layer]*100
            print("Accuracy with quantized weights/biases: %.2f%%" %(acc.mean()))
        for i in range(self.epochs):
            for layer_no in range(0,len(self.start_layer)):
                if layer_no==0:
                    net.forward(end=str(self.end_layer[layer_no]))
                else:
                    net.forward(start=str(self.start_layer[layer_no]),end=str(self.end_layer[layer_no]))
                if layer_no < len(self.start_layer)-1: # not quantizing accuracy layer
                    net.blobs[self.end_layer[layer_no]].data[:]=np.floor(net.blobs[self.end_layer[layer_no]].data*\
                        (2**self.act_dec_bits[self.end_layer[layer_no]]))
                    net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data>126]=127
                    net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data<-127]=-128
                    net.blobs[self.end_layer[layer_no]].data[:]=net.blobs[self.end_layer[layer_no]].data/\
                        (2**self.act_dec_bits[self.end_layer[layer_no]])
            acc[i] = net.blobs[self.accuracy_layer].data*100
        accuracy = acc.mean()
        print("Accuracy with quantized weights/biases and activations: %.2f%%" %(accuracy))
        return accuracy
    
    @torch.no_grad()
    def get_layer_info(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        net = torch.load(self.weight_file)
        self.hook_m.load_state_dict(net)
        self.hook_m.to(self.device)
        
        parse = nested_children(self.hook_m)
        # parse = get_layers(self.hook_m)
        inp = torch.randn(self.config['input_shape']).to(self.device)
        self.hook_m(inp)
        self.model_info = parse

    #   Quantize weights to 8 bits
    #       Using min and max of weights as nearest power of 2, quantize to 8bits (QM.N) and check accuracy
    #       If accuracy is lost, try QM-1:N+1, QM-2,N+2,... with saturation to find out the best combination
    #           with least accuracy loss (Trading-off weights that occur infrequently for more precision)
    #
    #                     -2^(M+N)		0	   2^(M+N)
    #			 |		^             |
    #			 |	      *|||*	      |
    #		<--------|	     *|||||*          |------->
    #	        Saturated|	    *|||||||*         |Saturated
    #			 |	  *|||||||||||*       |
    #			 |     *|||||||||||||||||*    |
    #			*|			      |*
    #	*		 |<-------------------------->|			*
    #		             Weight quantization and  
    #                            truncation with minimal
    #			        loss of accuracy
    #
    @torch.no_grad()
    def validation(self, net, name, layer):
        test_loss = 0
        accuracy = 0
        for i in range(0,self.epochs):
            for inputs, targets in self.data_loader:
                outputs:torch.Tensor = net(inputs)
                test_loss += F.cross_entropy(outputs, targets, reduction='sum').item()
                pred = outputs.data.max(1, keepdim=True)[1]
                correct = pred.eq(targets.data.view_as(pred)).sum().item()
                accuracy += correct
        test_loss /= len(self.data_loader.dataset)
        accuracy /= len(self.data_loader.dataset)
        return accuracy
    
    def quantization(self, layer_name, net, search_range=3, tolerance=0.001):
        """ q7 format으로 변경한 뒤에 다시 복구하여 target_accuracy(원래 정확도)와 비교하여
            tolerance보다 작으면 그때의 정확도를 반환한다.

        Args:
            layer_name (_type_): _description_
            net (_type_): _description_
            search_range (int, optional): _description_. Defaults to 3.
            tolerance (float, optional): _description_. Defaults to 0.001.
        """
        
        #Start with min/max of weights to the rounded up to nearest power of 2.
        wt_max = net['weight'].data.max()
        wt_min = net['bias'].data.min()
        # Quantize to nearest power of 2
        self.wt_int_bits[layer_name] = int(np.ceil(np.log2(max(abs(wt_min),abs(wt_max))))) 
        # 8 bit quantization get 7 fractional bits
        self.wt_dec_bits[layer_name] = 7-self.wt_int_bits[layer_name] #8 bit quantization get 7 fractional bits
        max_int_bits = self.wt_int_bits[layer_name]-search_range 
        print('Layer: '+ layer_name + ' weights max: '+str(wt_max)+' min: '+str(wt_min)+\
            ' Format: Q'+str(self.wt_int_bits[layer_name])+'.'+str(self.wt_dec_bits[layer_name]))
        #Quantization to QM.N format
        net['weight']=np.round(net['weight']*(2**self.wt_dec_bits[layer_name]))/(2**self.wt_dec_bits[layer_name])
        accuracy = self.validation(net, self.data_loader, self.accuracy_layer)
        
        print("Accuracy: %.2f%%" %(accuracy))
        best_int_bits = self.wt_int_bits[layer_name]
        best_dec_bits = self.wt_dec_bits[layer_name]
        best_accuracy = accuracy
        while target_accuracy-accuracy>tolerance and self.wt_int_bits[layer_name] > max_int_bits:
            #Quantize weights to QM-1.N+1
            self.wt_int_bits[layer_name] = self.wt_int_bits[layer_name]-1
            self.wt_dec_bits[layer_name] = self.wt_dec_bits[layer_name]+1
            
            net.copy_from(self.quant_weight_file)
            
            #DEQuantization
            net.params[layer_name][0].data[:]=np.round(net.params[layer_name][0].data*\
                (2**self.wt_dec_bits[layer_name]))
            #CLIP
            net.params[layer_name][0].data[net.params[layer_name][0].data>126]=127
            net.params[layer_name][0].data[net.params[layer_name][0].data<-127]=-128
            # Dequantization
            net.params[layer_name][0].data[:]=net.params[layer_name][0].data/(2**self.wt_dec_bits[layer_name])
            for i in range(0,self.iterations):
                out = net.forward()
                acc[i] = out[self.accuracy_layer]*100
            accuracy = acc.mean()
            print('Format Q'+str(self.wt_int_bits[layer_name])+'.'+\
                str(self.wt_dec_bits[layer_name])+' Accuracy: %.2f%%' %(accuracy))
            if accuracy>best_accuracy:
                best_int_bits = self.wt_int_bits[layer_name]
                best_dec_bits = self.wt_dec_bits[layer_name]
                best_accuracy = accuracy
        self.wt_int_bits[layer_name] = best_int_bits
        self.wt_dec_bits[layer_name] = best_dec_bits
        net.copy_from(self.quant_weight_file)
        net.params[layer_name][0].data[:]=np.round(net.params[layer_name][0].data*\
            (2**self.wt_dec_bits[layer_name]))
        net.params[layer_name][0].data[net.params[layer_name][0].data>126]=127
        net.params[layer_name][0].data[net.params[layer_name][0].data<-127]=-128
        net.params[layer_name][0].data[:]=net.params[layer_name][0].data/\
            (2**self.wt_dec_bits[layer_name])
        print('Final '+layer_name+ ' weights format Q'+str(best_int_bits)+'.'+\
            str(best_dec_bits)+' Accuracy: %.2f%%' %(best_accuracy))
        net.save(self.quant_weight_file)
    
    @torch.no_grad()
    def quantize_wts_8bit(self, tolerance=0.001, search_range=3):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        net = self.model_info
        
        test_loss = 0
        correct = 0 
        print(net.keys())
        """
        for inputs, targets in self.data_loader:
            outputs:torch.Tensor = self.model(inputs)
            test_loss += F.cross_entropy(outputs, targets, reduction='sum').item()
            pred = outputs.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).sum().item()
        
        target_accuracy = 100. * correct / len(self.data_loader.dataset)
        print("Full precision accuracy: %.2f%%" %(target_accuracy))
        """
        self.quant_weight_file = self.weight_file.split('.')[0]
        self.quant_weight_file = os.path.join(self.config['save_path'],
                                            f"{self.config['model']}_{self.quant_weight_file}_quantized.pth")
        # quantization파일 저장
        torch.save(self.model.state_dict(), self.quant_weight_file)
        # 모던 레이어 정보 수집
        from collections import defaultdict
        def get_layers(name, layer):
            for n, m in layer.items():
                #dict is end of the layer
                if isinstance(m, (dict)): 
                    yield f'{name}.{n}', m
                else: #Default dict means there are more layers
                    yield from get_layers(f'{name}.{n}', m)
        
        # Conv Layer와 FC Layer만 추출
        # for layer_name in self.conv_layer+self.ip_layer:
        for layer_name, layer in self.model_info.items():
            #Start with min/max of weights to the rounded up to nearest power of 2.
            q_module = None
            if isinstance(layer, defaultdict):
                if layer['type'] == 'conv' and layer['type'] == 'fc':
                    q_module = copy.deepcopy(layer)
                else:
                    q_module = None
            #Quantization
            elif isinstance(layer, dict):
                for n_m in get_layers(layer_name, layer):
                    n, m = n_m[0], n_m[1]
                    if m['type'] == 'conv' and m['type'] == 'fc':
                        q_module = copy.deepcopy(m)
                    else:
                        q_module = None
                    ## Quantization
                    wt_max = q_module['weight'].max()
                    wt_min = q_module['weight'].data.min()
                    # Quantize to nearest power of 2
                    self.wt_int_bits[n] = int(np.ceil(np.log2(max(abs(wt_min),abs(wt_max))))) 
                    # 8 bit quantization get 7 fractional bits
                    self.wt_dec_bits[n] = 7-self.wt_int_bits[n] #8 bit quantization get 7 fractional bits
                    max_int_bits = self.wt_int_bits[n]-search_range 
            else:
                q_module = None
        #! TODO: quantization
        ################### 해야됨 ######################
        
        for layer_name in self.conv_layer+self.ip_layer:
            #Start with min/max of weights to the rounded up to nearest power of 2.
            wt_max = net.params[layer_name][0].data.max()
            wt_min = net.params[layer_name][0].data.min()
            self.wt_int_bits[layer_name] = int(np.ceil(np.log2(max(abs(wt_min),abs(wt_max)))))
            self.wt_dec_bits[layer_name] = 7-self.wt_int_bits[layer_name]
            max_int_bits = self.wt_int_bits[layer_name]-search_range
            print('Layer: '+ layer_name + ' weights max: '+str(wt_max)+' min: '+str(wt_min)+\
                ' Format: Q'+str(self.wt_int_bits[layer_name])+'.'+str(self.wt_dec_bits[layer_name]))
            net.params[layer_name][0].data[:]=np.round(net.params[layer_name][0].data*\
                (2**self.wt_dec_bits[layer_name]))/(2**self.wt_dec_bits[layer_name])
            for i in range(0,self.iterations):
                out = net.forward()
                acc[i] = out[self.accuracy_layer]*100
            accuracy = acc.mean()
            print("Accuracy: %.2f%%" %(accuracy))
            best_int_bits = self.wt_int_bits[layer_name]
            best_dec_bits = self.wt_dec_bits[layer_name]
            best_accuracy = accuracy
            # full precision accuracy와 quantize accracy가 일정 범위보다 작으면 
            # W-bit를 조절해서 계속해서 정확도를 높인다.
            while target_accuracy-accuracy>tolerance and self.wt_int_bits[layer_name]>max_int_bits:
                #Quantize weights to QM-1.N+1
                self.wt_int_bits[layer_name] = self.wt_int_bits[layer_name]-1
                self.wt_dec_bits[layer_name] = self.wt_dec_bits[layer_name]+1
                #임시 저장한 것 가져옴.
                net.copy_from(self.quant_weight_file)
                #Quantization
                net.params[layer_name][0].data[:]=np.round(net.params[layer_name][0].data*(2**self.wt_dec_bits[layer_name]))
                #CLIP
                net.params[layer_name][0].data[net.params[layer_name][0].data>126]=127
                net.params[layer_name][0].data[net.params[layer_name][0].data<-127]=-128
                #Dequantization
                net.params[layer_name][0].data[:]=net.params[layer_name][0].data/(2**self.wt_dec_bits[layer_name])
                # Validation
                for i in range(0,self.iterations):
                    out = net.forward()
                    acc[i] = out[self.accuracy_layer]*100
                accuracy = acc.mean()
                print('Format Q'+str(self.wt_int_bits[layer_name])+'.'+\
                    str(self.wt_dec_bits[layer_name])+' Accuracy: %.2f%%' %(accuracy))
                # 정확도가 높아지면 저장
                if accuracy>best_accuracy:
                    best_int_bits = self.wt_int_bits[layer_name]
                    best_dec_bits = self.wt_dec_bits[layer_name]
                    best_accuracy = accuracy
            self.wt_int_bits[layer_name] = best_int_bits
            self.wt_dec_bits[layer_name] = best_dec_bits
            net.copy_from(self.quant_weight_file)
            # Quantization
            net.params[layer_name][0].data[:]=np.round(net.params[layer_name][0].data*(2**self.wt_dec_bits[layer_name]))
            # CLIP
            net.params[layer_name][0].data[net.params[layer_name][0].data>126]=127
            net.params[layer_name][0].data[net.params[layer_name][0].data<-127]=-128
            # Dequantizaiton
            net.params[layer_name][0].data[:]=net.params[layer_name][0].data/(2**self.wt_dec_bits[layer_name])
            print('Final '+layer_name+ ' weights format Q'+str(best_int_bits)+'.'+\
                str(best_dec_bits)+' Accuracy: %.2f%%' %(best_accuracy))
            net.save(self.quant_weight_file)

    #   Quantize activations (inter-layer data) to 8 bits
    #       Using min and max of activations as nearest power of 2, quantize to 8bits (QM.N) and check accuracy
    #       If accuracy is lost, try QM-1:N+1, QM-2,N+2,... with saturation to find out the best combination
    #           with least accuracy loss (Trading-off activations that occur infrequently for more precision)
    def quantize_activations_8bit(self,tolerance=0.001,search_range=3):
        # Load to GPU Devices
        if self.device==True:
            caffe.set_mode_gpu()
        net = caffe.Net(self.model_file,self.quant_weight_file,caffe.TEST)
        # Epochs
        acc = np.zeros(self.iterations)
        # Validation
        for i in range(0,self.iterations):
            out = net.forward()
            acc[i] = out[self.accuracy_layer]*100
        
        target_accuracy = acc.mean()
        print("Accuracy with quantized weights: %.2f%%" %(target_accuracy))
        max_val={}
        min_val={}
        quant_layer_flag={}
        # 각 레이어를 거치기 전 input/ 거친 후의 output activation의 각 레이어 최대값과 최소값 초기화를 함.
        for layer in self.end_layer:
            max_val[layer]=float('-inf')
            min_val[layer]=float('inf')
            quant_layer_flag[layer]=0
        # Finding min max for output of all layers
        # 각 레이어를 거치기 전 input/ 거친 후의 output activation의 각 레이어 최대값과 최소값을 구함.
        for i in range(0,self.iterations):
            #샘플 데이터셋 마다 다를 수 있으니 여러번 반복해서 최대값과 최소값을 구함.
            for layer_no in range(0,len(self.start_layer)):
                if layer_no==0:
                    net.forward(end=str(self.end_layer[layer_no]))
                else:
                    net.forward(start=str(self.start_layer[layer_no]),end=str(self.end_layer[layer_no]))
                layer_max = net.blobs[self.end_layer[layer_no]].data.max()
                layer_min = net.blobs[self.end_layer[layer_no]].data.min()
                if(layer_max>max_val[self.end_layer[layer_no]]):
                    max_val[self.end_layer[layer_no]]=layer_max
                if(layer_min<min_val[self.end_layer[layer_no]]):
                    min_val[self.end_layer[layer_no]]=layer_min
                #print("Running %s layer, max,min : %.2f,%.2f" %(self.end_layer[layer_no],layer_max,layer_min)) 
        max_int_bits={}
        for layer in self.end_layer:
            #정수부(decimal part)
            self.act_int_bits[layer] = int(np.ceil(np.log2(max(abs(max_val[layer]),abs(min_val[layer])))))
            #소수부(fraction part)
            self.act_dec_bits[layer] = 7-self.act_int_bits[layer]
            max_int_bits[layer]=self.act_int_bits[layer]-search_range
            print('Layer: '+layer+' max: '+ str(max_val[layer]) + ' min: '+str(min_val[layer])+ \
                ' Format: Q'+str(self.act_int_bits[layer])+'.'+str(self.act_dec_bits[layer]))
        quant_max_val={}
        quant_min_val={}
        for layer in self.end_layer:
            quant_max_val[layer]=float('-inf')
            quant_min_val[layer]=float('inf')
        
        #! 주의 마지막 레이어는 Quantize 하지 않음.
        for quant_layer_no in range(0,len(self.start_layer)-1): #No need to quantize accuracy layer
            quant_layer=self.end_layer[quant_layer_no]
            quant_layer_flag[quant_layer]=1
            #풀링 layer의 경우
            if((self.layer_type[quant_layer]=='pooling' or self.layer_type[quant_layer]=='17') and \
                    self.pool_type[quant_layer]==1):
                #이전레이어의 정수부와 소수부를 가져옴.
                prev_layer=self.end_layer[quant_layer_no-1]
                self.act_int_bits[quant_layer]=self.act_int_bits[prev_layer]
                self.act_dec_bits[quant_layer]=self.act_dec_bits[prev_layer]
                continue
            
            # Quantize activations의 사전준비로 quantize layer의 이전 레이어의 정수부와 소수부를 가져와서 특정 레이어만 Quantize한다.
            # quantize layer by layer
            for i in range(0,self.iterations):
                for layer_no in range(0,len(self.start_layer)): # 전부다함.
                    if layer_no==0:
                        # 처음 레이어의 경우.
                        net.forward(end=str(self.end_layer[layer_no]))
                    else:
                        # 첫번째 이상, 마지막 이하의 레이어의 경우.
                        net.forward(start=str(self.start_layer[layer_no]),
                                        end=str(self.end_layer[layer_no]))
                    # quantize incrementally layer by layer
                    if quant_layer_flag[self.end_layer[layer_no]]==1:
                        # quantize layer -> QM.N -> Deqantize Layer
                        net.blobs[self.end_layer[layer_no]].data[:]=np.floor(net.blobs[self.end_layer[layer_no]].data*\
                            (2**self.act_dec_bits[self.end_layer[layer_no]]))/(2**self.act_dec_bits[self.end_layer[layer_no]])
                    layer_max = net.blobs[self.end_layer[layer_no]].data.max()
                    layer_min = net.blobs[self.end_layer[layer_no]].data.min()
                    if(layer_max>quant_max_val[self.end_layer[layer_no]]):
                        quant_max_val[self.end_layer[layer_no]]=layer_max
                    if(layer_min<quant_min_val[self.end_layer[layer_no]]):
                        quant_min_val[self.end_layer[layer_no]]=layer_min
                acc[i] = net.blobs[self.accuracy_layer].data*100
            accuracy=acc.mean()
            print('Layer-'+quant_layer+' max: '+str(quant_max_val[quant_layer])+\
                ' min: '+str(quant_min_val[quant_layer])+' format: Q'+\
                str(self.act_int_bits[quant_layer])+'.'+str(self.act_dec_bits[quant_layer])+\
                ' accuracy: %.2f%%' %(acc.mean()))
            best_accuracy = accuracy
            best_int_bits = self.act_int_bits[quant_layer]
            best_dec_bits = self.act_dec_bits[quant_layer]
            
            # quantize layer by layer, quantize layer의 정확도가 tolerance보다 작으면
            # 정확도를 높이기 위해 정수부를 줄이고 소수부를 늘린다.
            while target_accuracy-accuracy>tolerance and self.act_int_bits[quant_layer] > max_int_bits[quant_layer]:
                for layer in self.end_layer:
                    quant_max_val[layer]=float('-inf')
                    quant_min_val[layer]=float('inf')
                self.act_int_bits[quant_layer] = self.act_int_bits[quant_layer]-1
                self.act_dec_bits[quant_layer] = self.act_dec_bits[quant_layer]+1
                
                for i in range(0,self.iterations):
                    for layer_no in range(0,len(self.start_layer)):
                        if layer_no==0:
                            net.forward(end=str(self.end_layer[layer_no]))
                        else:
                            net.forward(start=str(self.start_layer[layer_no]),end=str(self.end_layer[layer_no]))
                        if quant_layer_flag[self.end_layer[layer_no]]==1:	
                            # quantize layer -> QM.N -> Deqantize Layer
                            net.blobs[self.end_layer[layer_no]].data[:]=np.floor(net.blobs[self.end_layer[layer_no]].data*(2**self.act_dec_bits[self.end_layer[layer_no]]))
                            #CLIP
                            net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data>126]=127
                            net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data<-127]=-128
                            # Deq
                            net.blobs[self.end_layer[layer_no]].data[:]=net.blobs[self.end_layer[layer_no]].data/(2**self.act_dec_bits[self.end_layer[layer_no]])
                            
                        layer_max = net.blobs[self.end_layer[layer_no]].data.max()
                        layer_min = net.blobs[self.end_layer[layer_no]].data.min()
                        if(layer_max>quant_max_val[self.end_layer[layer_no]]):
                            quant_max_val[self.end_layer[layer_no]]=layer_max
                        if(layer_min<quant_min_val[self.end_layer[layer_no]]):
                            quant_min_val[self.end_layer[layer_no]]=layer_min
                    acc[i] = net.blobs[self.accuracy_layer].data*100
                accuracy=acc.mean()
                if accuracy>best_accuracy:
                    best_int_bits = self.act_int_bits[quant_layer]
                    best_dec_bits = self.act_dec_bits[quant_layer]
                    best_accuracy = accuracy
                print('Layer-'+quant_layer+' max: '+str(quant_max_val[quant_layer])+\
                    'min: '+str(quant_min_val[quant_layer])+' format: Q'+\
                    str(self.act_int_bits[quant_layer])+'.'+str(self.act_dec_bits[quant_layer])+\
                    ' accuracy: %.2f%%' %(acc.mean()))
            self.act_int_bits[quant_layer] = best_int_bits 
            self.act_dec_bits[quant_layer] = best_dec_bits 
            print('Layer-'+quant_layer+' final format: Q'+str(self.act_int_bits[quant_layer])+\
                '.'+str(self.act_dec_bits[quant_layer])+ ' accuracy: %.2f%%' %(best_accuracy))
    
    def quantize_bias_8bit(self,tolerance=0.001,search_range=3):
        if self.device==True:
            caffe.set_mode_gpu()
        net = caffe.Net(self.model_file,self.quant_weight_file,caffe.TEST)
        acc = np.zeros(self.iterations)
        for i in range(0,self.iterations):
            net.forward()
            acc[i] = net.blobs[self.accuracy_layer].data*100
        target_accuracy = acc.mean()
        print("Accuracy with quantized weights: %.2f%%" %(target_accuracy))
        for i in range(0,self.iterations):
            for layer_no in range(0,len(self.start_layer)):
                if layer_no==0:
                    net.forward(end=str(self.end_layer[layer_no]))
                else:
                    net.forward(start=str(self.start_layer[layer_no]),end=str(self.end_layer[layer_no]))
                if layer_no < len(self.start_layer)-1: # not quantizing accuracy layer
                    net.blobs[self.end_layer[layer_no]].data[:]=np.floor(net.blobs[self.end_layer[layer_no]].data*\
                        (2**self.act_dec_bits[self.end_layer[layer_no]]))
                    net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data>126]=127
                    net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data<-127]=-128
                    net.blobs[self.end_layer[layer_no]].data[:]=net.blobs[self.end_layer[layer_no]].data/\
                        (2**self.act_dec_bits[self.end_layer[layer_no]])
            acc[i] = net.blobs[self.accuracy_layer].data*100
        target_accuracy = acc.mean()
        print("Accuracy with quantized weights and activations: %.2f%%" %(target_accuracy))
        input_of={}
        for i in range (1,len(self.end_layer)):
            input_of[self.end_layer[i]]=self.end_layer[i-1]
        for layer_name in self.conv_layer+self.ip_layer:
            mac_dec_bits = self.wt_dec_bits[layer_name]+self.act_dec_bits[input_of[layer_name]]
            bias_max = net.params[layer_name][1].data.max()
            bias_min = net.params[layer_name][1].data.min()
            int_bits = int(np.ceil(np.log2(max(abs(bias_min),abs(bias_max)))))
            dec_bits = 7-int_bits
            max_int_bits = int_bits-search_range
            if(dec_bits>mac_dec_bits):
                dec_bits=mac_dec_bits
                int_bits=7-dec_bits
                max_int_bits=int_bits #can't increase dec_bits any more as they will be shifted right anyway
            print('Layer: '+ layer_name + ' biases max: '+str(bias_max)+' min: '+str(bias_min)+\
                ' Format: Q'+str(int_bits)+'.'+str(dec_bits))
            net.params[layer_name][1].data[:]=np.round(net.params[layer_name][1].data*(2**dec_bits))/(2**dec_bits)
            for i in range(0,self.iterations):
                for layer_no in range(0,len(self.start_layer)):
                    if layer_no==0:
                        net.forward(end=str(self.end_layer[layer_no]))
                    else:
                        net.forward(start=str(self.start_layer[layer_no]),end=str(self.end_layer[layer_no]))
                    if layer_no < len(self.start_layer)-1: # not quantizing accuracy layer
                        net.blobs[self.end_layer[layer_no]].data[:]=np.floor(net.blobs[self.end_layer[layer_no]].data*\
                            (2**self.act_dec_bits[self.end_layer[layer_no]]))
                        net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data>126]=127
                        net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data<-127]=-128
                        net.blobs[self.end_layer[layer_no]].data[:]=net.blobs[self.end_layer[layer_no]].data/\
                            (2**self.act_dec_bits[self.end_layer[layer_no]])
                acc[i] = net.blobs[self.accuracy_layer].data*100
            accuracy = acc.mean()
            print("Accuracy: %.2f%%" %(accuracy))
            best_int_bits = int_bits
            best_dec_bits = dec_bits
            best_accuracy = accuracy
            while target_accuracy-accuracy>tolerance and int_bits>max_int_bits:
                int_bits = int_bits-1
                dec_bits = dec_bits+1
                net.copy_from(self.quant_weight_file)
                net.params[layer_name][1].data[:]=np.round(net.params[layer_name][1].data*(2**dec_bits))
                net.params[layer_name][1].data[net.params[layer_name][1].data>126]=127
                net.params[layer_name][1].data[net.params[layer_name][1].data<-127]=-128
                net.params[layer_name][1].data[:]=net.params[layer_name][1].data/(2**dec_bits)
                for i in range(0,self.iterations):
                    for layer_no in range(0,len(self.start_layer)):
                        if layer_no==0:
                            net.forward(end=str(self.end_layer[layer_no]))
                        else:
                            net.forward(start=str(self.start_layer[layer_no]),end=str(self.end_layer[layer_no]))
                        if layer_no < len(self.start_layer)-1: # not quantizing accuracy layer
                            net.blobs[self.end_layer[layer_no]].data[:]=np.floor(net.blobs[self.end_layer[layer_no]].data*\
                                (2**self.act_dec_bits[self.end_layer[layer_no]]))
                            net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data>126]=127
                            net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data<-127]=-128
                            net.blobs[self.end_layer[layer_no]].data[:]=net.blobs[self.end_layer[layer_no]].data/\
                                (2**self.act_dec_bits[self.end_layer[layer_no]])
                    acc[i] = net.blobs[self.accuracy_layer].data*100
                accuracy = acc.mean()
                print('Format Q'+str(int_bits)+'.'+str(dec_bits)+' Accuracy: %.2f%%' %(accuracy))
                if accuracy>best_accuracy:
                    best_int_bits = int_bits
                    best_dec_bits = dec_bits
                    best_accuracy = accuracy
            self.bias_int_bits[layer_name] = best_int_bits
            self.bias_dec_bits[layer_name] = best_dec_bits
            self.bias_lshift[layer_name]=mac_dec_bits-best_dec_bits
            self.act_rshift[layer_name]=mac_dec_bits-self.act_dec_bits[layer_name]
            net.copy_from(self.quant_weight_file)
            net.params[layer_name][1].data[:]=np.round(net.params[layer_name][1].data*(2**best_dec_bits))
            net.params[layer_name][1].data[net.params[layer_name][1].data>126]=127
            net.params[layer_name][1].data[net.params[layer_name][1].data<-127]=-128
            net.params[layer_name][1].data[:]=net.params[layer_name][1].data/(2**best_dec_bits)
            print('Final '+layer_name+ ' biases format Q'+str(best_int_bits)+'.'+str(best_dec_bits)+\
                ' Accuracy: %.2f%%' %(best_accuracy))
            net.save(self.quant_weight_file)
from PIL import Image
from torchvision import transforms

if __name__ == '__main__':
    # 이걸 왜 하는가?
    # 현재 IMG Q bit를 구하긴 했는데, (weight quantization)
    # Shift bit를 알기 위해서 빠르게 번역중
    
    
    # parser.add_argument('--tolerance', type=float, default=0.001,
    #         help='accuracy tolerance')
    filename = "./img/dog.jpg"
    save_path = './models/'
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    with open('./config.toml', 'r') as f:
        config = toml.load(f)
    # "models/cifar10_m4_train_test.prototxt"
    model:nn.Module = torch.hub.load('pytorch/vision:v0.10.0', 
                                    config['model'], 
                                    pretrained=False)
    if torch.cuda.is_available():
        gpu_flag = config['gpu']
    else:
        gpu_flag = False
    weight_file= config['weight_file']
    target_accuracy_layer= config['target_accuracy_layer']
    iterations= config['batch_size']
    # tolerance=accuracy_tolerance
    #Import torch hub
    from torch.hub import load_state_dict_from_url
    my_model= torch_Quantizer(model = model,
                            Qconfig = config)
    my_model.get_layer_info()
    # my_model.run_full_network()
    
    #First quantize weights to 8 bits
    my_model.quantize_wts_8bit()
    #Then quantize activations to 8 bits
    my_model.quantize_activations_8bit()
    #Quantize biases to 8 bits based on the quantization outputs of weights and activations
    my_model.quantize_bias_8bit()
    my_model.run_quantized_network()
    exit()
    my_model.save_quant_params(cmd_args.save)
    #To load the parameters use the following:
    #my_model.load_quant_params('mymodel.p')
 
    #Print dataformats
    print('Input: '+my_model.data_layer+' Q'+str(my_model.act_int_bits[my_model.data_layer])+'.'+\
        str(my_model.act_dec_bits[my_model.data_layer])+'(scaling factor:'+\
        str(2**(my_model.act_dec_bits[my_model.data_layer]))+')')
    for layer in my_model.conv_layer+my_model.ip_layer:
        print('Layer: '+layer+' Q'+str(my_model.act_int_bits[layer])+'.'+str(my_model.act_dec_bits[layer])+\
            ' (scaling factor:'+str(2**(my_model.act_dec_bits[layer]))+') Wts: Q'+\
            str(my_model.wt_int_bits[layer])+'.'+str(my_model.wt_dec_bits[layer])+\
            ' (scaling factor:'+str(2**(my_model.wt_dec_bits[layer]))+') Biases: Q'+\
            str(my_model.bias_int_bits[layer])+'.'+str(my_model.bias_dec_bits[layer])+\
            '(scaling factor:'+str(2**(my_model.bias_dec_bits[layer]))+')')
 
    #Print data shifts to be used by ML kernels
    for layer in my_model.conv_layer+my_model.ip_layer:
        print('Layer: '+layer+' bias left shift: '+str(my_model.bias_lshift[layer])+\
            ' act_rshift: '+str(my_model.act_rshift[layer]))
 
