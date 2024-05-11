import torch
import torch.nn as nn
import numpy as np


if __name__ == '__main__':
    torch.manual_seed(0)
    KH = 3
    KW = 3
    Ci = 16
    Co = 32
    # a = torch.randint(low = -128, high=127, size = (Co, Ci, KH, KW))
    # a = a.permute(1, 2, 0, 3)
    # b = a.reshape(Co, Ci, -1)
    # print(a.shape)
    # C, H, W
    a =torch.arange(1, Co*KW*KH*Ci+1)
    a = a.reshape(Co, KH, KW, Ci) #Co, H, W, Ci #Conv2d Format
    #
    print("----------NHWC----------")
    print(a.shape)
    print(a)
    print("-"*10)
    HWNC_weight = a.permute(1, 2, 0, 3).numpy() #H, W, Co, Ci
    print(HWNC_weight.shape)
    print(HWNC_weight)
    # HW, Co, Ci 16*2, 8->4 
    # tmp_list = [HWNC_weight[:,:,:,:4]]
    #version 1
    
    # print("HWNC4------------")
    #version1 = np.concatenate([i for i in np.split(HWNC_weight, Ci//4, axis=3)], axis=2)
    
    # print("Version1-----------")
    # print(version1)
    # print(version1.shape)
    
    #version 2
    version2 = HWNC_weight.reshape(KH, KW, Co*(Ci//4), 4)
    
    print("Version2-----------")
    print(version2)
    
    print(version2.shape)
    exit()
    # c = a.reshape(Co//2, KH, KW, Ci*2)
    # # c = np.transpose(c, (1, 2, 0, 3))
    # print(c)
    # a =torch.arange(1, Co*KW*KH*Ci+1).reshape(KH, KW, Co, Ci).repeat(1, 1, 1, Ci)
    # a = a.permute(1, 2, 0, 3)
    # print(a)
    # print(a.shape)
    # HWC
    # CoCiHW
    #Co//4, 4, KH, KW, Ci
    # CHW 
    # wt = nn.Conv2d(Ci, Co, (KH, KW), bias=True, padding=1, stride=1)
    # print(wt.weight.shape)
    # weight = wt.weight.reshape(32//4, 4, Ci, KH, KW).permute(0, 1, 4, 3, 2)
    # print(weight.shape)
    # print(wt.weight)
    # a = a.reshape(32//4, 4, Ci, KH, KW) #Co, Ci, H, W #Conv2d Format
    # a = a.permute(0, 1, 4, 3, 2) #Co//4, 4, H, W, Ci
    # a = a.reshape(-1) 
    # 
    # print(a)
    # a = (32, 3, 3, 3).reshape(32//4,4, 3, 3, 3).permute(0, 3, 4, 2, 1)
    # print(a)
    # print(a.reshape(-1))