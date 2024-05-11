import torch
from torch import nn

from quantization.fold import fold_CBR
from quantization import (Conv2d, 
                          BatchNorm2d, 
                          ReLU, 
                          MaxPool2d, 
                          Linear)
from quantization.qformat import QConv2d, QLinear, fold_CBR_qfmt
# Conv2d = QConv2d
# Linear = QLinear
# ReLU = ReLU6

class LeNet(nn.Module):
    # ARM Example Model
    def __init__(self, num_channels=3, num_classes=10, model='arm', qat=True):
        super(LeNet,self).__init__()
        
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=5, stride=1, padding=2)\
                        if not qat else Conv2d(in_channels=num_channels, out_channels=32, kernel_size=5, stride=1, padding=2, 
                                           weight_bit_width=8, bias_bit_width=8, inter_bit_width=32, retrain=True, quant=True)
        self.bn1 = nn.BatchNorm2d(32)\
                        if not qat else BatchNorm2d(32, bias_bit_width=8, weight_bit_width=8, retrain=True, quant=True)
                        
        self.relu1 = nn.ReLU()\
                        if not not qat else ReLU(acti_bit_width=8)

        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=0)
        if model == 'arm':
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2)\
                            if not qat else Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2, 
                                               weight_bit_width=8, bias_bit_width=8, inter_bit_width=32, retrain=True, quant=True)
            self.bn2 = nn.BatchNorm2d(16)\
                            if not qat else BatchNorm2d(16, bias_bit_width=8, weight_bit_width=8, retrain=True, quant=True)
        elif model=='cmsis':
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)\
                            if not qat else Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, 
                                               weight_bit_width=8, bias_bit_width=8, inter_bit_width=32, retrain=True, quant=True)
            self.bn2 = nn.BatchNorm2d(32) \
                                        if not qat else BatchNorm2d(32, bias_bit_width=8, weight_bit_width=8, retrain=True, quant=True)
        else:
            raise ValueError("Model not supported")
        
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        if model == 'arm':
            self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)\
                            if not qat else Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2, 
                                               weight_bit_width=8, bias_bit_width=8, inter_bit_width=32, retrain=True, quant=True)
            self.bn3 = nn.BatchNorm2d(32)\
                            if not qat else BatchNorm2d(32, bias_bit_width=8, weight_bit_width=8,  retrain=True, quant=True)
        elif model=='cmsis':
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)\
                            if not qat else Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, 
                                               weight_bit_width=8, bias_bit_width=8, inter_bit_width=32, retrain=True, quant=True)
            self.bn3 = nn.BatchNorm2d(64)\
                            if not qat else BatchNorm2d(64, bias_bit_width=8, weight_bit_width=8, retrain=True, quant=True)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        
        # if qat == True:
        #     self.conv1, self.bn1, self.relu1 = fold_CBR(conv=Conv2d(in_channels=self.conv1.in_channels, 
        #                                                             out_channels=self.conv1.out_channels, 
        #                                                             kernel_size=self.conv1.kernel_size, 
        #                                                             stride=self.conv1.stride, 
        #                                                             padding=self.conv1.padding),
        #                                                 bn=BatchNorm2d(self.bn1.num_features),
        #                                                 relu=ReLU())
        #     self.conv2, self.bn2, self.relu2 = fold_CBR(conv=Conv2d(in_channels=self.conv2.in_channels,
        #                                                             out_channels=self.conv2.out_channels,
        #                                                             kernel_size=self.conv2.kernel_size,
        #                                                             stride=self.conv2.stride,
        #                                                             padding=self.conv2.padding), 
        #                                                 bn=BatchNorm2d(self.bn2.num_features), 
        #                                                 relu=ReLU())
        #     self.conv3, self.bn3, self.relu3 = fold_CBR(conv=Conv2d(in_channels=self.conv3.in_channels,
        #                                                             out_channels=self.conv3.out_channels,
        #                                                             kernel_size=self.conv3.kernel_size,
        #                                                             stride=self.conv3.stride,
        #                                                             padding=self.conv3.padding), 
        #                                                 bn=BatchNorm2d(self.bn3.num_features), 
        #                                                 relu=ReLU())
        # else:
        #     self.conv1, self.bn1, self.relu1 = fold_CBR_qfmt(conv=self.conv1, 
        #                                                      bn=self.bn1, 
        #                                                      relu=self.relu1)
        #     self.conv2, self.bn2, self.relu2 = fold_CBR_qfmt(conv=self.conv2, 
        #                                                      bn=self.bn2, 
        #                                                      relu=self.relu2)
        #     self.conv3, self.bn3, self.relu3 = fold_CBR_qfmt(conv=self.conv3, 
        #                                                      bn=self.bn3, 
        #                                                      relu=self.relu3)
        self.flatten = nn.Flatten()
        if model == 'arm':
            self.fc1 = nn.Linear(in_features=512, out_features=num_classes)\
                if not qat else Linear(in_features=512, out_features=num_classes, 
                                   weight_bit_width=8, bias_bit_width=8, inter_bit_width=32, retrain=True, quant=True)
        else:
            self.fc1 = nn.Linear(in_features=1024, out_features=num_classes)\
                if not qat else Linear(in_features=1024, out_features=num_classes, 
                                   weight_bit_width=8, bias_bit_width=8, inter_bit_width=32, retrain=True, quant=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        return x
