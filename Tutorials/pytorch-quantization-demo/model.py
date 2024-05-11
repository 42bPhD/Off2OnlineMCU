import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d as C
from torch.nn import Linear as L
from module import QConv2d as QC
from module import QLinear as QL
from module import QReLU, QMaxPooling2d, QConvBNReLU
from qfmt_layer import QConv2d, QLinear

Conv2d = lambda in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=True: QConv2d(C(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias))
Linear = lambda in_features, out_features: QLinear(L(in_features=in_features, out_features=out_features))

class QNet(nn.Module):
    def __init__(self, num_channels=3):
        super(QNet, self).__init__()
        self.conv1 = Conv2d(num_channels, 40, 3, 1, 0)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = Conv2d(40, 40, 3, 1, 0)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = Linear(6*6*40, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    def quantize(self):
        self.conv1.freeze()
        self.conv2.freeze()
        self.fc.freeze()
    def quantize_forward(self, x):
        x = self.conv1.quantize_inference(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.conv2.quantize_inference(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = self.fc.quantize_inference(x)

class QNetBN(nn.Module):
    def __init__(self, num_channels=3):
        super(QNetBN, self).__init__()
        self.conv1 = Conv2d(num_channels, 40, 3, 1, 0)
        self.bn1 = nn.BatchNorm2d(40)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = Conv2d(40, 40, 3, 1, 0)
        self.bn2 = nn.BatchNorm2d(40)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = Linear(6*6*40, 10)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Net(nn.Module):

    def __init__(self, num_channels=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.conv2 = nn.Conv2d(40, 40, 3, 1)
        self.fc = nn.Linear(6*6*40, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 6*6*40)
        x = self.fc(x)
        return x

    def quantize(self, num_bits=8):
        self.qconv1 = QConv2d(self.conv1, qi=True, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU()
        self.qmaxpool2d_1 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qconv2 = QConv2d(self.conv2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu2 = QReLU()
        self.qmaxpool2d_2 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits)

    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qrelu1(x)
        x = self.qmaxpool2d_1(x)
        x = self.qconv2(x)
        x = self.qrelu2(x)
        x = self.qmaxpool2d_2(x)
        x = x.view(-1, 6*6*40)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qconv1.freeze()
        self.qrelu1.freeze(self.qconv1.qo)
        self.qmaxpool2d_1.freeze(self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qrelu2.freeze(self.qconv2.qo)
        self.qmaxpool2d_2.freeze(self.qconv2.qo)
        self.qfc.freeze(qi=self.qconv2.qo)

    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qmaxpool2d_1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qmaxpool2d_2.quantize_inference(qx)
        qx = qx.view(-1, 6*6*40)
        qx = self.qfc.quantize_inference(qx)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out


class NetBN(nn.Module):

    def __init__(self, num_channels=3):
        super(NetBN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.bn1 = nn.BatchNorm2d(40)
        self.conv2 = nn.Conv2d(40, 40, 3, 1)
        self.bn2 = nn.BatchNorm2d(40)
        self.fc = nn.Linear(6 * 6 * 40, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        
        x = x.view(-1, 6 * 6 * 40)
        x = self.fc(x)
        return x

    def quantize(self, num_bits=8):
        self.qconv1 = QConvBNReLU(self.conv1, self.bn1, qi=True, qo=True, num_bits=num_bits)
        self.qmaxpool2d_1 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qconv2 = QConvBNReLU(self.conv2, self.bn2, qi=False, qo=True, num_bits=num_bits)
        self.qmaxpool2d_2 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits)

    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qmaxpool2d_1(x)
        x = self.qconv2(x)
        x = self.qmaxpool2d_2(x)
        x = x.view(-1, 6*6*40)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qconv1.freeze()
        self.qmaxpool2d_1.freeze(self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qmaxpool2d_2.freeze(self.qconv2.qo)
        self.qfc.freeze(qi=self.qconv2.qo)

    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qmaxpool2d_1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qmaxpool2d_2.quantize_inference(qx)
        qx = qx.view(-1, 6*6*40)

        qx = self.qfc.quantize_inference(qx)
        
        out = self.qfc.qo.dequantize_tensor(qx)
        return out
