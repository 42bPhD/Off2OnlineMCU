import torch
from torch import nn
from torch.quantization import QuantStub, DeQuantStub

class LeNet(nn.Module):
    # ARM Example Model
    def __init__(self, num_channels=3, num_classes=10, model='arm'):
        super(LeNet,self).__init__()
        
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.quant = QuantStub()	# 입력을 양자화 하는 QuantStub()
        self.dequant = DeQuantStub()	# 출력을 역양자화 하는 DeQuantStub()
        
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=0)
        if model == 'arm':
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2)
            self.bn2 = nn.BatchNorm2d(16)
        elif model=='cmsis':
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
            self.bn2 = nn.BatchNorm2d(32)
        else:
            raise ValueError("Model not supported")
        
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        if model == 'arm':
            self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
            self.bn3 = nn.BatchNorm2d(32)
        elif model=='cmsis':
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
            self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        
        self.flatten = nn.Flatten()
        if model == 'arm':
            self.fc1 = nn.Linear(in_features=512, out_features=num_classes)
        elif model=='cmsis':
            self.fc1 = nn.Linear(in_features=1024, out_features=num_classes)    
        
    def forward(self, x):
        x = self.quant(x)
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
        x = self.dequant(x)
        return x

    def get_shape(self, batch_size, input_shape):
        x = torch.randn(size=(1, 3, 32, 32))
        
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
        return x