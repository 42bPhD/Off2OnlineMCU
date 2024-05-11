import matplotlib.pyplot as plt
from torchprofile import profile_macs
from torch import nn
import torch
import torchsummary
def get_model_flops(model, inputs):
    num_macs = profile_macs(model, inputs)
    return num_macs

def get_model_size(model: nn.Module, data_width=32):
    """
    calculate the model size in bits
    :param data_width: #bits per element
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fcl = nn.Linear(in_features=4*4*32, out_features=10)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        bs = x.shape[0]
        x = x.view(bs, -1)
        x = self.fcl(x)
        return x
    
if __name__ == '__main__':
    print("Hello World!")
    model = LeNet()
    inp = torch.rand(1, 3, 32, 32)
    
    mac = profile_macs(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=3), inp)
    # model_sz = get_model_size(model, data_width=8)
    suumary = torchsummary.summary(model.cuda(), (3, 32, 32))
    # print(suumary)
    #Mega <million> Operations Per Second
    print(f'Total MAC: {mac:,}')
    # print(f'ModelSize: {model_sz:,} Bytes')