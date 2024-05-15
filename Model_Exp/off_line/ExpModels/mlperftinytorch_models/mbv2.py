from torch import nn
from torch.ao.quantization import QuantStub, DeQuantStub
import torch

class ConvBNRelu6( nn.Sequential ):
    """
    Convolutional layer with batchnorm and clipping ReLU6
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNRelu6, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class ConvBNRelu( nn.Sequential ):
    """
    Convolutional layer with batchnorm and clipping ReLU
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNRelu, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

class InvertedResidual( nn.Module ):
    """
    Bottleneck Block
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual,self).__init__()
        self.stride = stride
    
        assert stride in [1,2]

        hidden_dim = int(round(expand_ratio*inp))
        
        # Residual connection only if channels are preserved (inp == oup).
        # and the image dimension is preserved (stride == 1)
        
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []

        if expand_ratio != 1:
            layers.append(ConvBNRelu6(inp, hidden_dim, kernel_size=1))

        layers.extend([
            # Depthwise Convolution/Fully-grouped Convolution
            ConvBNRelu6(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            
            # Pointwise Linear Convolution
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self,x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class CIDR_MobileNetv2( nn.Module ):
    """
    CIDR Implementation of MBV2
    """
    def __init__(self, num_classes=10, width_mult=1.0):
        super(CIDR_MobileNetv2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        block_setting = [
            # t, c, n, stride
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # first layer
        input_channel = int(input_channel * width_mult)
        features = [ConvBNRelu6(3, input_channel, stride=2)]

        self.last_channel = int(last_channel * max(1.0, width_mult))
        
        # blocks
        for t, c, n, stride in block_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = stride if i == 0 else 1
                features.append( block(input_channel, output_channel, stride, expand_ratio=t) )
                input_channel = output_channel
        
        # last layer
        features.append( ConvBNRelu6(input_channel, self.last_channel, kernel_size=1) )
        
        self.features = nn.Sequential(*features)

        self.projector = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.last_channel, num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        # what'stride the point of taking this mean?
        x = x.mean(-1).mean(-1)
        x = self.projector(x)
        return x

class Bottleneck_Block(nn.Module):
    '''Kuangliu implementation of CIFAR MBV2'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Bottleneck_Block, self).__init__()
        self.stride = stride

        mid_planes = expansion * in_planes

        # One of main differences is that this ReLU doesn't clip at 6

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        # If the input planes are not equal to output planes, the residual connection fails.
        # Make them the same by adding a convolution layer. 
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.relu2(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.stride==1:
            return self.skip_add.add(y,self.shortcut(x)) 
        else:
            return y

class KL_MBV2(nn.Module):

    cfg = [
            # t, c, n, stride
            [1, 16, 1, 1], 
            [6, 24, 2, 1], #s=2
            [6, 32, 3, 1], #s=2
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

    def __init__(self,num_classes=10):
        super(KL_MBV2, self).__init__()

        self.features = nn.Sequential(
            ConvBNRelu6(3,32,3,1,1),
            self._make_layers(in_planes=32),
            ConvBNRelu6(320,1280,1,1),
            # this is the other main difference, this averagepool after the network.
            nn.AvgPool2d(4)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280,num_classes)
        )

    def _make_layers(self, in_planes):
        layers = []
        for expansion,out_planes,num_blocks,stride in self.cfg:
            stride = [stride] + [1]*(num_blocks-1)
            for stride in stride:
                layers.append(Bottleneck_Block(in_planes,out_planes,expansion,stride))
                in_planes = out_planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0),-1)
        x = x.mean(-1).mean(-1)
        x = self.classifier(x)
        return x
    
class KL_MBV2_forVWW(nn.Module):

    cfg = [
            # t, c, n, stride
            [1, 16, 1, 1], 
            [6, 24, 2, 1], #s=2
            [6, 32, 3, 2], #s=2
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

    def __init__(self,num_classes=2):
        super(KL_MBV2_forVWW, self).__init__()

        self.features = nn.Sequential(
            ConvBNRelu6(3,32,3,1,1),
            self._make_layers(in_planes=32),
            ConvBNRelu6(320,1280,1,1),
            # this is the other main difference, this averagepool after the network.
            nn.AvgPool2d(4)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280,num_classes)
        )

    def _make_layers(self, in_planes):
        layers = []
        for expansion,out_planes,num_blocks,stride in self.cfg:
            stride = [stride] + [1]*(num_blocks-1)
            for stride in stride:
                layers.append(Bottleneck_Block(in_planes,out_planes,expansion,stride))
                in_planes = out_planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0),-1)
        x = x.mean(-1).mean(-1)
        x = self.classifier(x)
        return x

class KL_MBV2_Q(nn.Module):

    cfg = [
            # t, c, n, stride
            [1, 16, 1, 1], 
            [6, 24, 2, 1], #s=2
            [6, 32, 3, 1], #s=2
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

    def __init__(self,num_classes=10):
        super(KL_MBV2_Q, self).__init__()

        self.features = nn.Sequential(
            ConvBNRelu(3,32,3,1,1),
            self._make_layers(in_planes=32),
            ConvBNRelu(320,1280,1,1),
            # this is the other main difference, this averagepool after the network.
            nn.AvgPool2d(4)
        )

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # third difference: no dropout layer
        self.classifier = nn.Linear(1280,num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion,out_planes,num_blocks,stride in self.cfg:
            stride = [stride] + [1]*(num_blocks-1)
            for stride in stride:
                layers.append(Bottleneck_Block(in_planes,out_planes,expansion,stride))
                in_planes = out_planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        #x = x.view(x.size(0),-1)
        x = x.mean(-1).mean(-1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNRelu:
                torch.ao.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            # if type(m) == Bottleneck_Block:
            #     for idx in range(len(m.conv)):
            #         if type(m.conv[idx]) == nn.Conv2d:
            #             torch.ao.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)


