
import torch
from torch.nn import init
from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, Linear, Softmax, AdaptiveAvgPool2d, Flatten

from quantization.fold import fold_CBR
# from quantization import (Conv2d, 
#                           BatchNorm2d, 
#                           ReLU,
#                           MaxPool2d,
#                           Linear)
# from quantization.qformat import QConv2d, QLinear, fold_CBR_qfmt
import torchprofile

class DepthwiseConv2D(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding, bias=True):
        super(DepthwiseConv2D, self).__init__(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=strides,
                                              padding=padding,
                                              groups=in_channels,
                                              bias=bias)
        self.depthwise = Conv2d(in_channels, 
                                   out_channels, 
                                   kernel_size=kernel_size, 
                                   stride=strides, 
                                   padding=padding, 
                                   groups=in_channels, 
                                   bias=bias)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, Linear):
                init.kaiming_uniform_(m.weight)
                init.constant_(m.bias, 0)
                    
    def forward(self, x):
        out = self.depthwise(x)
        return out
    
class PointwieConv2D(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
        super(PointwieConv2D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             bias=bias)
        self.pointwise = Conv2d(in_channels, 
                                   out_channels, 
                                   kernel_size=kernel_size, 
                                   stride=stride, 
                                   padding=padding, 
                                   bias=bias)
        self.initialize_weights()
        
    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, Linear):
                init.kaiming_uniform_(m.weight)
                init.constant_(m.bias, 0)
    def forward(self, x):
        out = self.pointwise(x)
        return out

class MobileNetV1(Module):
    def __init__(self, num_classes=2, num_filters=8):
        super(MobileNetV1, self).__init__()
        self.num_filters = num_filters

        # 1st layer, pure conv
        # Keras 2.2 model has padding='valid' and disables bias
        
        self.conv1 = Conv2d(3, self.num_filters, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1 = BatchNorm2d(self.num_filters)
        self.relu1 = ReLU(inplace=False)
        # self.conv1, self.bn1, self.relu1 = fold_CBR_qfmt(self.conv1, self.bn1, self.relu1)
        # 2nd layer, depthwise separable conv
        # Filter size is always doubled before the pointwise conv
        # Keras uses ZeroPadding2D() and padding='valid'
        
        self.conv2_1 = DepthwiseConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        strides=1,
                                        padding=1)
        self.bn2_1 = BatchNorm2d(self.num_filters)
        self.relu2_1 = ReLU(inplace=False)
        # self.conv2_1, self.bn2_1, self.relu2_1 = fold_CBR(self.conv2_1, self.bn2_1, self.relu2_1)
        
        self.conv2_2 = PointwieConv2D(in_channels=self.num_filters,
                                     out_channels=self.num_filters*2)
        self.num_filters *= 2
        self.bn2_2 = BatchNorm2d(self.num_filters)
        self.relu2_2 = ReLU(inplace=False)
        # self.conv2_2, self.bn2_2, self.relu2_2 = fold_CBR(self.conv2_2, self.bn2_2, self.relu2_2)
        
        # 3rd layer, depthwise separable conv
        self.conv3_1 = DepthwiseConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        strides=2,
                                        padding=1)
        self.bn3_1 = BatchNorm2d(self.num_filters)
        self.relu3_1 = ReLU(inplace=False)
        # self.conv3_1, self.bn3_1, self.relu3_1 = fold_CBR(self.conv3_1, self.bn3_1, self.relu3_1)
        
        self.conv3_2 = PointwieConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters*2)
        self.num_filters *= 2
        self.bn3_2 = BatchNorm2d(self.num_filters)
        self.relu3_2 = ReLU(inplace=False)
        # self.conv3_2, self.bn3_2, self.relu3_2 = fold_CBR(self.conv3_2, self.bn3_2, self.relu3_2)

        
        # 4th layer, depthwise separable conv
        self.conv4_1 = DepthwiseConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        strides=1,
                                        padding=1)
        self.bn4_1 = BatchNorm2d(self.num_filters)
        self.relu4_1 = ReLU(inplace=False)
        # self.conv4_1, self.bn4_1, self.relu4_1 = fold_CBR(self.conv4_1, self.bn4_1, self.relu4_1)
        
        self.conv4_2 = PointwieConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters)
        self.bn4_2 = BatchNorm2d(self.num_filters)
        self.relu4_2 = ReLU(inplace=False)
        # self.conv4_2, self.bn4_2, self.relu4_2 = fold_CBR(self.conv4_2, self.bn4_2, self.relu4_2)
        
        # 5th layer, depthwise separable conv
        self.conv5_1 = DepthwiseConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        strides=2,
                                        padding=1)
        self.bn5_1 = BatchNorm2d(self.num_filters)
        self.relu5_1 = ReLU(inplace=False)
        # self.conv5_1, self.bn5_1, self.relu5_1 = fold_CBR(self.conv5_1, self.bn5_1, self.relu5_1)
        
        self.conv5_2 = PointwieConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters*2)
        self.num_filters *= 2
        self.bn5_2 = BatchNorm2d(self.num_filters)
        self.relu5_2 = ReLU(inplace=False)
        # self.conv5_2, self.bn5_2, self.relu5_2 = fold_CBR(self.conv5_2, self.bn5_2, self.relu5_2)
        
        # 6th layer, depthwise separable conv
        self.conv6_1 = DepthwiseConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        strides=1,
                                        padding=1)
        self.bn6_1 = BatchNorm2d(self.num_filters)
        self.relu6_1 = ReLU(inplace=False)
        # self.conv6_1, self.bn6_1, self.relu6_1 = fold_CBR(self.conv6_1, self.bn6_1, self.relu6_1)
        
        self.conv6_2 = PointwieConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters)
        self.bn6_2 = BatchNorm2d(self.num_filters)
        self.relu6_2 = ReLU(inplace=False)
        # self.conv6_2, self.bn6_2, self.relu6_2 = fold_CBR(self.conv6_2, self.bn6_2, self.relu6_2)

        # 7th layer, depthwise separable conv
        self.conv7_1 = DepthwiseConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        strides=2,
                                        padding=1)
        self.bn7_1 = BatchNorm2d(self.num_filters)
        self.relu7_1 = ReLU(inplace=False)
        # self.conv7_1, self.bn7_1, self.relu7_1 = fold_CBR(self.conv7_1, self.bn7_1, self.relu7_1)

        self.conv7_2 = PointwieConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters*2)
        self.num_filters *= 2
        self.bn7_2 = BatchNorm2d(self.num_filters)
        self.relu7_2 = ReLU(inplace=False)
        # self.conv7_2, self.bn7_2, self.relu7_2 = fold_CBR(self.conv7_2, self.bn7_2, self.relu7_2)
        
        # 8th-12th layers, identical depthwise separable convs
        # 8th
        self.conv8_1 = DepthwiseConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        strides=1,
                                        padding=1)
        self.bn8_1 = BatchNorm2d(self.num_filters)
        self.relu8_1 = ReLU(inplace=False)
        # self.conv8_1, self.bn8_1, self.relu8_1 = fold_CBR(self.conv8_1, self.bn8_1, self.relu8_1)
        
        self.conv8_2 = PointwieConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters)
        self.bn8_2 = BatchNorm2d(self.num_filters)
        self.relu8_2 = ReLU(inplace=False)
        # self.conv8_2, self.bn8_2, self.relu8_2 = fold_CBR(self.conv8_2, self.bn8_2, self.relu8_2)
        
        # 9th
        self.conv9_1 = DepthwiseConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        strides=1,
                                        padding=1)
        self.bn9_1 = BatchNorm2d(self.num_filters)
        self.relu9_1 = ReLU(inplace=False)
        # self.conv9_1, self.bn9_1, self.relu9_1 = fold_CBR(self.conv9_1, self.bn9_1, self.relu9_1)
        
        self.conv9_2 = PointwieConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters)
        self.bn9_2 = BatchNorm2d(self.num_filters)
        self.relu9_2 = ReLU(inplace=False)
        # self.conv9_2, self.bn9_2, self.relu9_2 = fold_CBR(self.conv9_2, self.bn9_2, self.relu9_2)
        
        # 10th
        self.conv10_1 = DepthwiseConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        strides=1,
                                        padding=1)
        self.bn10_1 = BatchNorm2d(self.num_filters)
        self.relu10_1 = ReLU(inplace=False)
        # self.conv10_1, self.bn10_1, self.relu10_1 = fold_CBR(self.conv10_1, self.bn10_1, self.relu10_1)
        
        self.conv10_2 = PointwieConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters)
        self.bn10_2 = BatchNorm2d(self.num_filters)
        self.relu10_2 = ReLU(inplace=False)
        # self.conv10_2, self.bn10_2, self.relu10_2 = fold_CBR(self.conv10_2, self.bn10_2, self.relu10_2)
        
        # 11th
        self.conv11_1 = DepthwiseConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        strides=1,
                                        padding=1)
        self.bn11_1 = BatchNorm2d(self.num_filters)
        self.relu11_1 = ReLU(inplace=False)
        # self.conv11_1, self.bn11_1, self.relu11_1 = fold_CBR(self.conv11_1, self.bn11_1, self.relu11_1)
        
        self.conv11_2 = PointwieConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters)
        self.bn11_2 = BatchNorm2d(self.num_filters)
        self.relu11_2 = ReLU(inplace=False)
        # self.conv11_2, self.bn11_2, self.relu11_2 = fold_CBR(self.conv11_2, self.bn11_2, self.relu11_2)
        
        # 12th
        self.conv12_1 = DepthwiseConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        strides=1,
                                        padding=1)
        self.bn12_1 = BatchNorm2d(self.num_filters)
        self.relu12_1 = ReLU(inplace=False)
        # self.conv12_1, self.bn12_1, self.relu12_1 = fold_CBR(self.conv12_1, self.bn12_1, self.relu12_1)
        
        self.conv12_2 = PointwieConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters)
        self.bn12_2 = BatchNorm2d(self.num_filters)
        self.relu12_2 = ReLU(inplace=False)
        # self.conv12_2, self.bn12_2, self.relu12_2 = fold_CBR(self.conv12_2, self.bn12_2, self.relu12_2)
        
        # 13th layer, depthwise separable conv
        self.conv13_1 = DepthwiseConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        strides=2,
                                        padding=1)
        self.bn13_1 = BatchNorm2d(self.num_filters)
        self.relu13_1 = ReLU(inplace=False)
        # self.conv13_1, self.bn13_1, self.relu13_1 = fold_CBR(self.conv13_1, self.bn13_1, self.relu13_1)
        
        
        self.conv13_2 = PointwieConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters*2)
        self.num_filters *= 2
        self.bn13_2 = BatchNorm2d(self.num_filters)
        self.relu13_2 = ReLU(inplace=False)
        # self.conv13_2, self.bn13_2, self.relu13_2 = fold_CBR(self.conv13_2, self.bn13_2, self.relu13_2)
        
        # 14th layer, depthwise separable conv
        self.conv14_1 = DepthwiseConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        strides=1,
                                        padding=1)
        self.bn14_1 = BatchNorm2d(self.num_filters)
        self.relu14_1 = ReLU(inplace=False)
        # self.conv14_1, self.bn14_1, self.relu14_1 = fold_CBR(self.conv14_1, self.bn14_1, self.relu14_1)
        
        self.conv14_2 = PointwieConv2D(in_channels=self.num_filters,
                                        out_channels=self.num_filters)
        self.bn14_2 = BatchNorm2d(self.num_filters)
        self.relu14_2 = ReLU(inplace=False)
        # self.conv14_2, self.bn14_2, self.relu14_2 = fold_CBR(self.conv14_2, self.bn14_2, self.relu14_2)
        
        # Average pooling, max polling may be used also
        # Keras employs GlobalAveragePooling2D 
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.dense = Linear(self.num_filters, num_classes)
        self.softmax = Softmax(dim=1)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, Linear):
                init.kaiming_uniform_(m.weight)
                init.constant_(m.bias, 0)
        
        
    def forward(self, x, softmax=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)
        
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)
        
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu3_2(x)
        
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu4_1(x)
        
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu4_2(x)
        
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu5_1(x)
        
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.relu5_2(x)
        
        x = self.conv6_1(x)
        x = self.bn6_1(x)
        x = self.relu6_1(x)
        
        x = self.conv6_2(x)
        x = self.bn6_2(x)
        x = self.relu6_2(x)
        
        x = self.conv7_1(x)
        x = self.bn7_1(x)
        x = self.relu7_1(x)
        
        x = self.conv7_2(x)
        x = self.bn7_2(x)
        x = self.relu7_2(x)
        
        x = self.conv8_1(x)
        x = self.bn8_1(x)
        x = self.relu8_1(x)
        
        x = self.conv8_2(x)
        x = self.bn8_2(x)
        x = self.relu8_2(x)
        
        x = self.conv9_1(x)
        x = self.bn9_1(x)
        x = self.relu9_1(x)
        
        x = self.conv9_2(x)
        x = self.bn9_2(x)
        x = self.relu9_2(x)
        
        x = self.conv10_1(x)
        x = self.bn10_1(x)
        x = self.relu10_1(x)
        
        x = self.conv10_2(x)
        x = self.bn10_2(x)
        x = self.relu10_2(x)
        
        x = self.conv11_1(x)
        x = self.bn11_1(x)
        x = self.relu11_1(x)
        
        x = self.conv11_2(x)
        x = self.bn11_2(x)
        x = self.relu11_2(x)
        
        x = self.conv12_1(x)
        x = self.bn12_1(x)
        x = self.relu12_1(x)
        
        x = self.conv12_2(x)
        x = self.bn12_2(x)
        x = self.relu12_2(x)
        
        x = self.conv13_1(x)
        x = self.bn13_1(x)
        x = self.relu13_1(x)
        
        x = self.conv13_2(x)
        x = self.bn13_2(x)
        x = self.relu13_2(x)
        
        x = self.conv14_1(x)
        x = self.bn14_1(x)
        x = self.relu14_1(x)
        
        x = self.conv14_2(x)
        x = self.bn14_2(x)
        x = self.relu14_2(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        # if softmax:
            # x = self.softmax(x)
        x = self.softmax(x)
        return x
    
def mobilenet_v1(num_classes=10, num_filters=8):
    model = MobileNetV1(num_classes=num_classes, 
                        num_filters=num_filters)
    return model
    
if __name__ == '__main__':
    import torchsummary
    import os
    model = mobilenet_v1()
    inp = torch.randn(1, 3, 96, 96)
    print(inp.shape)
    # macs = torchprofile.profile_macs(model, inp)
    res = model(inp)
    print(res.shape)
    # print(f'{macs:,}')
    torchsummary.summary(model, (3, 32, 32), device='cpu')
    os.makedirs('./trained_models/mbv1/default/', exist_ok=True)
    torch.save(model.state_dict(), './trained_models/mbv1/default/mbv1.pth')