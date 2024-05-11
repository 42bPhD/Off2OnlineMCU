
import torch
import torch.nn as nn
import torch.optim as optim

from convert_float_fixed import ConvertFloatFixed


class PytorchCNN(nn.Module):
    """Class for the simple sequential PyTorch CNN with quantization layers
    or as original floating point model

    Args:
    input_shape (Tuple of integers): Input shape for input layer
    num_outputs (integer): number of output neurons / number of classes
    path_trained_weights (string, optional): Absolute or relative path to weights if 
                                            weights must be loaded when loading the 
                                            model. Defaults to None.
    """

    def __init__(self, input_shape, num_outputs, path_trained_weights=None):
        super(PytorchCNN, self).__init__()
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.path_trained_weights = path_trained_weights
        
        self.conv1 = self._add_conv_layer(32, (3, 3))
        self.conv2 = self._add_conv_layer(32, (3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = self._add_conv_layer(64, (3, 3), padding=1)
        self.conv4 = self._add_conv_layer(64, (3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(0.25)
        
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout3 = nn.Dropout(0.25)
        self.fc = nn.Linear(64, num_outputs)
        self.softmax = nn.Softmax(dim=1)

    def _add_conv_layer(self, out_channels, kernel_size, stride=1, padding=0):
        """Add convolutional layer with BatchNorm
        
        Args:
            out_channels (integer): Number of output channels
            kernel_size (2-Tuple of integers): Size of the kernel
            stride (integer): Stride of the convolution. Default: 1
            padding (integer): Padding added to all four sides of the input. Default: 0
        """
        conv = nn.Conv2d(self.input_shape[0] if not hasattr(self, 'conv1') else out_channels,
                         out_channels, kernel_size, stride, padding)
        bn = nn.BatchNorm2d(out_channels)
        return nn.Sequential(conv, bn, nn.ReLU(inplace=True))

    def get_fxp_model(self, quant_params):
        """Get CNN model with quantization layers for quantizing layer outputs/activations
        
        Args:
            quant_params (dict): Dictionary of layer names as keys with values of
                                [bw, f] to quantize the layer's output to

        Returns:
            nn.Module: PyTorch model with quantization layers
        """
        model = PytorchCNN(self.input_shape, self.num_outputs)

        for name, module in model.named_modules():
            if name in quant_params:
                bw, f = quant_params[name]
                cff = ConvertFloatFixed(bw, f)
                quant_layer = cff.quantize_pt
                model.add_module(name + '_quant', quant_layer)

        if self.path_trained_weights:
            model.load_state_dict(torch.load(self.path_trained_weights))

        return model

    def get_float_model(self):
        """Get floating point precision model of the PyTorch CNN
        
        Returns:
            nn.Module: PyTorch model
        """
        model = PytorchCNN(self.input_shape, self.num_outputs)

        if self.path_trained_weights:
            model.load_state_dict(torch.load(self.path_trained_weights))

        return model

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout3(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x


class PytorchCNNLarge(nn.Module):
    """Class for a longer/larger sequential PyTorch CNN with quantization layers 
    or as original floating point model
    
    Args:
    input_shape (Tuple of integers): Input shape for input layer
    num_outputs (int): number of output neurons / number of classes
    num_kernels (int): number of kernels for the first half of the network
    num_stages (int): number of stages to add. Each stage consists of 2 convolutional layers. Defaults to 2.
    pool_layer_interval (int): stage intervals at which to place the max pooling layer. Defaults to 1
    path_trained_weights (string, optional): Absolute or relative path to weights if 
                                            weights must be loaded when loading the 
                                            model. Defaults to None.
    """

    def __init__(self, input_shape, num_outputs, num_kernels, num_stages=2, pool_layer_interval=1, path_trained_weights=None):
        super(PytorchCNNLarge, self).__init__()
        
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.num_kernels = num_kernels
        self.num_stages = num_stages
        self.pool_layer_interval = pool_layer_interval
        self.path_trained_weights = path_trained_weights

        self.layers = self._make_layers()
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(num_kernels * 2 ** (num_stages // 2), num_outputs)
        self.softmax = nn.Softmax(dim=1)

    def _add_conv_layer(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1):
        """Add convolutional layer with BatchNorm
        
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (tuple): Size of the convolving kernel. Default: (3, 3)
            stride (int): Stride of the convolution. Default: 1
            padding (int): Padding added to all four sides of the input. Default: 1
        """
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        bn = nn.BatchNorm2d(out_channels)
        return nn.Sequential(conv, bn, nn.ReLU(inplace=True))

    def _make_layers(self):
        """Create the layers for the network
        
        Returns:
            nn.Sequential: Sequential container of all the layers
        """
        layers = []
        in_channels = self.input_shape[0]
        num_kernels = self.num_kernels

        for i in range(self.num_stages):
            layers.append(self._add_conv_layer(in_channels, num_kernels))
            layers.append(self._add_conv_layer(num_kernels, num_kernels))

            if (i + 1) % self.pool_layer_interval == 0:
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), padding=1))
                layers.append(nn.Dropout(0.25))

            if (i + 1) * 2 == (self.num_stages // 2) * 2:
                num_kernels *= 2

            in_channels = num_kernels

        return nn.Sequential(*layers)

    def get_fxp_model(self, quant_params):
        """Get CNN model with quantization layers for quantizing layer outputs/activations
        
        Args:
            quant_params (dict): Dictionary of layer names as keys with values of
                                [bw, f] to quantize the layer's output to
        
        Returns:
            nn.Module: PyTorch model with quantization layers
        """
        model = PytorchCNNLarge(self.input_shape, self.num_outputs, self.num_kernels,
                                self.num_stages, self.pool_layer_interval)

        for name, module in model.named_modules():
            if name in quant_params:
                bw, f = quant_params[name]
                cff = ConvertFloatFixed(bw, f)
                quant_layer = cff.quantize_pt
                model.add_module(name + '_quant', quant_layer)

        if self.path_trained_weights:
            model.load_state_dict(torch.load(self.path_trained_weights))

        return model

    def get_float_model(self):
        """Get floating point precision model of the PyTorch CNN
        
        Returns:
            nn.Module: PyTorch model
        """
        model = PytorchCNNLarge(self.input_shape, self.num_outputs, self.num_kernels,
                                self.num_stages, self.pool_layer_interval)

        if self.path_trained_weights:
            model.load_state_dict(torch.load(self.path_trained_weights))

        return model

    def forward(self, x):
        x = self.layers(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x
