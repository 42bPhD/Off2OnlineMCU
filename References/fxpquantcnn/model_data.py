import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from tqdm import tqdm
from torch.utils.data import DataLoader
def get_model_weights(model:nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get weights and biases of the CNN model. Includes Conv and Linear layers

    Returns:
        tensor: concatenated weights
        tensor: concatenated biases
    """
    
    weights_list = []
    biases_list = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weights_list.append(module.weight.data.view(-1))
            if module.bias is not None:
                biases_list.append(module.bias.data.view(-1))

    weights_tensor = torch.cat(weights_list)
    biases_tensor = torch.cat(biases_list)

    return weights_tensor, biases_tensor


def get_model_weights_by_layer(model:nn.Module, dense=False)->Tuple[torch.Tensor]:
    """Get weights and biases of CNN model as a list, separately for each layer
        
    Args:
        dense (bool, optional): Include linear layer weights and biases. Defaults to False.
    Returns:
        list of tensors: weights and biases for each layer.
    """

    layer_weights = []

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            layer_weights.append([module.weight.data, module.bias.data if module.bias is not None else None])
        elif dense and isinstance(module, nn.Linear):
            layer_weights.append([module.weight.data, module.bias.data if module.bias is not None else None])
    
    return layer_weights


def get_conv_layer_indices(model:nn.Module)->List[int]:
    """Get all indices of Convolutional layers in the PyTorch Model
    
    Args:
        model (Object): PyTorch Model
    
    Returns:
        list: conv layer indices
    """

    layer_num = []
    for i, module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            layer_num.append(i)

    return layer_num


def get_linear_layer_indices(model:nn.Module)->List[int]:
    """Get all indices of Linear layers in the PyTorch Model
    
    Args:
        model (Object): PyTorch Model
    
    Returns:
        list: linear layer indices
    """
    layer_num = []
    for i, module in enumerate(model.modules()):
        if isinstance(module, nn.Linear):
            layer_num.append(i)

    return layer_num


def get_activation_maps(model:nn.Module, x_test_sample:torch.Tensor, layer_name:str)->Dict[str, torch.Tensor]:

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(get_activation(name))
    
    model(x_test_sample)
    return activation[layer_name]


def get_num_params_per_layer(model:nn.Module, layer_names:List[str])->Tuple[List[int], List[int] ]:
    """Get the number of parameters (weights and biases) for each layer
    
    Args:
        layer_names (sequence): Sequence of layer names
    
    Returns:
        sequence: Number of weights per layer
        sequence: Number of biases per layer
    """

    num_weights = []
    num_biases = []
    for name, module in model.named_modules():
        if name in layer_names:
            num_weights.append(module.weight.data.numel())
            if module.bias is not None:
                num_biases.append(module.bias.data.numel())
            else:
                num_biases.append(0)
    
    return num_weights, num_biases


def get_num_activations_per_layer(model, layer_names, exclude_kernels=False):
    """Get the number of activations for each layer
    
    Args:
        layer_names (sequence): Sequence of layer names
    
    Returns:
        sequence: Number of activations per layer
    """

    num_activations = []
    for name, module in model.named_modules():
        if name in layer_names:
            if exclude_kernels and isinstance(module, nn.Conv2d):
                num_activations.append(module.out_channels)
            else:
                num_activations.append(module.out_channels * module.kernel_size[0] * module.kernel_size[1])
    
    return num_activations


class Model:
    """Class for containing properties of the PyTorch model
        
    Args:
        name (string): name of the model
        test_data (Tuple of tensors): Test data (x, y) to evaluate the network on
        model (Object, optional): PyTorch model object if loaded model already 
                                        available. Defaults to None.
        path (String, optional): Path to PyTorch model to load the model from.
                                Defaults to None.
    """

    def __init__(self, name:str, test_data, model:nn.Module=None, path:str=None):

        self.name = name
        self.test_data = test_data
        self.path = path
        self.batch_size = 32
        
        if self.path is not None:
            model.load_state_dict(torch.load(self.path))
            self.model = model
        else:
            self.model = model

    def __str__(self):
        return self.name

    @property
    def x_test(self):
        """Test data X
        """
        return self.test_data[0]
    
    @property
    def y_test(self):
        """Test data y
        """
        return self.test_data[1]
    
    @property
    def conv_layer_indices(self):
        """Indices of convolutional layers
        """
        return get_conv_layer_indices(self.model)
    
    @property
    def linear_layer_indices(self):
        """Indices of linear layers
        """
        return get_linear_layer_indices(self.model)

    def load_model_from_path(self):
        """load PyTorch model from given path
        """
        if self.path is not None:
            self.model.load_state_dict(torch.load(self.path))
            return self
        else:
            raise ValueError('Path variable is empty')

    def evaluate_accuracy(self):
        """Evaluate inference accuracy of the network

        Returns:
            list: Loss and accuracy of the model for the given test data
        """
        #! TODO: Using Dataloader for test data
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()

        dataloader = DataLoader(self.test_data, shuffle=False)
        
        
        num_samples = 0
        num_correct = 0

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
                # Move the data from CPU to GPU
                inputs = inputs.cuda()
                
                targets = targets.cuda()

                # Inference
                outputs =  self.model(inputs)

                # Convert logits to class indices
                outputs = outputs.argmax(dim=1)

                # Update metrics
                num_samples += targets.size(0)
                num_correct += (outputs == targets).sum()

            return (num_correct / num_samples).item()

    
    def get_num_params_per_layer(self, layer_names):
        """Get the number of parameters (weights and biases) for each layer
        
        Args:
            layer_names (sequence): Sequence of layer names
        
        Returns:
            sequence: Number of weights per layer
            sequence: Number of biases per layer
        """

        num_weights = []
        num_biases = []
        for name, module in self.model.named_modules():
            if name in layer_names:
                num_weights.append(module.weight.data.numel())
                if module.bias is not None:
                    num_biases.append(module.bias.data.numel())
                else:
                    num_biases.append(0)
        
        return num_weights, num_biases
    
    def get_num_activations_per_layer(self, layer_names):
        """Get the number of activations for each layer
        
        Args:
            layer_names (sequence): Sequence of layer names
        
        Returns:
            sequence: Number of activations per layer
        """

        num_activations = []
        for name, module in self.model.named_modules():
            if name in layer_names:
                num_activations.append(module.out_channels * module.kernel_size[0] * module.kernel_size[1])
        
        return num_activations


    def get_model_weights(self):
        """Get weights and biases of the CNN model. Includes Conv and Linear layers

        Returns:
            tensor: concatenated weights
            tensor: concatenated biases
        """
    
        weights_list = []
        biases_list = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weights_list.append(module.weight.data.view(-1))
                if module.bias is not None:
                    biases_list.append(module.bias.data.view(-1))
        
        weights_tensor = torch.cat(weights_list)
        biases_tensor = torch.cat(biases_list)

        return weights_tensor, biases_tensor

    def get_model_weights_by_layer(self, dense=False):
        """Get weights and biases of CNN model as a list, separately for each layer
        
        Args:
            dense (bool, optional): Include linear layer weights and biases. Defaults to False.
        Returns:
            list of tensors: weights and biases for each layer.
        """
        layer_weights = []

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                layer_weights.append([module.weight.data, module.bias.data if module.bias is not None else None])
            elif dense and isinstance(module, nn.Linear):
                layer_weights.append([module.weight.data, module.bias.data if module.bias is not None else None])
        
        return layer_weights

    def get_activation_maps(self, x_test_sample, layer_name):
        """Get activation maps from a given layer after passing a test sample 
        image through the network
        
        Args:
            x_test_sample (tensor): One or more images to pass through the network
            layer_name (string): Name of layer to collect values from
        
        Returns:
            tensor: Activations from the specified layer
        """

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(get_activation(name))
        
        self.model(x_test_sample)
        return activation[layer_name]
