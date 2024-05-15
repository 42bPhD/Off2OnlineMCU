This folder contains the models with FP64 accuracies as shown below.
We also have the .onnx versions.

Models are without file extensions.

| Task                                         	| MLPerfTiny Prescribed  Model                             	| MLPerfTiny Target Performance 	| Single-node performance with  default hyperparameters 	|
|----------------------------------------------	|----------------------------------------------------------	|-------------------------------	|-------------------------------------------	|
| Visual Wakewords                             	| MobileNet (v1)                                           	| 0.80 (top-1)                  	| 0.91                                      	|
| Keyword Spotting (Hello Edge Ver.)           	| DS-CNN (Hello Edge)                                      	| 0.90 (top-1)                  	| 0.94                                      	|
| Image Classification (CIFAR-10)              	| 3-Block ResNet                                           	| 0.85 (top-1)                  	| 0.87                                      	|
| Anomaly Detection (DCASE 2020 Task 2 ToyCar) 	| Fully-connected Autoencoder (DCASE 2020 Task 2 Baseline) 	| 0.85 (AUC)                    	| 0.89                                      	|


### How to load

```Python

# From the repository's root directory

import torch

# Load model with torch load.

model = torch.load('final_models/ic_final')

# Load the dataset as per the runme examples in the solo_train_scripts directory

import image_classification, torchvision
trainset = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=False,
        transform=image_classification.utils.t_cropflip_augment
    )
testset = torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=False,
        transform=image_classification.utils.t_normalize
    )

# Put into an IC node from the framework to borrow the test functionality (you can also make a test function yourself)

from image_classification.ic_node import ic_node

node = ic_node(my_config)
node.dp_model.test_set = testset
node.dp_model.train_set = trainset
node.dp_model.load_loaders()
node.dp_model.model = model

loss, accuracy = node.dp_model.test()
print(f'Loss:{Loss}\tAccuracy:{accuracy}')

```
