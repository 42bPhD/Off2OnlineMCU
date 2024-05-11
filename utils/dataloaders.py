import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np
#Get mean ans std of dataset
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

# 학습 및 검증 데이터셋에 대한 사용자 정의 서브셋
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices=None, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.indices = indices

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]] if self.indices else self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.indices)

    
def get_class_distribution_loaders(dataloader_obj:DataLoader,
                                   dataset_obj: ImageFolder):
    dataset_obj.class_to_idx
    idx2class = {v: k for k, v in dataset_obj.class_to_idx.items()}
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    
    for _,j in dataloader_obj:
        y_idx = j.item()
        y_lbl = idx2class[y_idx]
        count_dict[str(y_lbl)] += 1
            
    return count_dict

def get_class_distribution(dataset_obj):
    idx2class = {v: k for k, v in dataset_obj.class_to_idx.items()}
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    
    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
            
    return count_dict

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight 

def get_subnet_dataloader(data_dir:str, 
                          subset_len:int = 1000, 
                          batch_size: int = 16, 
                          image_size: int = 96,
                          num_workers: int = 2,
                          shuffle: bool = True):
    # Reference : https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(data_dir)
    
    # Stratified Sampling for train and val
    from sklearn.model_selection import train_test_split
    train_idx, validation_idx = train_test_split(list(range(len(dataset.targets))),
                                                test_size=subset_len,
                                                random_state=999,
                                                shuffle=shuffle,
                                                stratify=dataset.targets)
    from torch.utils.data import Subset
    # Subset dataset for train and val
    # train_dataset = Subset(dataset, train_idx)
    
    # np.random.shuffle(validation_idx)
    # validation_idx = validation_idx[:subset_len]
    # val_dataset = Subset(dataset, validation_idx)
    
    # 변환을 적용한 데이터셋 생성
    val_dataset = CustomDataset(dataset, validation_idx, transform=val_transform)
    # DataLoader 생성
    data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True)
    return data_loader

def get_dataloader(
    dataset_dir: str = "vw_coco2014_96",
    batch_size: int = 16,
    image_size: int = 96,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader for training data.

    Parameters
    ----------
    cifar_10_dir: str
        Path to CIFAR10 data root in torchvision format.
    batch_size: int
        Batch size for dataloader.
    num_workers: int
        Number of subprocesses for data loading.
    shuffle: bool
        Flag for shuffling training data.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader for training data.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

        
    # 원본 데이터셋 로드 (변환 없이)
    dataset = ImageFolder(dataset_dir)
        
    # 데이터셋 분할
    validation_split = 0.1
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    train_indices, val_indices = random_split(dataset, [train_size, val_size])
    
    # 변환을 적용한 데이터셋 생성
    train_dataset = CustomDataset(train_indices, transform=train_transform)
    val_dataset = CustomDataset(val_indices, transform=val_transform)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, val_loader



from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader
def transform_cifar10():
    "transforms for the cifar 10"
    return Compose(
        [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
def sample_from_class(data_set, k):
    """
    function to sample data and their labels from a dataset in pytorch in
    a stratified manner
    Args
    ----
    data_set
    k: the number of samples that will be accuimulated in the new slit
    Returns
    -----
    train_dataset
    val_dataset
    """
    class_counts = {}
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for data, label in data_set:
        class_i = label.item() if isinstance(label, torch.Tensor) else label
        class_counts[class_i] = class_counts.get(class_i, 0) + 1
        if class_counts[class_i] <= k:
            train_data.append(data)
            train_label.append(label)
        else:
            test_data.append(data)
            test_label.append(label)

    train_data = torch.stack(train_data)
    train_label = torch.tensor(train_label, dtype=torch.int64)
    test_data = torch.stack(test_data)
    test_label = torch.tensor(test_label, dtype=torch.int64)

    return (
        TensorDataset(train_data, train_label),
        TensorDataset(test_data, test_label),
    )
def load_cifar(data_dir="E:/2_Quantization/torch2cmsis/examples/cifar/data/data_cifar10",
                batch_size=32,
               num_workers=4,
               val_num=500):
    train_set = CIFAR10(
        root="E:/2_Quantization/torch2cmsis/examples/cifar/data/data_cifar10",
        train=True,
        transform=transform_cifar10(),
        download=True
        )
    val_set, tr_set = sample_from_class(train_set, val_num)
    test_set = CIFAR10(
        root="E:/2_Quantization/torch2cmsis/examples/cifar/data/data_cifar10",
        train=False,
        transform=transform_cifar10(),
        download=True
        )
    
    dataloaders = {
        i: DataLoader(
            sett, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        for i, sett in zip(["train", "val", "test"], [train_set, val_set, test_set])
    }
    return dataloaders['train'], dataloaders['val'], dataloaders['test']