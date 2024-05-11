from model import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp

def quantize_array(x, bit_depth=8):
    min_x = x.min() 
    max_x = x.max()

    #find number of integer bits to represent this range
    int_bits = int(np.ceil(np.log2(max(abs(min_x),abs(max_x)))))
    
    if int_bits < 0: 
        int_bits = 0

    frac_bits = bit_depth - 1 - int_bits #remaining bits are fractional bits (1-bit for sign)

    #floating point weights are scaled and rounded to [-2^(bit-1),2^(bit-1)-1], which are used in 
    #the fixed-point operations on the actual hardware (i.e., microcontroller)
    q_x = np.round(x*(2**frac_bits))

    #To quantify the impact of quantized weights, scale them back to
    # original range to run inference using quantized weights
    fp_x = q_x/(2**frac_bits)
    
    return q_x, fp_x, int_bits, frac_bits


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))

def initialize_weights(model:nn.Module):
    # track all layers
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    batch_size = 64
    test_batch_size = 64
    seed = 1
    epochs = 15
    lr = 0.01
    momentum = 0.5
    save_model = True
    case = 'qfmt'
    # zero scale

    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('data', train=True, download=True, 
    #                    transform=transforms.Compose([
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=test_batch_size, shuffle=True, num_workers=1, pin_memory=True
    # )
    import torchvision
    def get_cifar10_loader():
        print('=> loading cifar10 data...')
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        train_dataset = torchvision.datasets.CIFAR10(
            root='E:/2_Quantization/torch2cmsis/examples/cifar/data/data_cifar10',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        test_dataset = torchvision.datasets.CIFAR10(
            root='E:/2_Quantization/torch2cmsis/examples/cifar/data/data_cifar10',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
        return trainloader, testloader
    train_loader, test_loader = get_cifar10_loader()
    ## Case 1
    # model = Net().to(device) #68%
    model = NetBN().to(device) #70%
    # else:
    ## Case 2(Proposed Idea)
    # model = QNet().to(device) #70%
    # model = QNetBN().to(device) #w/o weight normalization 70%
    # model = QNetBN().to(device) #w/ weight normalization 72%
    initialize_weights(model)
        
    

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    
    if save_model:
        if not osp.exists('ckpt'):
            os.makedirs('ckpt')
        # if using_bn:
        #     torch.save(model.state_dict(), 'ckpt/mnist_cnnbn.pt')
        # else:
        torch.save(model.state_dict(), 'ckpt/cifar_cnn_qfmt.pt')