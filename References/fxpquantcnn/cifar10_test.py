
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

from convert_float_fixed import ConvertFloatFixed
# CNN 모델 정의
class Net(nn.Module):
    def __init__(self, num_classes, input_shape=(3, 32, 32), path_trained_weights=None):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.num_outputs = num_classes
        self.path_trained_weights = path_trained_weights
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2304, 512)
        self.relu5 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = self.relu5(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x
    
    def get_fxp_model(self, quant_params):
        """Get CNN model with quantization layers for quantizing layer outputs/activations
        
        Args:
            quant_params (dict): Dictionary of layer names as keys with values of
                                [bw, f] to quantize the layer's output to

        Returns:
            nn.Module: PyTorch model with quantization layers
        """
        model = Net(num_classes=self.num_outputs, input_shape=self.input_shape)

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
        model = Net(num_classes= self.num_outputs, input_shape=self.input_shape)

        if self.path_trained_weights:
            model.load_state_dict(torch.load(self.path_trained_weights))

        return model


class LeNet(nn.Module):
    # ARM Example Model
    def __init__(self, num_channels=3, shape=(3, 32, 32), batch_size=4, model='arm'):
        super(LeNet,self).__init__()
        
        self.input_shape = shape
        self.batch_size = batch_size
        
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
        self.fc1_shape = self.get_shape()
        self.fc1 = nn.Linear(in_features=self.fc1_shape.numel(), out_features=10)
        
        
    def get_shape(self):
        sample = torch.randn(size=(self.batch_size, *self.input_shape))
        x = self.conv1(sample)
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
        
        return x.shape[1:]
    def forward(self, x):
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
        return x

from tqdm import tqdm
from torch.utils.data import DataLoader
@torch.no_grad()
def get_accuracy(
  model: nn.Module,
  dataloader: DataLoader,
  extra_preprocess = None
) -> float:
  model.eval()

  num_samples = 0
  num_correct = 0

  for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
    # Move the data from CPU to GPU
    inputs = inputs.cuda()
    if extra_preprocess is not None:
        for preprocess in extra_preprocess:
            inputs = preprocess(inputs)

    targets = targets.cuda()

    # Inference
    outputs = model(inputs)

    # Convert logits to class indices
    outputs = outputs.argmax(dim=1)

    # Update metrics
    num_samples += targets.size(0)
    num_correct += (outputs == targets).sum()

  return (num_correct / num_samples * 100).item()


if __name__ =='__main__':
    # 하이퍼파라미터 설정
    batch_size = 32
    num_classes = 10
    epochs = 100
    data_augmentation = True
    num_predictions = 20
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_arm_example.pth'

    # CIFAR-10 데이터셋 로드
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='E:/2_Quantization/torch2cmsis/examples/cifar/data/data_cifar10', 
                                            train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='E:/2_Quantization/torch2cmsis/examples/cifar/data/data_cifar10', 
                                           train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # net = Net(num_classes=num_classes)
    net = LeNet(num_channels=3, shape=(3, 32, 32), batch_size=batch_size, model='arm')
    from torchsummary import summary
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=0.0001, weight_decay=1e-6)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    # 모델 학습
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                print(f'[{epoch+1}/{epochs}, { i + 1}/{len(trainloader)}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
        acc = get_accuracy(net, testloader)
        print('Test accuracy: ', acc)
        
    print('Finished Training')

    # 모델 저장
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    torch.save(net.state_dict(), model_path)
    print('Saved trained model at %s ' % model_path)

    # 모델 평가
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
