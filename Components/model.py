import torch
import torch.nn as nn
import torch.nn.functional as F
from Components.load_config import Path

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 256 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def get_model():
    """
    Get the model based on configuration
    """
    config_loader = Path()
    config = config_loader.config
    
    model_name = config['model']['name']
    dataset = config['data']['dataset']
    
    if model_name == 'cnn':
        if dataset == 'mnist':
            return MnistCNN()
        elif dataset == 'cifar10':
            return CifarCNN()
    elif model_name == 'cifar_cnn':
        return CifarCNN()
    
    raise ValueError(f"Model {model_name} not supported for dataset {dataset}")

def get_model_parameters(model):
    """
    Get model parameters as a list of numpy arrays
    """
    return [param.data.clone().detach().cpu() for param in model.parameters()]

def set_model_parameters(model, parameters):
    """
    Set model parameters from a list of tensors
    """
    for param, data in zip(model.parameters(), parameters):
        # Make sure the data is on the same device as the model
        param.data.copy_(data.to(param.device))