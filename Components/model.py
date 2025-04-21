import torch
import torch.nn as nn
import torch.nn.functional as F
from Components.load_config import Path

class MnistCNN(nn.Module):
    def __init__(self, device=None):
        super(MnistCNN, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
        
        # Move model to device
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CifarCNN(nn.Module):
    def __init__(self, device=None):
        super(CifarCNN, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        
        # Move model to device
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
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
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = config['model']['name']
    dataset = config['data']['dataset']
    
    if model_name == 'cnn':
        if dataset == 'mnist':
            return MnistCNN(device=device)
        elif dataset == 'cifar10':
            return CifarCNN(device=device)
    elif model_name == 'cifar_cnn':
        return CifarCNN(device=device)
    
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