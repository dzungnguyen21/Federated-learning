import torch
from torchvision import datasets, transforms
import numpy as np
import sys
import os
import shutil

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Components.load_config import Path

def load_mnist_data():
    """
    Load MNIST dataset using PyTorch's datasets.MNIST
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Use 'data' folder in project directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # Load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset

def load_cifar10_data():
    """
    Load CIFAR-10 dataset using PyTorch's datasets.CIFAR10
    """
    # Define transforms for training data
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Define transforms for test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Use 'data' folder in project directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # Load training data
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Load test data
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    print(f"CIFAR-10 dataset loaded: {len(train_dataset)} training and {len(test_dataset)} test samples")
    return train_dataset, test_dataset

def load_dataset():
    """
    Load dataset based on configuration
    """
    # Use Path class to load config
    config_loader = Path()
    config = config_loader.config
    
    dataset_name = config['data']['dataset'].lower()
    
    if dataset_name == 'mnist':
        print("Loading MNIST dataset...")
        return load_mnist_data()
    elif dataset_name == 'cifar10':
        print("Loading CIFAR-10 dataset...")
        return load_cifar10_data()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

def cleanup_dataset_files():
    """
    Cleanup downloaded dataset files to avoid pushing them to Git
    This can be called after training if you want to clean up disk space
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if os.path.exists(data_dir):
        print(f"Cleaning up dataset files in {data_dir}...")
        shutil.rmtree(data_dir)
        print("Dataset files removed successfully!")

if __name__ == "__main__":
    # Test data loading
    train_dataset, test_dataset = load_dataset()
    print(f"Dataset loaded successfully with {len(train_dataset)} training samples and {len(test_dataset)} test samples")
