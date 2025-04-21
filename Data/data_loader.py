import torch
from torchvision import datasets, transforms
import numpy as np
import sys
import os

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Components.load_config import Path

def load_mnist_data():
    """
    Load MNIST dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset

def load_cifar10_data():
    """
    Load CIFAR-10 dataset
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
    
    # Load training data with augmentation
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Load test data
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
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
        # print("Loading MNIST dataset")
        return load_mnist_data()
    elif dataset_name == 'cifar10':
        # print("Loading CIFAR-10 dataset")
        return load_cifar10_data()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")