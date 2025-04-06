import torch
from torchvision import datasets, transforms
import numpy as np
import yaml
import os

def load_config():
    with open('d:/AI/S2_Y3/Du_an/FL_1/config.yaml', 'r') as file:
        return yaml.safe_load(file)

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
    config = load_config()
    dataset_name = config['data']['dataset'].lower()
    
    if dataset_name == 'mnist':
        return load_mnist_data()
    elif dataset_name == 'cifar10':
        return load_cifar10_data()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
