import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
import os
import sys

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Components.load_config import Path

def iid_split(dataset, num_clients):
    """
    Split dataset in IID fashion among clients
    """
    # Use Path class to load config
    config_loader = Path()
    config = config_loader.config
    
    samples_per_client = config['data']['samples_per_client']
    num_samples = len(dataset)
    print(f"Total samples: {num_samples}")
    
    # Generate random indices for each client
    indices = np.random.permutation(num_samples)
    client_datasets = {}
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = min((i + 1) * samples_per_client, num_samples)
        
        if start_idx < num_samples:
            client_datasets[i] = Subset(dataset, indices[start_idx:end_idx])
    
    return client_datasets

def non_iid_split(dataset, num_clients, num_shards=200):
    """
    Split dataset in non-IID fashion among clients
    Each client gets data from specific classes (label-based non-IID)
    """
    # Use Path class to load config
    config_loader = Path()
    config = config_loader.config
    
    samples_per_client = config['data']['samples_per_client']
    num_classes = config['model']['num_classes']
    
    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
        if isinstance(labels, list):
            labels = torch.tensor(labels)
    else:
        labels = dataset.train_labels
        
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    # Sort data by labels
    label_indices = {}
    for i in range(num_classes):
        label_indices[i] = np.where(labels == i)[0]
    
    # Assign shards to clients
    shard_size = len(dataset) // num_shards
    shards_per_client = num_shards // num_clients
    
    client_datasets = {}
    shard_indices = np.random.permutation(num_shards)
    
    for client_idx in range(num_clients):
        client_shards = shard_indices[client_idx * shards_per_client:(client_idx + 1) * shards_per_client]
        client_indices = []
        
        # Collect indices for all shards assigned to this client
        for shard in client_shards:
            class_idx = shard % num_classes
            class_start = (shard // num_classes) * shard_size
            class_indices = label_indices[class_idx][class_start:class_start + shard_size]
            client_indices.extend(class_indices)
        
        # Limit to samples_per_client if specified
        if samples_per_client > 0:
            client_indices = client_indices[:samples_per_client]
            
        client_datasets[client_idx] = Subset(dataset, client_indices)
    
    return client_datasets

def create_client_data(dataset, num_clients, iid=False):
    """
    Create data loaders for each client
    """
    # Use Path class to load config
    config_loader = Path()
    config = config_loader.config
    
    batch_size = config['training']['batch_size']
    
    if iid:
        print("Creating IID data split")
        client_datasets = iid_split(dataset, num_clients)
    else:
        print("Creating non-IID data split")
        client_datasets = non_iid_split(dataset, num_clients)
    
    # Create data loaders
    client_loaders = {}
    for client_idx, client_dataset in client_datasets.items():
        client_loaders[client_idx] = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True
        )
    
    return client_loaders
