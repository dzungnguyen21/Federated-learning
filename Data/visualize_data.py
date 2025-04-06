import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os
import yaml
from collections import Counter

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.data_loader import load_dataset
from Data.data_split import create_client_data, load_config

def get_client_label_distribution(client_loaders):
    """
    Get label distribution for each client
    """
    client_labels = {}
    
    for client_id, loader in client_loaders.items():
        labels = []
        for _, target in loader.dataset:
            if isinstance(target, torch.Tensor):
                labels.append(target.item())
            else:
                labels.append(target)
        
        # Count label frequencies
        client_labels[client_id] = Counter(labels)
    
    return client_labels

def plot_label_distribution(client_loaders):
    """
    Plot label distribution across clients
    """
    client_labels = get_client_label_distribution(client_loaders)
    num_clients = len(client_labels)
    num_classes = 10  # MNIST has 10 classes
    
    # Create figure
    fig, axs = plt.subplots(1, num_clients, figsize=(16, 4), sharey=True)
    
    # Plot distribution for each client
    for i, (client_id, label_counter) in enumerate(client_labels.items()):
        distribution = [label_counter.get(j, 0) for j in range(num_classes)]
        axs[i].bar(range(num_classes), distribution)
        axs[i].set_title(f'Client {client_id}')
        axs[i].set_xlabel('Label')
        if i == 0:
            axs[i].set_ylabel('Samples')
    
    plt.suptitle('Label Distribution Across Clients')
    plt.tight_layout()
    plt.savefig('label_distribution.png')
    plt.show()

def plot_sample_images(dataset, num_samples=10):
    """
    Plot sample images from the dataset
    """
    # Load config to determine dataset
    with open('d:/AI/S2_Y3/Du_an/FL_1/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    dataset_name = config['data']['dataset'].lower()
    
    # Create figure
    fig, axs = plt.subplots(2, 5, figsize=(12, 5))
    axs = axs.flatten()
    
    # Get random samples
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    # Get class names for CIFAR-10 if applicable
    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                     'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Plot images
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        
        # Handle different image formats
        if dataset_name == 'mnist':
            # For MNIST: single channel grayscale image
            img_display = img.squeeze().numpy()
            axs[i].imshow(img_display, cmap='gray')
            label_name = str(label)
        else:
            # For CIFAR-10: 3-channel color image
            img_display = img.permute(1, 2, 0).numpy()  # Change to [H, W, 3] for matplotlib
            
            # Un-normalize the image
            if dataset_name == 'cifar10':
                mean = np.array([0.4914, 0.4822, 0.4465])
                std = np.array([0.2470, 0.2435, 0.2616])
                img_display = img_display * std + mean
                img_display = np.clip(img_display, 0, 1)
                label_name = cifar_classes[label]
            else:
                label_name = str(label)
                
            axs[i].imshow(img_display)
            
        axs[i].set_title(f'Label: {label_name}')
        axs[i].axis('off')
    
    plt.suptitle(f'Sample {dataset_name.upper()} Images')
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.show()

def visualize_data():
    """
    Main function to visualize data
    """
    # Load configuration
    config = load_config()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load dataset
    train_dataset, _ = load_dataset()
    
    # Plot sample images
    plot_sample_images(train_dataset)
    
    # Create client data splits
    num_clients = config['data']['num_clients']
    iid = config['data']['iid']
    client_loaders = create_client_data(train_dataset, num_clients, iid)
    
    # Plot label distribution
    plot_label_distribution(client_loaders)
    
    # Print distribution information
    client_labels = get_client_label_distribution(client_loaders)
    print("\nLabel Distribution Summary:")
    for client_id, counter in client_labels.items():
        print(f"Client {client_id}:")
        for label in sorted(counter.keys()):
            class_name = label
            if config['data']['dataset'].lower() == 'cifar10':
                cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                                'dog', 'frog', 'horse', 'ship', 'truck']
                class_name = f"{label} ({cifar_classes[label]})"
                
            print(f"  Label {class_name}: {counter[label]} samples")
        print(f"  Total: {sum(counter.values())} samples")
        print()

if __name__ == "__main__":
    visualize_data()