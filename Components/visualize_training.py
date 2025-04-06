import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch
from torch.utils.data import DataLoader

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.data_loader import load_mnist_data
from Data.data_split import create_client_data
from Client.client import Client
from Server.global_model import Server
from Components.model import get_model
from Components.load_config import Path

class TrainingVisualizer:
    def __init__(self):
        self.accuracy_history = []
        self.loss_history = []
        self.client_accuracy_history = {}
        self.client_loss_history = {}
    
    def update_metrics(self, round_num, test_loss, accuracy, client_metrics=None):
        """
        Update training metrics history
        """
        self.accuracy_history.append(accuracy)
        self.loss_history.append(test_loss)
        
        if client_metrics:
            for client_id, (loss, acc) in client_metrics.items():
                if client_id not in self.client_accuracy_history:
                    self.client_accuracy_history[client_id] = []
                    self.client_loss_history[client_id] = []
                
                self.client_accuracy_history[client_id].append(acc)
                self.client_loss_history[client_id].append(loss)
    
    def plot_global_metrics(self):
        """
        Plot global model metrics over training rounds
        """
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.accuracy_history) + 1), self.accuracy_history, marker='o')
        plt.title('Global Model Accuracy')
        plt.xlabel('Round')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o', color='r')
        plt.title('Global Model Loss')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('global_metrics.png')
        plt.show()
    
    def plot_client_metrics(self):
        """
        Plot client metrics over training rounds
        """
        if not self.client_accuracy_history:
            print("No client metrics available")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Plot client accuracies
        plt.subplot(1, 2, 1)
        for client_id, accuracies in self.client_accuracy_history.items():
            plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', label=f'Client {client_id}')
        plt.title('Client Accuracies')
        plt.xlabel('Round')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # Plot client losses
        plt.subplot(1, 2, 2)
        for client_id, losses in self.client_loss_history.items():
            plt.plot(range(1, len(losses) + 1), losses, marker='o', label=f'Client {client_id}')
        plt.title('Client Losses')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('client_metrics.png')
        plt.show()
    
    def visualize_model_performance(self, model, test_loader):
        """
        Visualize model performance on test dataset
        """
        # Load config using Path class
        config_loader = Path()
        config = config_loader.config
        
        dataset = config['data']['dataset'].lower()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        # Get predictions for a batch of test data
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
        
        # Plot images with predictions
        plt.figure(figsize=(12, 8))
        for i in range(min(16, len(images))):
            plt.subplot(4, 4, i + 1)
            
            # Handle different image formats (MNIST vs CIFAR)
            img = images[i].cpu()
            
            if dataset == 'mnist':
                # For MNIST: single channel grayscale image
                img = img.squeeze().numpy()
                plt.imshow(img, cmap='gray')
            else:
                # For CIFAR-10: 3-channel color image [3, H, W]
                img = img.permute(1, 2, 0).numpy()  # Change to [H, W, 3] for matplotlib
                
                # Un-normalize the image (approximate reversal of normalization)
                if dataset == 'cifar10':
                    mean = np.array([0.4914, 0.4822, 0.4465])
                    std = np.array([0.2470, 0.2435, 0.2616])
                    img = img * std + mean
                    
                # Clip values to valid range for display
                img = np.clip(img, 0, 1)
                plt.imshow(img)
            
            # Set title color based on prediction correctness
            correct = predicted[i] == labels[i]
            color = "green" if correct else "red"
            plt.title(f"True: {labels[i].item()}\nPred: {predicted[i].item()}", color=color)
            plt.axis('off')
        
        dataset_name = config['data']['dataset'].upper()
        plt.suptitle(f'Model Predictions on {dataset_name} Test Data')
        plt.tight_layout()
        plt.savefig('model_predictions.png')
        plt.show()
    
    def compare_iid_vs_noniid(self):
        """
        Compare training with IID vs non-IID data distribution
        """
        plt.figure(figsize=(15, 7))
        
        # Load configuration using Path class
        config_loader = Path()
        config = config_loader.config
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Load MNIST dataset
        train_dataset, test_dataset = load_mnist_data()
        
        # Create test data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False
        )
        
        # List to store accuracy histories
        iid_accuracy = []
        noniid_accuracy = []
        
        # Train with IID data
        print("\n--- Training with IID data ---")
        iid_accuracy = run_federated_learning(True, test_loader, train_dataset)
        
        # Train with non-IID data
        print("\n--- Training with non-IID data ---")
        noniid_accuracy = run_federated_learning(False, test_loader, train_dataset)
        
        # Plot comparison
        plt.plot(range(1, len(iid_accuracy) + 1), iid_accuracy, marker='o', label='IID')
        plt.plot(range(1, len(noniid_accuracy) + 1), noniid_accuracy, marker='s', label='Non-IID')
        plt.title('IID vs Non-IID Data Distribution')
        plt.xlabel('Round')
        plt.ylabel('Global Model Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('iid_vs_noniid.png')
        plt.show()

def run_federated_learning(iid, test_loader, train_dataset, num_rounds=5):
    """
    Run federated learning for comparison purposes
    """
    # Load configuration using Path class
    config_loader = Path()
    config = config_loader.config
    
    # Split training data among clients
    num_clients = config['data']['num_clients']
    client_loaders = create_client_data(train_dataset, num_clients, iid)
    
    # Initialize server
    server = Server(test_loader)
    
    # Initialize clients
    clients = []
    for client_id, loader in client_loaders.items():
        clients.append(Client(client_id, loader))
    
    # Training history
    accuracy_history = []
    
    # Start federated learning process
    for round_num in range(num_rounds):
        print(f"\n----- Round {round_num + 1}/{num_rounds} -----")
        
        # Select clients for this round
        selected_clients = server.select_clients(clients)
        print(f"Selected {len(selected_clients)} clients for training")
        
        # Get current global model parameters
        global_parameters = server.get_global_parameters()
        
        # Train on selected clients
        client_parameters = []
        for client in selected_clients:
            # Update client model with global parameters
            client.update_parameters(global_parameters)
            
            # Perform local training
            updated_parameters = client.train()
            client_parameters.append(updated_parameters)
        
        # Aggregate updated parameters from clients
        server.aggregate_parameters(client_parameters)
        
        # Evaluate global model
        test_loss, accuracy = server.evaluate()
        print(f"Global model - Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        accuracy_history.append(accuracy)
    
    return accuracy_history

def modify_local_test_for_visualization():
    """
    This function explains how to modify local_test.py to include visualization
    """
    print("""
    To add visualization to local_test.py:
    
    1. Import the visualizer at the top:
       from Components.visualize_training import TrainingVisualizer
    
    2. Create a visualizer instance before the training loop:
       visualizer = TrainingVisualizer()
    
    3. Update metrics after each round:
       visualizer.update_metrics(round_num, test_loss, accuracy)
    
    4. Plot results after training:
       visualizer.plot_global_metrics()
       visualizer.visualize_model_performance(server.global_model, test_loader)
    """)

if __name__ == "__main__":
    # Show how to modify local_test.py
    modify_local_test_for_visualization()
    
    # Create visualizer and show IID vs non-IID comparison
    visualizer = TrainingVisualizer()
    visualizer.compare_iid_vs_noniid()
