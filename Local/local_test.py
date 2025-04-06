import sys
import os
import torch
from torch.utils.data import DataLoader
import argparse

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.data_loader import load_dataset
from Data.data_split import create_client_data
from Client.client import Client
from Server.global_model import Server
from Components.visualize_training import TrainingVisualizer
from Components.load_config import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning Training')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Enable visualization of training metrics')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load configuration using Path class
    config_loader = Path()
    config = config_loader.config
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Load dataset based on configuration
    train_dataset, test_dataset = load_dataset()
    
    # Create test data loader (central evaluation)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Split training data among clients
    num_clients = config['data']['num_clients']
    iid = config['data']['iid']
    client_loaders = create_client_data(train_dataset, num_clients, iid)
    
    # Initialize server
    server = Server(test_loader)
    
    # Initialize clients
    clients = []
    for client_id, loader in client_loaders.items():
        clients.append(Client(client_id, loader))
    
    # Initialize visualizer if enabled
    visualizer = TrainingVisualizer() if args.visualize else None
    
    # Print training configuration
    print("\n==== Training Configuration ====")
    print(f"Dataset: {config['data']['dataset']}")
    print(f"Model: {config['model']['name']}")
    print(f"Algorithm: {config['server']['aggregation']}")
    print(f"FedProx enabled: {config['client'].get('fedprox', {}).get('enabled', False)}")
    print(f"Optimizer: {config['client']['optimizer']}")
    print(f"Data distribution: {'IID' if config['data']['iid'] else 'Non-IID'}")
    print(f"Number of clients: {config['data']['num_clients']}")
    print(f"Local epochs: {config['training']['local_epochs']}")
    print(f"Global rounds: {config['training']['global_rounds']}")
    print("==============================\n")
    
    # Start federated learning process
    num_rounds = config['training']['global_rounds']
    
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
        
        # Update visualization if enabled
        if visualizer:
            visualizer.update_metrics(round_num, test_loss, accuracy)
    
    # Show plots if visualization is enabled
    if visualizer:
        try:
            visualizer.plot_global_metrics()
            visualizer.visualize_model_performance(server.global_model, test_loader)
        except Exception as e:
            print(f"Visualization error: {e}")
            print("Training completed successfully despite visualization error.")

if __name__ == "__main__":
    main()
