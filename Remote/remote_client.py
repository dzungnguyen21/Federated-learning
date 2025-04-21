import sys
import os
import torch
import requests
import json
import argparse
import time
from io import BytesIO
import base64

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Client.client import Client
from Data.data_loader import load_dataset
from Data.data_split import create_client_data
from Components.model import get_model, get_model_parameters, set_model_parameters
from Components.load_config import Path
from torch.utils.data import DataLoader

def tensor_to_base64(tensor):
    """
    Convert a PyTorch tensor to base64 encoded string
    """
    # Ensure tensor is on CPU before serialization
    tensor_cpu = tensor.cpu()
    buffer = BytesIO()
    torch.save(tensor_cpu, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def base64_to_tensor(b64_string):
    """
    Convert a base64 encoded string back to PyTorch tensor
    """
    buffer = BytesIO(base64.b64decode(b64_string))
    return torch.load(buffer, map_location=torch.device('cpu'))

class RemoteClient:
    def __init__(self, client_id, server_url):
        """
        Initialize remote client
        """
        self.client_id = client_id
        self.server_url = server_url
        # Determine device - use GPU if available 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Client {client_id} using device: {self.device}")
        
        # Load configuration using Path class
        config_loader = Path()
        self.config = config_loader.config
        
        # Load dataset
        train_dataset, _ = load_dataset()
        
        # Split data among clients
        num_clients = self.config['data']['num_clients']
        iid = self.config['data']['iid']
        client_loaders = create_client_data(train_dataset, num_clients, iid)
        
        # Initialize client with its data
        if self.client_id in client_loaders:
            self.client = Client(self.client_id, client_loaders[self.client_id])
        else:
            raise ValueError(f"Client ID {self.client_id} not found in dataset splits")
    
    def get_server_status(self):
        """
        Get current server status
        """
        try:
            response = requests.get(f"{self.server_url}/status")
            return response.json()
        except Exception as e:
            print(f"Error getting server status: {str(e)}")
            return None
    
    def get_global_model(self, current_round):
        """
        Get current global model from server
        """
        try:
            response = requests.get(f"{self.server_url}/get_model")
            data = response.json()
            
            if data['status'] == 'success':
                # Convert base64 strings back to tensors
                model_params = [base64_to_tensor(param) for param in data['model_params']]
                return model_params, data['round']
            elif data['status'] == 'completed':
                print("Training completed on server")
                return None, -1
            else:
                print(f"Error getting global model: {data.get('message', 'Unknown error')}")
                return None, data.get('round', -1)
                
        except Exception as e:
            print(f"Error getting global model: {str(e)}")
            return None, -1
    
    def submit_update(self, model_params, current_round):
        """
        Submit updated model parameters to server
        """
        try:
            # Convert tensors to base64 strings - ensures they're on CPU
            param_data = [tensor_to_base64(param) for param in model_params]
            
            # Prepare payload
            payload = {
                'client_id': self.client_id,
                'round': current_round,
                'model_params': param_data
            }
            
            # Send update to server
            response = requests.post(
                f"{self.server_url}/submit_update",
                json=payload
            )
            
            return response.json()
        except Exception as e:
            print(f"Error submitting update: {str(e)}")
            return None
    
    def train(self, max_rounds=None):
        """
        Perform federated learning with the server
        """
        if max_rounds is None:
            max_rounds = self.config['training']['global_rounds']
        
        current_round = 0
        
        while current_round < max_rounds:
            # Check server status
            status = self.get_server_status()
            if status is None:
                print("Could not connect to server. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            
            # Check if we're in sync with the server
            server_round = status.get('current_round', 0)
            if server_round != current_round:
                print(f"Client round ({current_round}) out of sync with server round ({server_round})")
                current_round = server_round
                
                if current_round >= max_rounds:
                    print("Training completed")
                    break
            
            # Get current global model
            global_params, model_round = self.get_global_model(current_round)
            
            if global_params is None:
                if model_round == -1:
                    # Training completed
                    break
                
                # Round mismatch - synchronize with server round
                if model_round >= 0 and model_round != current_round:
                    print(f"Synchronizing client round from {current_round} to {model_round}")
                    current_round = model_round
                    continue
                    
                # Error or round mismatch
                print("Failed to get global model. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            
            # Update local model with global parameters
            self.client.update_parameters(global_params)
            
            # Perform local training
            print(f"\n--- Client {self.client_id} training for round {current_round + 1} ---")
            updated_params = self.client.train()
            
            # Submit update to server
            result = self.submit_update(updated_params, current_round)
            
            if result is None:
                print("Failed to submit update. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            
            if 'status' in result and result['status'] == 'error':
                if 'message' in result and 'round' in result['message']:
                    # Server reported round mismatch - sync with server
                    print(f"Error: {result['message']}")
                    
                    # Extract server round from error message if possible
                    try:
                        error_msg = result['message']
                        if "server round" in error_msg:
                            server_round_str = error_msg.split("server round")[1].strip()
                            if server_round_str.startswith(str(current_round + 1)):
                                print(f"Synchronizing with server: moving to round {current_round + 1}")
                                current_round += 1
                    except Exception:
                        # If parsing fails, just check status again
                        pass
                        
                    time.sleep(2)
                    continue
            
            print(f"Update submitted successfully: {result.get('message', 'Unknown status')}")
            
            if result.get('round_completed', False):
                # Round completed, move to next round
                current_round += 1
                print(f"Round {current_round} completed.")
                
                # Display metrics if provided
                if 'metrics' in result:
                    print(f"Loss: {result['metrics'].get('loss', 'N/A'):.4f}, "
                          f"Accuracy: {result['metrics'].get('accuracy', 'N/A'):.2f}%")
            else:
                # Wait for other clients to complete the round
                clients_received = result.get('clients_received', 0)
                clients_needed = result.get('clients_needed', 1)
                print(f"Waiting for other clients... {clients_received}/{clients_needed} clients received")
                time.sleep(5)
        
        print("Training completed")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Remote Federated Learning Client')
    parser.add_argument('--client_id', type=int, required=True,
                        help='Client ID')
    parser.add_argument('--server_url', type=str, default='http://localhost:5000',
                        help='Server URL')
    parser.add_argument('--rounds', type=int, default=None,
                        help='Number of training rounds')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    try:
        client = RemoteClient(args.client_id, args.server_url)
        client.train(args.rounds)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")