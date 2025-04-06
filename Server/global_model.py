import torch
import random
import sys
import os
from Components.load_config import Path

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Components.model import get_model, get_model_parameters, set_model_parameters

class Server:
    def __init__(self, test_loader):
        """
        Initialize server with global model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = get_model().to(self.device)
        self.test_loader = test_loader
        
        # Load configuration using Path class
        config_loader = Path()
        self.config = config_loader.config
            
        self.clients_per_round = max(1, int(self.config['server']['fraction_clients'] * self.config['data']['num_clients']))
        self.aggregation = self.config['server']['aggregation'].lower()
    
    def select_clients(self, client_list):
        """
        Select a fraction of clients randomly
        """
        return random.sample(client_list, self.clients_per_round)
    
    def aggregate_parameters(self, client_parameters, algorithm=None):
        """
        Aggregate model parameters from clients (FedAvg or FedProx)
        """
        # If algorithm is provided, use it; otherwise use config
        if algorithm is None:
            algorithm = self.aggregation
        
        # FedAvg and FedProx use the same aggregation method
        # The difference is in how clients optimize locally
        
        # Get model shape from current global model
        global_parameters = get_model_parameters(self.global_model)
        
        # Initialize aggregated parameters with zeros
        aggregated_parameters = []
        for param in global_parameters:
            aggregated_parameters.append(torch.zeros_like(param))
        
        # Simple averaging of client parameters
        num_clients = len(client_parameters)
        for client_idx, params in enumerate(client_parameters):
            for i, param in enumerate(params):
                # Ensure parameters are on CPU for aggregation
                if param.device != aggregated_parameters[i].device:
                    param = param.to(aggregated_parameters[i].device)
                aggregated_parameters[i] += param / num_clients
        
        # Update global model
        set_model_parameters(self.global_model, aggregated_parameters)
        
        return get_model_parameters(self.global_model)
    
    def evaluate(self):
        """
        Evaluate global model on test dataset
        """
        self.global_model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_loss /= len(self.test_loader)
        accuracy = 100.0 * correct / total
        
        return test_loss, accuracy
    
    def get_global_parameters(self):
        """
        Get current global model parameters
        """
        return get_model_parameters(self.global_model)
