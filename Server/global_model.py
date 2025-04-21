import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Components.model import get_model, get_model_parameters, set_model_parameters
from Components.load_config import Path

class Server:
    def __init__(self, test_loader):
        """
        Initialize FL server with global model
        """
        # Determine device - use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Server using device: {self.device}")
        
        # Load config
        config_loader = Path()
        self.config = config_loader.config
        
        # Initialize model
        self.model = get_model()
        
        # Set loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Store test data loader
        self.test_loader = test_loader
        
        # Store aggregated updates
        self.updates_received = 0
    
    def get_global_parameters(self):
        """
        Get current global model parameters
        """
        return get_model_parameters(self.model)
    
    def aggregate_parameters(self, client_parameters_list):
        """
        Aggregate client parameters using FedAvg
        """
        # Create a list of parameters
        # Each element in client_parameters_list is a list of parameter tensors
        
        # Initialize with first client's parameters shape
        num_parameters = len(client_parameters_list[0])
        
        # Calculate average of all client parameters
        # First, sum all client parameters
        aggregated_parameters = []
        for i in range(num_parameters):
            aggregated_parameters.append(
                torch.stack([client_params[i] for client_params in client_parameters_list]).mean(dim=0)
            )
        
        # Set the aggregated parameters to the global model
        set_model_parameters(self.model, aggregated_parameters)
        
        # Increment number of updates received
        self.updates_received += 1
    
    def evaluate(self):
        """
        Evaluate the global model on test data
        """
        self.model.eval()  # Set model to evaluation mode
        test_loss = 0
        correct = 0
        total = 0
        
        # Disable gradient computation for efficiency
        with torch.no_grad():
            for data, target in self.test_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Sum up batch loss
                test_loss += self.criterion(output, target).item()
                
                # Get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                
                # Count correct predictions
                total += target.size(0)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Calculate average loss and accuracy
        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / total
        
        return test_loss, accuracy
