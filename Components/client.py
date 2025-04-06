import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import sys
import os
import copy

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Components.model import get_model, get_model_parameters, set_model_parameters

class Client:
    def __init__(self, client_id, train_loader):
        """
        Initialize client with its data
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open('d:/AI/S2_Y3/Du_an/FL_1/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Set up model and training parameters
        self.model = get_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = self.config['training']['learning_rate']
        self.local_epochs = self.config['training']['local_epochs']
        
        # Configure optimizer based on config
        optimizer_name = self.config['client']['optimizer'].lower()
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )
        else:  # Default to SGD
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.learning_rate,
                momentum=self.config['training']['momentum']
            )
        
        # FedProx settings
        self.fedprox_enabled = self.config['client'].get('fedprox', {}).get('enabled', False)
        self.mu = self.config['client'].get('fedprox', {}).get('mu', 0.01)
        self.global_model_params = None
    
    def update_parameters(self, parameters):
        """
        Update local model parameters from server
        """
        # Save global model parameters if FedProx is enabled
        if self.fedprox_enabled:
            # Create a deep copy of parameters for FedProx regularization
            # Move global params to the same device as local model
            self.global_model_params = [param.clone().to(self.device) for param in parameters]
            
        set_model_parameters(self.model, parameters)
    
    def proximal_term(self):
        """
        Calculate the proximal term for FedProx
        """
        if not self.fedprox_enabled or self.global_model_params is None:
            return 0.0
            
        proximal_term = 0.0
        for local_param, global_param in zip(self.model.parameters(), self.global_model_params):
            # Ensure both tensors are on the same device
            if local_param.device != global_param.device:
                global_param = global_param.to(local_param.device)
                
            proximal_term += ((local_param - global_param) ** 2).sum()
        
        return 0.5 * self.mu * proximal_term
    
    def train(self):
        """
        Train the model on local data
        """
        self.model.train()
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                
                # Standard cross-entropy loss
                loss = self.criterion(output, target)
                
                # Add proximal term if FedProx is enabled
                if self.fedprox_enabled:
                    prox_term = self.proximal_term()
                    loss += prox_term
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            accuracy = 100.0 * correct / total
            
            # Print training progress
            algorithm = "FedProx" if self.fedprox_enabled else "Standard"
            optimizer_name = self.config['client']['optimizer']
            print(f"Client {self.client_id} - Epoch {epoch+1} [{algorithm}, {optimizer_name}]: "
                  f"Loss: {epoch_loss/len(self.train_loader):.4f}, Accuracy: {accuracy:.2f}%")
        
        # Return updated model parameters
        return get_model_parameters(self.model)
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on test data
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        
        return test_loss, accuracy
