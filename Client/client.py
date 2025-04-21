import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Components.model import get_model, get_model_parameters, set_model_parameters
from Components.load_config import Path

class Client:
    def __init__(self, client_id, train_loader):
        """
        Initialize client with its data
        """
        self.client_id = client_id
        self.train_loader = train_loader
        
        # Determine device - use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        config_loader = Path()
        self.config = config_loader.config
        
        # Initialize model and move to device
        self.model = get_model()  # Model is moved to device in get_model()
        
        # Set optimizer and criterion
        if self.config['client']['optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=self.config['training']['momentum']
            )
        elif self.config['client']['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate']
            )
        
        if self.config['client']['criterion'].lower() == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        
        # FedProx regularization
        self.fedprox = self.config['client']['fedprox']['enabled']
        if self.fedprox:
            self.mu = self.config['client']['fedprox']['mu']
            self.global_params = None
    
    def update_parameters(self, parameters):
        """
        Update client model with global parameters
        """
        set_model_parameters(self.model, parameters)
        
        # Store global parameters for FedProx if enabled
        if self.fedprox:
            self.global_params = parameters
    
    def compute_proximal_term(self):
        """
        Compute proximal term for FedProx
        """
        if not self.fedprox or self.global_params is None:
            return 0.0
        
        # Get current local model parameters
        local_params = [p.data for p in self.model.parameters()]
        
        # Compute L2 distance
        proximal_term = 0.0
        for local, global_param in zip(local_params, self.global_params):
            # Move global parameter to the same device as local parameter
            global_param = global_param.to(local.device)
            proximal_term += torch.sum((local - global_param) ** 2)
        
        return 0.5 * self.mu * proximal_term
    
    def train(self):
        """
        Perform local training for one epoch
        """
        local_epochs = self.config['training']['local_epochs']
        
        # Set model to training mode
        self.model.train()
        
        # Training loop
        for epoch in range(local_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(data)
                
                # Compute loss
                loss = self.criterion(outputs, target)
                
                # Add proximal term if FedProx is enabled
                if self.fedprox and self.global_params is not None:
                    proximal_term = self.compute_proximal_term()
                    loss += proximal_term
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            # Print epoch statistics
            accuracy = 100.0 * correct / total
            print(f"Client {self.client_id}, Epoch {epoch+1}/{local_epochs}: "
                  f"Loss: {running_loss/len(self.train_loader):.4f}, "
                  f"Accuracy: {accuracy:.2f}%")
        
        # Return updated parameters (moved to CPU for serialization)
        return get_model_parameters(self.model)
