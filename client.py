# client.py - Windows-compatible version

import flwr as fl
import tensorflow as tf
import numpy as np
import argparse
import os
import time
import sys

# Import từ các module khác
from model import create_model, get_client_data, TrainingCallback
from config import CLIENT_CONFIG

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower client")
parser.add_argument("--client_id", type=int, choices=[1, 2, 3, 4], required=True, 
                    help="Client ID (1-4)")
parser.add_argument("--server_address", type=str, default=None,
                    help="Server address (IP:port). If not provided, will read from server_port.txt")
parser.add_argument("--retries", type=int, default=CLIENT_CONFIG["max_retries"],
                    help="Number of connection retries")
args = parser.parse_args()

# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, client_id):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.client_id = client_id
        
        # Create client directory for saving models
        self.save_dir = f"{CLIENT_CONFIG['client_data_dir_prefix']}{client_id}{CLIENT_CONFIG['client_data_dir_suffix']}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"Client {client_id} initialized and ready")

    def get_parameters(self, config):
        print(f"Client {self.client_id}: Retrieving model parameters")
        return [np.array(param) for param in self.model.get_weights()]

    def fit(self, parameters, config):
        print(f"Client {self.client_id}: Starting local training")
        
        # Update local model with global parameters
        self.model.set_weights([np.array(param) for param in parameters])
        
        # Get current round from config
        round_num = config.get("round_num", 0)
        print(f"Client {self.client_id}: Training round {round_num}")
        
        # Train the model
        batch_size = CLIENT_CONFIG["batch_size"]
        epochs = CLIENT_CONFIG["epochs_per_round"]
        
        history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=CLIENT_CONFIG["validation_split"],
            verbose=1,
            callbacks=[TrainingCallback()]
        )
        
        # Save local model
        model_path = os.path.join(self.save_dir, f"model_round_{round_num}.h5")
        self.model.save(model_path)
        print(f"Client {self.client_id}: Saved local model to {model_path}")
        
        # Return updated model parameters, train size, and metrics
        print(f"Client {self.client_id}: Completed local training")
        return (
            [np.array(param) for param in self.model.get_weights()],
            len(self.x_train),
            {"loss": history.history["loss"][-1], 
             "accuracy": history.history["sparse_categorical_accuracy"][-1]}
        )

    def evaluate(self, parameters, config):
        print(f"Client {self.client_id}: Evaluating global model")
        
        # Update local model with global parameters
        self.model.set_weights([np.array(param) for param in parameters])
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        print(f"Client {self.client_id}: Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(self.x_test), {"sparse_categorical_accuracy": accuracy}

# Main execution
def main():
    client_id = args.client_id
    server_address = args.server_address
    
    # If server_address is not provided, try to read from file
    if server_address is None:
        try:
            with open("server_port.txt", "r") as f:
                port = f.read().strip()
            server_address = f"localhost:{port}"
        except Exception as e:
            print("ERROR: Could not read server port from server_port.txt")
            print("Please make sure the server is running and server_port.txt exists")
            print("Or provide the server address manually using --server_address")
            sys.exit(1)
    
    max_retries = args.retries
    
    print(f"Starting client {client_id}, connecting to {server_address}")
    
    # Load data partition for this client
    (x_train, y_train), (x_test, y_test) = get_client_data(client_id)
    
    # Create and compile model
    model = create_model()
    
    # Start Flower client with retry mechanism
    client = MnistClient(model, x_train, y_train, x_test, y_test, client_id)
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"Client {client_id}: Attempting to connect to server (attempt {retry_count+1})")
            fl.client.start_numpy_client(server_address=server_address, client=client)
            break
        except Exception as e:
            retry_count += 1
            wait_time = min(30, 2 ** retry_count)  # Exponential backoff
            print(f"Connection error: {e}")
            print(f"Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})")
            time.sleep(wait_time)
    
    if retry_count >= max_retries:
        print(f"Failed to connect after {max_retries} attempts. Please check the server status.")
        sys.exit(1)

if __name__ == "__main__":
    main()