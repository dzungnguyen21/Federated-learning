import flwr as fl
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
import numpy as np
from flwr.common import Metrics, Parameters
from flwr.server.client_proxy import ClientProxy
import os
import socket
import sys
import time

# Import từ các module khác
from model import create_model
from config import SERVER_CONFIG

# Check if server is already running
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port=SERVER_CONFIG["min_port"], max_port=SERVER_CONFIG["max_port"]):
    for port in range(start_port, max_port + 1):
        if not is_port_in_use(port):
            return port
    return None

# Define the aggregation strategy
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0
    
    def aggregate_fit(self, server_round, results, failures):
        # Call the parent's aggregate_fit
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Save the global model
        if aggregated_parameters is not None:
            print(f"Saving global model for round {server_round}")
            os.makedirs(SERVER_CONFIG["server_data_dir"], exist_ok=True)
            
            # Chuyển đổi Parameters thành danh sách numpy arrays
            weights_list = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Lưu weights vào file
            np.savez(
                f"{SERVER_CONFIG['server_data_dir']}/global_model_round_{server_round}.npz", 
                *weights_list
            )
        
        self.current_round = server_round
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        print(f"Round {server_round} evaluation metrics: {aggregated_metrics}")
        return aggregated_metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Calculate weighted average of metrics
    accuracies = [num_examples * m["sparse_categorical_accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"sparse_categorical_accuracy": sum(accuracies) / sum(examples)}

# Start the server
def start_server():
    # Find available port
    port = find_available_port()
    if port is None:
        print(f"ERROR: No available ports found between {SERVER_CONFIG['min_port']} and {SERVER_CONFIG['max_port']}")
        print("Please check if:")
        print("1. Another instance of the server is running")
        print("2. Another application is using these ports")
        print("3. The previous server process didn't shut down properly")
        print("\nTo fix this:")
        print("1. Close any running server instances")
        print(f"2. Use 'netstat -ano | findstr :{SERVER_CONFIG['min_port']}' to find processes using port {SERVER_CONFIG['min_port']}")
        print("3. Use 'taskkill /PID <PID> /F' to kill the process")
        sys.exit(1)
    
    # Create server directory for logs and models
    os.makedirs(SERVER_CONFIG["server_data_dir"], exist_ok=True)
    
    # Initialize model and get initial parameters
    model = create_model()
    initial_parameters = [np.array(param) for param in model.get_weights()]
    
    # Define strategy
    strategy = SaveModelStrategy(
        fraction_fit=SERVER_CONFIG["fraction_fit"],
        fraction_evaluate=SERVER_CONFIG["fraction_evaluate"],
        min_fit_clients=SERVER_CONFIG["min_clients"],
        min_evaluate_clients=SERVER_CONFIG["min_clients"],
        min_available_clients=SERVER_CONFIG["min_clients"],
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
    )
    
    # Start server
    server_address = f"{SERVER_CONFIG['host']}:{port}"
    print(f"Starting Flower server on {server_address}")
    print("Waiting for clients to connect...")
    print(f"Make sure to run clients with: python client.py --client_id <1-4> --server_address localhost:{port}")
    
    # Save port to file for clients to read
    with open("server_port.txt", "w") as f:
        f.write(str(port))
    
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=SERVER_CONFIG["num_rounds"]),
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server()
