import flwr as fl
import tensorflow as tf
from typing import List, Tuple
import numpy as np
from flwr.common import Metrics
import os
import socket
import sys

# Check if server is already running
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Define the model architecture
def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

# Define strategy with initial parameters
def get_strategy():
    # Initialize model
    model = get_model()
    
    # Get initial parameters as numpy arrays
    initial_parameters = [np.array(param) for param in model.get_weights()]
    
    # Define metric aggregation function
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["sparse_categorical_accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"sparse_categorical_accuracy": sum(accuracies) / sum(examples)}
    
    # Create strategy with initial parameters
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
    )

# Main function to start server
def main():
    # Server port
    port = 8080
    
    # Check if server is already running
    if is_port_in_use(port):
        print(f"ERROR: Server already running on port {port}")
        print("Terminate the existing server process before starting a new one")
        sys.exit(1)
    
    # Create server directory for logs and models
    os.makedirs("server_data", exist_ok=True)
    
    # Get strategy with initial parameters
    strategy = get_strategy()
    
    # Start server with proper config
    server_address = "0.0.0.0:8080"  # Listen on all interfaces
    
    print(f"Starting Flower server on {server_address}")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()