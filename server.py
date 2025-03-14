# server.py - Run this on your central server

import flwr as fl
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
import numpy as np
from flwr.common import Metrics, Parameters
from flwr.server.client_proxy import ClientProxy

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
            np.savez(f"global_model_round_{server_round}.npz", 
                     *[param.numpy() for param in aggregated_parameters])
        
        self.current_round = server_round
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        print(f"Round {server_round} evaluation metrics: {aggregated_metrics}")
        return aggregated_metrics

# Load and compile the model to get the model structure
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

# Start the server
def start_server():
    model = get_model()
    
    # Define strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=4,  # Wait until 4 clients are available
        min_evaluate_clients=4,  # Wait until 4 clients are available
        min_available_clients=4,  # Wait until 4 clients are available
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Start server
    server_address = "0.0.0.0:8080"  # Accessible from external network
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Calculate weighted average of metrics
    accuracies = [num_examples * m["sparse_categorical_accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"sparse_categorical_accuracy": sum(accuracies) / sum(examples)}

if __name__ == "__main__":
    start_server()