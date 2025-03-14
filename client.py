# client.py - Run this on each of your 4 computers with their respective client_id

import flwr as fl
import tensorflow as tf
import numpy as np
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower client")
parser.add_argument("--client_id", type=int, choices=[1, 2, 3, 4], required=True, 
                    help="Client ID (1-4)")
parser.add_argument("--server_address", type=str, default="192.168.1.100:8080", 
                    help="Server address (IP:port)")
args = parser.parse_args()

# Load MNIST data
def load_partition(client_id):
    # Load full MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Reshape
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Determine indices for this client
    n_clients = 4
    samples_per_client = len(x_train) // n_clients
    
    start_idx = (client_id - 1) * samples_per_client
    end_idx = client_id * samples_per_client if client_id < n_clients else len(x_train)
    
    client_x = x_train[start_idx:end_idx]
    client_y = y_train[start_idx:end_idx]
    
    print(f"Client {client_id} loaded {len(client_x)} training samples")
    return (client_x, client_y), (x_test, y_test)

# Define model
def create_model():
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

# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, client_id):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.client_id = client_id
        
        # Create client directory for saving models
        self.save_dir = f"client_{client_id}_models"
        os.makedirs(self.save_dir, exist_ok=True)

    def get_parameters(self, config):
        return [np.array(param) for param in self.model.get_weights()]

    def fit(self, parameters, config):
        # Update local model with global parameters
        self.model.set_weights([np.array(param) for param in parameters])
        
        # Get current round from config
        round_num = config.get("round_num", 0)
        
        # Train the model
        batch_size = 32
        epochs = 1  # Just one epoch per round
        
        history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )
        
        # Save local model
        self.model.save(f"{self.save_dir}/model_round_{round_num}.h5")
        
        # Return updated model parameters, train size, and metrics
        return (
            [np.array(param) for param in self.model.get_weights()],
            len(self.x_train),
            {"loss": history.history["loss"][-1], 
             "accuracy": history.history["sparse_categorical_accuracy"][-1]}
        )

    def evaluate(self, parameters, config):
        # Update local model with global parameters
        self.model.set_weights([np.array(param) for param in parameters])
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        return loss, len(self.x_test), {"sparse_categorical_accuracy": accuracy}

# Main execution
def main():
    client_id = args.client_id
    server_address = args.server_address
    
    # Load data partition for this client
    (x_train, y_train), (x_test, y_test) = load_partition(client_id)
    
    # Create and compile model
    model = create_model()
    
    # Start Flower client
    client = MnistClient(model, x_train, y_train, x_test, y_test, client_id)
    
    print(f"Client {client_id} connecting to server at {server_address}")
    fl.client.start_numpy_client(server_address=server_address, client=client)

if __name__ == "__main__":
    main()