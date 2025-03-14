# client.py - Windows-compatible version

import flwr as fl
import tensorflow as tf
import numpy as np
import argparse
import os
import time
import sys
from tensorflow.keras.callbacks import Callback

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower client")
parser.add_argument("--client_id", type=int, choices=[1, 2, 3, 4], required=True, 
                    help="Client ID (1-4)")
parser.add_argument("--server_address", type=str, required=True, 
                    help="Server address (IP:port)")
parser.add_argument("--retries", type=int, default=5,
                    help="Number of connection retries")
args = parser.parse_args()

# Custom callback for logging training progress
class TrainingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} completed. Loss: {logs['loss']:.4f}, "
              f"Accuracy: {logs['sparse_categorical_accuracy']:.4f}")

# Load MNIST data partition for this client
def load_partition(client_id):
    print(f"Loading data partition for client {client_id}...")
    
    # Load full MNIST dataset
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    except Exception as e:
        print(f"Error loading MNIST data: {e}")
        print("Attempting to download data explicitly...")
        # Windows may have issues with the default cache location
        os.environ['KERAS_HOME'] = os.path.join(os.getcwd(), '.keras')
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
    print("Creating and compiling model...")
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
        batch_size = 32
        epochs = 1  # Just one epoch per round
        
        history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
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
    max_retries = args.retries
    
    print(f"Starting client {client_id}, connecting to {server_address}")
    
    # Load data partition for this client
    (x_train, y_train), (x_test, y_test) = load_partition(client_id)
    
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