import tensorflow as tf
import numpy as np
import os
from config import MODEL_CONFIG, DATA_CONFIG

def create_model():
    """
    Tạo và biên dịch model CNN cho nhận dạng chữ số MNIST
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(MODEL_CONFIG["conv1_filters"], kernel_size=(3, 3), 
                              activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(MODEL_CONFIG["conv2_filters"], kernel_size=(3, 3), 
                              activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(MODEL_CONFIG["dense_units"], activation="relu"),
        tf.keras.layers.Dropout(MODEL_CONFIG["dropout_rate"]),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

def load_and_preprocess_data():
    """
    Tải và tiền xử lý dataset MNIST
    """
    try:
        # Tải dataset MNIST
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    except Exception as e:
        print(f"Error loading MNIST data: {e}")
        print("Attempting to download data explicitly...")
        # Windows may have issues with the default cache location
        os.environ['KERAS_HOME'] = os.path.join(os.getcwd(), DATA_CONFIG["keras_home"])
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Chuẩn hóa dữ liệu
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Reshape dữ liệu
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    return (x_train, y_train), (x_test, y_test)

def get_client_data(client_id):
    """
    Chia dữ liệu cho từng client
    
    Args:
        client_id: ID của client (1-4)
        
    Returns:
        Tuple chứa dữ liệu training và testing cho client
    """
    # Tải và tiền xử lý dữ liệu
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Chia dữ liệu cho client
    n_clients = DATA_CONFIG["num_clients"]
    samples_per_client = len(x_train) // n_clients
    
    start_idx = (client_id - 1) * samples_per_client
    end_idx = client_id * samples_per_client if client_id < n_clients else len(x_train)
    
    client_x = x_train[start_idx:end_idx]
    client_y = y_train[start_idx:end_idx]
    
    print(f"Client {client_id} loaded {len(client_x)} training samples")
    return (client_x, client_y), (x_test, y_test)

# Custom callback để theo dõi quá trình training
class TrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} completed. Loss: {logs['loss']:.4f}, "
              f"Accuracy: {logs['sparse_categorical_accuracy']:.4f}") 