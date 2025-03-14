# Cấu hình cho Federated Learning

# Số lượng clients trong hệ thống
NUM_CLIENTS = 2  # Tổng số clients trong hệ thống

# Cấu hình server
SERVER_CONFIG = {
    "min_port": 8080,                # Port bắt đầu để tìm port trống
    "max_port": 8090,                # Port kết thúc để tìm port trống
    "host": "0.0.0.0",               # Host address (0.0.0.0 để lắng nghe tất cả interfaces)
    "num_rounds": 5,                 # Số rounds training (giảm xuống để test nhanh hơn)
    "min_clients": 1,                # Số lượng clients tối thiểu cần thiết (giảm xuống để dễ test)
    "fraction_fit": 1.0,             # Tỉ lệ clients tham gia training
    "fraction_evaluate": 1.0,        # Tỉ lệ clients tham gia evaluation
    "server_data_dir": "server_data" # Thư mục lưu dữ liệu server
}

# Cấu hình client
CLIENT_CONFIG = {
    "batch_size": 32,                # Batch size cho training
    "epochs_per_round": 1,           # Số epochs mỗi round
    "validation_split": 0.1,         # Tỉ lệ dữ liệu validation
    "max_retries": 5,                # Số lần thử kết nối tối đa
    "client_data_dir_prefix": "client_",  # Prefix cho thư mục lưu dữ liệu client
    "client_data_dir_suffix": "_models"   # Suffix cho thư mục lưu dữ liệu client
}

# Cấu hình model
MODEL_CONFIG = {
    "learning_rate": 0.001,          # Learning rate
    "conv1_filters": 32,             # Số filters cho lớp Conv2D đầu tiên
    "conv2_filters": 64,             # Số filters cho lớp Conv2D thứ hai
    "dense_units": 128,              # Số units cho lớp Dense
    "dropout_rate": 0.2              # Tỉ lệ dropout
}

# Cấu hình dữ liệu
DATA_CONFIG = {
    "num_clients": NUM_CLIENTS,      # Tổng số clients
    "keras_home": ".keras"           # Thư mục cache cho Keras
} 