import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from model import create_model, load_and_preprocess_data
from config import SERVER_CONFIG

def load_global_model(round_num):
    """
    Tải global model từ file .npz
    """
    model_path = f"{SERVER_CONFIG['server_data_dir']}/global_model_round_{round_num}.npz"
    
    if not os.path.exists(model_path):
        print(f"Không tìm thấy model cho round {round_num}")
        return None
    
    # Tạo model với cấu trúc giống như trong training
    model = create_model()
    
    # Tải weights từ file .npz
    with np.load(model_path) as data:
        weights = [data[f'arr_{i}'] for i in range(len(data.files))]
    
    # Gán weights cho model
    model.set_weights(weights)
    print(f"Đã tải model từ round {round_num}")
    
    return model

def evaluate_model(model, x_test, y_test):
    """
    Đánh giá model trên tập test
    """
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy

def visualize_predictions(model, x_test, y_test, num_samples=10):
    """
    Hiển thị một số dự đoán của model
    """
    # Lấy ngẫu nhiên một số mẫu
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    samples = x_test[indices]
    labels = y_test[indices]
    
    # Dự đoán
    predictions = model.predict(samples)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Hiển thị
    plt.figure(figsize=(15, 8))
    for i in range(num_samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(samples[i].reshape(28, 28), cmap='gray')
        
        if predicted_labels[i] == labels[i]:
            color = 'green'
        else:
            color = 'red'
            
        plt.title(f"Pred: {predicted_labels[i]}, True: {labels[i]}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("predictions.png")
    print(f"Đã lưu hình ảnh dự đoán vào predictions.png")

def main():
    # Tải dữ liệu test
    _, (x_test, y_test) = load_and_preprocess_data()
    
    # Tìm round cuối cùng
    max_round = 0
    for file in os.listdir(SERVER_CONFIG['server_data_dir']):
        if file.startswith("global_model_round_") and file.endswith(".npz"):
            round_num = int(file.split("_")[-1].split(".")[0])
            max_round = max(max_round, round_num)
    
    if max_round == 0:
        print("Không tìm thấy model đã train")
        return
    
    print(f"Round cuối cùng: {max_round}")
    
    # Tải model
    model = load_global_model(max_round)
    if model is None:
        return
    
    # Đánh giá model
    evaluate_model(model, x_test, y_test)
    
    # Hiển thị một số dự đoán
    visualize_predictions(model, x_test, y_test)

if __name__ == "__main__":
    main() 