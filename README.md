# Federated Learning với MNIST Dataset

Dự án này triển khai một hệ thống Federated Learning sử dụng Flower framework để huấn luyện mô hình CNN trên dataset MNIST. Hệ thống bao gồm một server trung tâm và bốn clients phân tán.

## Cài đặt

1. Clone repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Tạo môi trường ảo (khuyến nghị):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Cách sử dụng

1. Khởi động server:
```bash
python server.py
```

2. Khởi động các clients (trong các terminal riêng biệt):
```bash
python client.py --client_id 1 --server_address localhost:8080
python client.py --client_id 2 --server_address localhost:8080
python client.py --client_id 3 --server_address localhost:8080
python client.py --client_id 4 --server_address localhost:8080
```

## Cấu trúc dự án

- `server.py`: Server trung tâm quản lý quá trình huấn luyện
- `client.py`: Client thực hiện huấn luyện cục bộ
- `requirements.txt`: Danh sách các thư viện cần thiết
- `server_data/`: Thư mục lưu model của server
- `client_X_models/`: Thư mục lưu model của từng client

## Thông số kỹ thuật

- Framework: Flower
- Deep Learning Framework: TensorFlow
- Dataset: MNIST
- Model: CNN
- Số lượng clients: 4
- Số rounds training: 10
- Batch size: 32
- Learning rate: 0.001

## License

MIT License 