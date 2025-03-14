# Federated Learning với MNIST Dataset

Dự án này triển khai một hệ thống Federated Learning sử dụng Flower framework để huấn luyện mô hình CNN trên dataset MNIST. Hệ thống bao gồm một server trung tâm và nhiều clients phân tán.

## Giới thiệu

Federated Learning (FL) là một phương pháp học máy cho phép huấn luyện mô hình trên dữ liệu phân tán mà không cần chia sẻ dữ liệu gốc. Trong dự án này:

- Server điều phối quá trình huấn luyện và tổng hợp các mô hình cục bộ
- Mỗi client huấn luyện mô hình trên dữ liệu riêng của mình
- Chỉ các tham số mô hình được chia sẻ, không phải dữ liệu gốc
- Sử dụng thuật toán FedAvg (Federated Averaging) để tổng hợp mô hình

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

## Cấu trúc dự án

- `server.py`: Server trung tâm quản lý quá trình huấn luyện
- `client.py`: Client thực hiện huấn luyện cục bộ
- `model.py`: Định nghĩa mô hình và các hàm xử lý dữ liệu
- `config.py`: Cấu hình cho server, client, model và dữ liệu
- `run_system.py`: Script để chạy toàn bộ hệ thống (server và clients)
- `run_clients.py`: Script để chạy nhiều clients
- `run_remote_client.py`: Script để chạy client kết nối với server từ xa
- `evaluate_model.py`: Script để đánh giá mô hình sau khi huấn luyện
- `requirements.txt`: Danh sách các thư viện cần thiết

## Cách sử dụng

### Chạy toàn bộ hệ thống trên một máy tính

Cách đơn giản nhất để chạy hệ thống là sử dụng script `run_system.py`:

```bash
python run_system.py
```

Script này sẽ:
- Khởi động server trong một terminal riêng
- Đợi 5 giây để server khởi động
- Khởi động 2 clients trong các terminal riêng
- Tự động kết nối các clients với server
- Bắt đầu quá trình training

### Chạy từng phần riêng biệt

Nếu bạn muốn kiểm soát chi tiết hơn, bạn có thể chạy từng phần riêng biệt:

1. Khởi động server:
```bash
python server.py
```

2. Khởi động clients (trong các terminal riêng biệt):
```bash
python client.py --client_id 1
python client.py --client_id 2
```

Hoặc sử dụng script để chạy nhiều clients cùng lúc:
```bash
python run_clients.py
```

### Chạy hệ thống trên nhiều máy tính

Bạn có thể chạy server và clients trên các máy tính khác nhau trong cùng một mạng LAN:

#### Máy tính 1 (Server):

1. Tìm địa chỉ IP của máy tính:
```bash
ipconfig  # Windows
ifconfig  # Linux/Mac
```

2. Khởi động server:
```bash
python server.py
```

3. Ghi lại port mà server đang sử dụng (hiển thị trong terminal)

#### Máy tính 2 (Client):

1. Sao chép toàn bộ code sang máy tính này
2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Chạy client với địa chỉ IP và port của server:
```bash
python client.py --client_id 1 --server_address <IP_của_server>:<port>
```

Hoặc sử dụng script `run_remote_client.py` để dễ dàng hơn:
```bash
python run_remote_client.py --client_id 1 --server_ip <IP_của_server> --server_port <port>
```

Ví dụ:
```bash
python run_remote_client.py --client_id 1 --server_ip 192.168.1.100 --server_port 8080
```

#### Lưu ý quan trọng:

- Đảm bảo tất cả các máy tính đều trong cùng một mạng LAN
- Tắt firewall hoặc mở port 8080-8090 trên máy tính chạy server
- Mỗi client phải có một ID duy nhất (1, 2, 3, 4)
- Sau khi training, bạn có thể sao chép thư mục `server_data` từ máy server về máy client để đánh giá model

### Đánh giá mô hình

Sau khi training hoàn tất, bạn có thể đánh giá mô hình:
```bash
python evaluate_model.py
```

Script này sẽ:
- Tải global model từ round cuối cùng
- Đánh giá model trên tập test
- Hiển thị một số dự đoán và lưu vào file `predictions.png`

## Tùy chỉnh cấu hình

Bạn có thể dễ dàng tùy chỉnh các tham số của hệ thống bằng cách chỉnh sửa file `config.py`:

- **SERVER_CONFIG**: Cấu hình cho server (port, số rounds, số clients tối thiểu, ...)
- **CLIENT_CONFIG**: Cấu hình cho client (batch size, epochs, validation split, ...)
- **MODEL_CONFIG**: Cấu hình cho model (learning rate, số filters, dropout rate, ...)
- **DATA_CONFIG**: Cấu hình cho dữ liệu (số clients, thư mục cache, ...)

## Xử lý sự cố

### Lỗi kết nối

Nếu bạn gặp lỗi kết nối, hãy kiểm tra:
- Firewall có đang chặn kết nối không
- Port 8080-8090 có đang được sử dụng bởi ứng dụng khác không
- Sử dụng lệnh `netstat -ano | findstr :8080` để kiểm tra port
- Sử dụng lệnh `taskkill /PID <PID> /F` để kill process đang sử dụng port

### Lỗi kết nối giữa các máy tính

- Kiểm tra xem các máy tính có thể ping được nhau không
- Đảm bảo địa chỉ IP và port chính xác
- Tắt tạm thời firewall để test kết nối
- Đảm bảo server đang lắng nghe trên tất cả các interfaces (0.0.0.0)

### Lỗi khác

- Nếu bạn gặp lỗi khi tải dataset MNIST, hãy kiểm tra kết nối internet
- Nếu bạn gặp lỗi khi lưu model, hãy đảm bảo thư mục `server_data` đã được tạo
- Nếu clients không thể kết nối với server, hãy kiểm tra file `server_port.txt`

## Thông số kỹ thuật

- Framework: Flower 1.5.0
- Deep Learning Framework: TensorFlow 2.15.0
- Dataset: MNIST
- Model: CNN
- Thuật toán: Federated Averaging (FedAvg)
- Số rounds mặc định: 5
- Số clients tối thiểu: 2

## Giấy phép

MIT License 