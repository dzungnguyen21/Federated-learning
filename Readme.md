# Federated Learning Framework

A comprehensive federated learning implementation that supports MNIST and CIFAR-10 datasets with various FL algorithms, including FedAvg and FedProx.

## Overview

This project implements a flexible federated learning framework that allows both local and remote training. It features:

- Support for MNIST and CIFAR-10 datasets
- IID and non-IID data distributions
- Multiple federated learning algorithms (FedAvg, FedProx)
- Local and distributed training capabilities
- Web-based dashboard for monitoring training progress

## Project Structure

```
FL/
├── SetupFL/
│   ├── Client/             # Client-side implementation
│   ├── Server/             # Server-side implementation
│   ├── Components/         # Shared components
│   ├── Data/               # Data loading and processing
│   ├── Remote/             # Remote training modules
│   ├── Local/              # Local training modules
│   └── config.yaml         # Configuration file
├── requirement.txt         # Project dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install the required packages:
   ```
   pip install -r requirement.txt
   ```

## Usage

### Configuration

Edit `config.yaml` to customize:
- Dataset (MNIST, CIFAR-10)
- Model architecture
- Training parameters
- Data distribution (IID or non-IID)
- Federated learning algorithm

### Local Training

Run local federated learning simulation:

```
python SetupFL/Local/local_test.py --visualize
```

### Remote Training

1. Start the server:
   ```
   python SetupFL/Remote/remote_test.py
   ```

2. Start multiple clients (in separate terminals):
   ```
   python SetupFL/Remote/remote_client.py --client_id 0 --server_url http://localhost:5000
   python SetupFL/Remote/remote_client.py --client_id 1 --server_url http://localhost:5000
   ```

3. Launch the dashboard:
   ```
   python SetupFL/Remote/dashboard.py --server_url http://localhost:5000
   ```
   Access the dashboard at http://localhost:8080

#### Using Ngrok for Remote Connections

To make your federated learning server accessible over the internet (for truly distributed training):

1. Install ngrok from [https://ngrok.com/download](https://ngrok.com/download)

2. Add ngrok to your system PATH or navigate to the directory containing the ngrok executable

3. Start your federated learning server:
   ```
   python SetupFL/Remote/remote_test.py
   ```

4. In a separate terminal, start ngrok:
   ```
   ngrok http 5000
   ```

5. Ngrok will display a forwarding URL (e.g., `https://abc123.ngrok-free.app`). Use this URL for your remote clients:
   ```
   python remote_client.py --client_id 0 --server_url "https://5af4-42-113-61-75.ngrok-free.app"
   ```

6. For the dashboard, also use the ngrok URL:
   ```
   python SetupFL/Remote/dashboard.py --server_url "https://5af4-42-113-61-75.ngrok-free.app"
   ```

Note: The ngrok URL will change each time you restart ngrok unless you have a paid account with a fixed subdomain.

## Data Visualization

Visualize dataset distribution:

```
python SetupFL/Data/visualize_data.py
```

## Models

The framework includes:
- CNN for MNIST
- CNN for CIFAR-10

You can modify or extend models in `Components/model.py`.

## Algorithms

Supported federated learning algorithms:
- **FedAvg**: Standard federated averaging
- **FedProx**: Federated Proximal focusing on heterogeneity with regularization

## Dashboard

The web dashboard provides real-time monitoring of:
- Training progress
- Model accuracy
- Loss curves
- Server status

## License

[MIT License](https://opensource.org/licenses/MIT)

## Troubleshooting

### Large File Issues with Git

If you encounter GitHub's file size limit errors with data files:

1. Install Git LFS:
   ```
   git lfs install
   ```

2. Track large files:
   ```
   git lfs track "*.tar.gz"
   ```

3. Add and commit:
   ```
   git add .gitattributes
   git add your-large-file.tar.gz
   git commit -m "Add large file using Git LFS"
   ```

### Environment Issues

If you encounter Python environment errors:
1. Ensure you're using the correct virtual environment
2. Try recreating the virtual environment
3. Check the Python version (3.7+ recommended)

## Acknowledgements

This project is based on research in federated learning, particularly drawing from:
- "Communication-Efficient Learning of Deep Networks from Decentralized Data" (McMahan et al., 2017)
- "Federated Optimization in Heterogeneous Networks" (Li et al., 2020)