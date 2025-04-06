import sys
import os
import torch
import numpy as np
import json
from flask import Flask, request, jsonify
from io import BytesIO
import base64

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.data_loader import load_dataset
from Server.global_model import Server
from Components.model import get_model, get_model_parameters, set_model_parameters
from Components.load_config import Path
from torch.utils.data import DataLoader

app = Flask(__name__)

# Global variables
global_server = None
global_round = 0
max_rounds = 0
clients_this_round = {}
round_completion = False

def init_server():
    """
    Initialize the federated learning server
    """
    global global_server, max_rounds
    
    # Load configuration
    config_loader = Path()
    config = config_loader.config
    max_rounds = config['training']['global_rounds']
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load dataset
    _, test_dataset = load_dataset()
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Initialize server
    global_server = Server(test_loader)
    
    print(f"Server initialized with {config['data']['dataset']} dataset and {config['model']['name']} model")
    print(f"Server will run for {max_rounds} rounds")

def tensor_to_base64(tensor):
    """
    Convert a PyTorch tensor to base64 encoded string
    """
    buffer = BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def base64_to_tensor(b64_string):
    """
    Convert a base64 encoded string back to PyTorch tensor
    """
    buffer = BytesIO(base64.b64decode(b64_string))
    return torch.load(buffer, map_location=torch.device('cpu'))

@app.route('/')
def index():
    """
    Root endpoint for the server
    """
    return "Federated Learning Server - Running"

@app.route('/status', methods=['GET'])
def status():
    """
    Get current server status
    """
    global global_round, max_rounds, round_completion, clients_this_round
    
    if global_server is None:
        return jsonify({
            'status': 'Not initialized',
            'message': 'Server not initialized yet'
        })
    
    # Get configuration for client count
    config_loader = Path()
    config = config_loader.config
    clients_per_round = max(1, int(config['server']['fraction_clients'] * config['data']['num_clients']))
    
    return jsonify({
        'status': 'Running',
        'current_round': global_round,
        'max_rounds': max_rounds,
        'round_completed': round_completion,
        'clients_this_round': len(clients_this_round),
        'clients_needed': clients_per_round
    })

@app.route('/get_model', methods=['GET'])
def get_model_params():
    """
    Provide the current global model to clients
    """
    global global_server, global_round, max_rounds
    
    if global_server is None:
        return jsonify({
            'status': 'error',
            'message': 'Server not initialized',
            'round': -1
        }), 500
    
    if global_round >= max_rounds:
        return jsonify({
            'status': 'completed',
            'message': 'Training completed',
            'round': global_round
        })
    
    # Get model parameters
    parameters = global_server.get_global_parameters()
    
    # Convert tensors to base64 strings
    param_data = [tensor_to_base64(param) for param in parameters]
    
    return jsonify({
        'status': 'success',
        'round': global_round,
        'model_params': param_data
    })

@app.route('/submit_update', methods=['POST'])
def submit_update():
    """
    Submit updated model parameters from client
    """
    global global_server, global_round, clients_this_round, round_completion
    
    if global_server is None:
        return jsonify({
            'status': 'error',
            'message': 'Server not initialized',
            'round': -1
        }), 500
    
    if global_round >= max_rounds:
        return jsonify({
            'status': 'completed',
            'message': 'Training completed',
            'round': global_round
        })
    
    # Get data from request
    data = request.json
    client_id = data.get('client_id')
    client_round = data.get('round')
    
    # Verify round number
    if client_round != global_round:
        return jsonify({
            'status': 'error',
            'message': f'Client round {client_round} does not match server round {global_round}',
            'round': global_round
        }), 400
    
    # Convert base64 strings back to tensors
    try:
        model_params = [base64_to_tensor(param) for param in data.get('model_params')]
        
        # Store client update
        clients_this_round[client_id] = model_params
        
        # Check server configuration for number of clients per round
        config_loader = Path()
        config = config_loader.config
        clients_per_round = max(1, int(config['server']['fraction_clients'] * config['data']['num_clients']))
        
        # If we have enough clients, aggregate and move to the next round
        if len(clients_this_round) >= clients_per_round:
            # Aggregate parameters
            global_server.aggregate_parameters(list(clients_this_round.values()))
            
            # Evaluate global model
            test_loss, accuracy = global_server.evaluate()
            print(f"Round {global_round + 1}: Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            # Move to the next round
            global_round += 1
            clients_this_round.clear()
            round_completion = True
            
            return jsonify({
                'status': 'success',
                'message': 'Update received and round completed',
                'round_completed': True,
                'new_round': global_round,
                'metrics': {
                    'loss': float(test_loss),
                    'accuracy': float(accuracy)
                }
            })
        else:
            return jsonify({
                'status': 'success',
                'message': 'Update received',
                'round_completed': False,
                'clients_received': len(clients_this_round),
                'clients_needed': clients_per_round,
                'round': global_round
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing update: {str(e)}',
            'round': global_round
        }), 500

@app.route('/reset', methods=['POST'])
def reset():
    """
    Reset the server to start training from the beginning
    """
    global global_server, global_round, clients_this_round, round_completion
    
    try:
        init_server()
        global_round = 0
        clients_this_round = {}
        round_completion = False
        
        return jsonify({
            'status': 'success',
            'message': 'Server reset successfully',
            'round': global_round
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error resetting server: {str(e)}'
        }), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get current model metrics
    """
    global global_server, global_round
    
    if global_server is None:
        return jsonify({
            'status': 'error',
            'message': 'Server not initialized'
        }), 500
    
    try:
        # Evaluate global model
        test_loss, accuracy = global_server.evaluate()
        
        return jsonify({
            'status': 'success',
            'round': global_round,
            'metrics': {
                'loss': float(test_loss),
                'accuracy': float(accuracy)
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error getting metrics: {str(e)}'
        }), 500

if __name__ == "__main__":
    # Initialize the server
    init_server()
    
    # Start Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)