import sys
import os
import torch
import requests
import json
import argparse
import time
from flask import Flask, render_template, jsonify, request
import threading

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Components.load_config import Path

app = Flask(__name__)

# Configuration
server_url = 'http://localhost:5000'  # Default FL server URL

# Route for dashboard
@app.route('/')
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Federated Learning Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            
            h1, h2 {
                color: #333;
            }
            
            .status-panel {
                background-color: #f0f0f0;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            
            .chart-container {
                position: relative;
                height: 400px;
                margin: 20px 0;
            }
            
            .button {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 5px;
            }
            
            .button-red {
                background-color: #f44336;
            }
            
            .grid-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            
            @media (max-width: 768px) {
                .grid-container {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Federated Learning Dashboard</h1>
            
            <div class="status-panel">
                <h2>Server Status</h2>
                <p>Status: <span id="server-status">Loading...</span></p>
                <p>Current Round: <span id="current-round">-</span> / <span id="max-rounds">-</span></p>
                <p>Clients this round: <span id="clients-count">-</span></p>
                <p>Server URL: <span id="server-url"></span></p>
                
                <button id="reset-button" class="button button-red">Reset Server</button>
                <button id="refresh-button" class="button">Refresh Status</button>
            </div>
            
            <div class="grid-container">
                <div>
                    <h2>Model Accuracy</h2>
                    <div class="chart-container">
                        <canvas id="accuracy-chart"></canvas>
                    </div>
                </div>
                <div>
                    <h2>Model Loss</h2>
                    <div class="chart-container">
                        <canvas id="loss-chart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Global chart variables
            let accuracyChart;
            let lossChart;
            let accuracyData = [];
            let lossData = [];
            let serverUrl = 'http://localhost:5000';
            
            // Initialize charts
            function initCharts() {
                // Accuracy chart
                const accuracyCtx = document.getElementById('accuracy-chart').getContext('2d');
                accuracyChart = new Chart(accuracyCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Accuracy (%)',
                            data: [],
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100
                            }
                        }
                    }
                });
                
                // Loss chart
                const lossCtx = document.getElementById('loss-chart').getContext('2d');
                lossChart = new Chart(lossCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Loss',
                            data: [],
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });
            }
            
            // Update server status
            function updateServerStatus() {
                $.ajax({
                    url: '/api/status',
                    method: 'GET',
                    success: function(data) {
                        if (data.error) {
                            $('#server-status').text('Error: ' + data.error);
                            return;
                        }
                        
                        $('#server-status').text(data.status);
                        $('#current-round').text(data.current_round);
                        $('#max-rounds').text(data.max_rounds);
                        $('#clients-count').text(data.clients_this_round);
                        $('#server-url').text(serverUrl);
                        
                        // Update training metrics
                        updateMetrics();
                    },
                    error: function() {
                        $('#server-status').text('Error connecting to server');
                    }
                });
            }
            
            // Update training metrics
            function updateMetrics() {
                $.ajax({
                    url: '/api/metrics',
                    method: 'GET',
                    success: function(data) {
                        if (data.error) {
                            console.error('Error getting metrics:', data.error);
                            return;
                        }
                        
                        if (data.training_history) {
                            // Update chart data
                            const rounds = data.training_history.rounds;
                            const accuracy = data.training_history.accuracy;
                            const loss = data.training_history.loss;
                            
                            // Update accuracy chart
                            accuracyChart.data.labels = rounds;
                            accuracyChart.data.datasets[0].data = accuracy;
                            accuracyChart.update();
                            
                            // Update loss chart
                            lossChart.data.labels = rounds;
                            lossChart.data.datasets[0].data = loss;
                            lossChart.update();
                        }
                    }
                });
            }
            
            // Reset server
            function resetServer() {
                if (confirm('Are you sure you want to reset the server? This will restart training from the beginning.')) {
                    $.ajax({
                        url: '/api/reset',
                        method: 'POST',
                        success: function(data) {
                            alert(data.message);
                            // Refresh status after reset
                            updateServerStatus();
                            // Clear charts
                            accuracyChart.data.labels = [];
                            accuracyChart.data.datasets[0].data = [];
                            lossChart.data.labels = [];
                            lossChart.data.datasets[0].data = [];
                            accuracyChart.update();
                            lossChart.update();
                        },
                        error: function() {
                            alert('Error resetting server');
                        }
                    });
                }
            }
            
            // Document ready
            $(document).ready(function() {
                initCharts();
                updateServerStatus();
                
                // Set up periodic refresh
                setInterval(updateServerStatus, 5000);
                
                // Set up button handlers
                $('#reset-button').click(resetServer);
                $('#refresh-button').click(updateServerStatus);
            });
        </script>
    </body>
    </html>
    """

# API endpoint to get server status
@app.route('/api/status')
def api_status():
    try:
        response = requests.get(f"{server_url}/status")
        return response.json()
    except Exception as e:
        return jsonify({"error": str(e)})

# API endpoint to get metrics
@app.route('/api/metrics')
def api_metrics():
    try:
        # Track training history
        global training_history
        if 'training_history' not in globals():
            training_history = {
                'rounds': [],
                'accuracy': [],
                'loss': []
            }
        
        # Get current metrics
        response = requests.get(f"{server_url}/metrics")
        data = response.json()
        
        if data['status'] == 'success':
            # Get server status to determine round
            status_response = requests.get(f"{server_url}/status")
            status = status_response.json()
            current_round = status['current_round']
            
            # Check if this round is already recorded
            if not training_history['rounds'] or training_history['rounds'][-1] != current_round:
                # Add new data point
                training_history['rounds'].append(current_round)
                training_history['accuracy'].append(data['metrics']['accuracy'])
                training_history['loss'].append(data['metrics']['loss'])
        
        return jsonify({
            "status": "success",
            "current_metrics": data['metrics'] if data['status'] == 'success' else None,
            "training_history": training_history
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# API endpoint to reset server
@app.route('/api/reset', methods=['POST'])
def api_reset():
    try:
        response = requests.post(f"{server_url}/reset")
        
        # Clear training history
        global training_history
        training_history = {
            'rounds': [],
            'accuracy': [],
            'loss': []
        }
        
        return jsonify({"message": "Server reset successfully"})
    except Exception as e:
        return jsonify({"error": str(e)})

# API endpoint to set server URL
@app.route('/api/set_url', methods=['POST'])
def api_set_url():
    global server_url
    data = request.json
    new_url = data.get('url')
    
    if new_url:
        server_url = new_url
        return jsonify({"status": "success", "message": f"Server URL set to {server_url}"})
    else:
        return jsonify({"status": "error", "message": "No URL provided"})

def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning Dashboard')
    parser.add_argument('--port', type=int, default=8080,
                        help='Dashboard port')
    parser.add_argument('--server_url', type=str, default='http://localhost:5000',
                        help='FL Server URL')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    server_url = args.server_url
    
    # Initialize training history
    training_history = {
        'rounds': [],
        'accuracy': [],
        'loss': []
    }
    
    # Start Flask app
    app.run(host='0.0.0.0', port=args.port, debug=True)