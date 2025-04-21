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
            
            h1, h2, h3 {
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
            
            .progress-container {
                width: 100%;
                background-color: #ddd;
                border-radius: 5px;
                margin: 10px 0;
            }
            
            .progress-bar {
                height: 20px;
                border-radius: 5px;
                background-color: #4CAF50;
                text-align: center;
                color: white;
                line-height: 20px;
            }
            
            .metrics-panel {
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
            }
            
            .confusion-matrix {
                margin-top: 15px;
                overflow-x: auto;
            }
            
            .confusion-matrix table {
                border-collapse: collapse;
                width: 100%;
                max-width: 600px;
                margin: 0 auto;
            }
            
            .confusion-matrix th, .confusion-matrix td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            
            .confusion-matrix th {
                background-color: #4CAF50;
                color: white;
            }
            
            .confusion-matrix td.diagonal {
                background-color: rgba(75, 192, 192, 0.2);
                font-weight: bold;
            }
            
            .metrics-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }
            
            .metrics-table th, .metrics-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            
            .metrics-table th {
                background-color: #4CAF50;
                color: white;
            }
            
            .final-accuracy {
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                margin: 15px 0;
                color: #4CAF50;
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
                <p>Waiting for clients: <span id="clients-count">-</span> / <span id="total-clients">-</span></p>
                <div class="progress-container">
                    <div id="client-progress" class="progress-bar" style="width:0%">0%</div>
                </div>
                <p>Server URL: <span id="server-url"></span></p>
                
                <button id="reset-button" class="button button-red">Reset Server</button>
                <button id="refresh-button" class="button">Refresh Status</button>
                <button id="show-metrics-button" class="button">Show Detailed Metrics</button>
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
            
            <div class="metrics-panel" id="detailed-metrics" style="display: none;">
                <h2>Detailed Model Metrics</h2>
                
                <div class="final-accuracy">
                    Final Test Accuracy: <span id="final-accuracy">-</span>%
                </div>
                
                <h3>Confusion Matrix</h3>
                <div class="confusion-matrix" id="confusion-matrix">
                    <p>Loading confusion matrix...</p>
                </div>
                
                <h3>Classification Report</h3>
                <div id="classification-report">
                    <table class="metrics-table" id="metrics-table">
                        <thead>
                            <tr>
                                <th>Class</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1-score</th>
                                <th>Support</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td colspan="5">Loading metrics...</td>
                            </tr>
                        </tbody>
                    </table>
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
            let detailedMetricsVisible = false;
            
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
                        $('#total-clients').text(data.clients_needed);
                        $('#server-url').text(serverUrl);
                        
                        // Update progress bar
                        if (data.clients_needed > 0) {
                            const percentage = Math.round((data.clients_this_round / data.clients_needed) * 100);
                            $('#client-progress').css('width', percentage + '%');
                            $('#client-progress').text(percentage + '%');
                        }
                        
                        // Update training metrics
                        updateMetrics();
                        
                        // Check if training is complete to show final metrics
                        if (data.current_round >= data.max_rounds) {
                            updateDetailedMetrics();
                        }
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
            
            // Update detailed metrics
            function updateDetailedMetrics() {
                $.ajax({
                    url: '/api/detailed_metrics',
                    method: 'GET',
                    success: function(data) {
                        if (data.error || data.status === 'error') {
                            console.error('Error getting detailed metrics:', data.error || data.message);
                            return;
                        }
                        
                        const metrics = data.detailed_metrics;
                        
                        // Update final accuracy
                        $('#final-accuracy').text(metrics.accuracy.toFixed(2));
                        
                        // Generate confusion matrix
                        generateConfusionMatrix(metrics.confusion_matrix);
                        
                        // Generate classification report
                        generateClassificationReport(metrics);
                    },
                    error: function(xhr, status, error) {
                        console.error('AJAX error:', status, error);
                    }
                });
            }
            
            // Generate confusion matrix table
            function generateConfusionMatrix(confusionMatrix) {
                if (!confusionMatrix) {
                    $('#confusion-matrix').html('<p>Confusion matrix not available</p>');
                    return;
                }
                
                const numClasses = confusionMatrix.length;
                
                // Create the table HTML
                let tableHTML = '<table>';
                
                // Header row with predicted labels
                tableHTML += '<tr><th></th><th colspan="' + numClasses + '">Predicted</th></tr>';
                tableHTML += '<tr><th>Actual</th>';
                for (let i = 0; i < numClasses; i++) {
                    tableHTML += '<th>' + i + '</th>';
                }
                tableHTML += '</tr>';
                
                // Data rows
                for (let i = 0; i < numClasses; i++) {
                    tableHTML += '<tr>';
                    tableHTML += '<th>' + i + '</th>';
                    
                    for (let j = 0; j < numClasses; j++) {
                        // Highlight the diagonal elements (correct predictions)
                        const className = (i === j) ? 'diagonal' : '';
                        tableHTML += '<td class="' + className + '">' + confusionMatrix[i][j] + '</td>';
                    }
                    
                    tableHTML += '</tr>';
                }
                
                tableHTML += '</table>';
                
                $('#confusion-matrix').html(tableHTML);
            }
            
            // Generate classification report table
            function generateClassificationReport(metrics) {
                if (!metrics.precision || !metrics.recall || !metrics.f1 || !metrics.support) {
                    $('#classification-report').html('<p>Classification metrics not available</p>');
                    return;
                }
                
                const numClasses = metrics.precision.length;
                
                // Create table body
                let tableBody = '';
                
                for (let i = 0; i < numClasses; i++) {
                    tableBody += '<tr>';
                    tableBody += '<td>' + i + '</td>';
                    tableBody += '<td>' + metrics.precision[i].toFixed(4) + '</td>';
                    tableBody += '<td>' + metrics.recall[i].toFixed(4) + '</td>';
                    tableBody += '<td>' + metrics.f1[i].toFixed(4) + '</td>';
                    tableBody += '<td>' + metrics.support[i] + '</td>';
                    tableBody += '</tr>';
                }
                
                // Insert the table body into the table
                $('#metrics-table tbody').html(tableBody);
            }
            
            // Toggle detailed metrics visibility
            function toggleDetailedMetrics() {
                detailedMetricsVisible = !detailedMetricsVisible;
                
                if (detailedMetricsVisible) {
                    $('#detailed-metrics').show();
                    $('#show-metrics-button').text('Hide Detailed Metrics');
                    updateDetailedMetrics();
                } else {
                    $('#detailed-metrics').hide();
                    $('#show-metrics-button').text('Show Detailed Metrics');
                }
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
                $('#show-metrics-button').click(toggleDetailedMetrics);
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
        data = response.json()
        return jsonify({
            "status": data.get('status', 'Unknown'),
            "current_round": data.get('current_round', 0),
            "max_rounds": data.get('max_rounds', 0),
            "clients_this_round": data.get('clients_this_round', 0),
            "clients_needed": data.get('clients_needed', 0)
        })
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

# API endpoint to get detailed metrics
@app.route('/api/detailed_metrics')
def api_detailed_metrics():
    try:
        response = requests.get(f"{server_url}/detailed_metrics")
        data = response.json()
        
        if data['status'] == 'success':
            return jsonify({
                "status": "success",
                "detailed_metrics": data['detailed_metrics']
            })
        else:
            return jsonify({"status": "error", "message": data.get('message', 'Unknown error')})
    except Exception as e:
        return jsonify({"error": str(e)})

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