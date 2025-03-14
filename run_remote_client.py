import argparse
import subprocess
import sys
import os

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Federated Learning client connecting to a remote server")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID (1-4)")
    parser.add_argument("--server_ip", type=str, required=True, help="IP address of the server")
    parser.add_argument("--server_port", type=int, required=True, help="Port of the server")
    args = parser.parse_args()
    
    # Construct server address
    server_address = f"{args.server_ip}:{args.server_port}"
    
    print(f"Starting client {args.client_id} connecting to server at {server_address}")
    
    # Run client
    cmd = [sys.executable, "client.py", "--client_id", str(args.client_id), "--server_address", server_address]
    
    try:
        # Run client process
        process = subprocess.Popen(cmd)
        
        # Wait for process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("Stopping client...")
        process.terminate()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 