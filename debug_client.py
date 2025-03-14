import argparse
import subprocess
import sys
import os
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("debug_client")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Federated Learning client with debug logs")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID (1-4)")
    parser.add_argument("--server_ip", type=str, required=True, help="IP address of the server")
    parser.add_argument("--server_port", type=int, required=True, help="Port of the server")
    args = parser.parse_args()
    
    # Construct server address
    server_address = f"{args.server_ip}:{args.server_port}"
    
    logger.info(f"Starting client {args.client_id} connecting to server at {server_address}")
    
    # Thiết lập biến môi trường để hiển thị logs chi tiết hơn
    env = os.environ.copy()
    env["GRPC_VERBOSITY"] = "DEBUG"
    env["GRPC_TRACE"] = "all"
    
    # Run client
    cmd = [sys.executable, "client.py", "--client_id", str(args.client_id), "--server_address", server_address]
    
    try:
        # Run client process với logs chi tiết
        logger.info(f"Executing command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Hiển thị output theo thời gian thực
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        # Đợi process kết thúc
        process.wait()
        logger.info(f"Client process exited with code {process.returncode}")
        
    except KeyboardInterrupt:
        logger.info("Stopping client...")
        process.terminate()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 