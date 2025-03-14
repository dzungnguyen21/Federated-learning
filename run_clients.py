import subprocess
import sys
import time
import os
from config import NUM_CLIENTS

def run_client(client_id, server_address=None):
    """
    Chạy một client với ID cụ thể
    """
    cmd = [sys.executable, "client.py", "--client_id", str(client_id)]
    
    if server_address:
        cmd.extend(["--server_address", server_address])
    
    # Mở một terminal mới và chạy client
    if os.name == 'nt':  # Windows
        return subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:  # Linux/Mac
        terminal_cmd = ["gnome-terminal", "--"] + cmd  # Cho Ubuntu/Debian
        # Nếu dùng Mac, hãy thay bằng: terminal_cmd = ["osascript", "-e", f'tell app "Terminal" to do script "{" ".join(cmd)}"']
        return subprocess.Popen(terminal_cmd)

def main():
    # Kiểm tra xem server_port.txt có tồn tại không
    try:
        with open("server_port.txt", "r") as f:
            port = f.read().strip()
            server_address = f"localhost:{port}"
            print(f"Đã tìm thấy server đang chạy trên {server_address}")
    except:
        server_address = None
        print("Không tìm thấy server_port.txt. Clients sẽ tự động tìm server.")
    
    # Số lượng clients muốn chạy (lấy từ config)
    num_clients = NUM_CLIENTS
    
    # Chạy các clients
    processes = []
    for i in range(1, num_clients + 1):
        print(f"Khởi động client {i}...")
        proc = run_client(i, server_address)
        processes.append(proc)
        time.sleep(1)  # Đợi 1 giây giữa các lần khởi động
    
    print(f"Đã khởi động {num_clients} clients.")
    print("Nhấn Ctrl+C để kết thúc tất cả các clients.")
    
    try:
        # Đợi cho đến khi người dùng nhấn Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Đang kết thúc các clients...")
        for proc in processes:
            proc.terminate()
        print("Đã kết thúc tất cả các clients.")

if __name__ == "__main__":
    main() 