import subprocess
import sys
import time
import os
import signal

def run_server():
    """
    Chạy server trong một terminal mới
    """
    cmd = [sys.executable, "server.py"]
    
    # Mở một terminal mới và chạy server
    if os.name == 'nt':  # Windows
        return subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:  # Linux/Mac
        terminal_cmd = ["gnome-terminal", "--"] + cmd  # Cho Ubuntu/Debian
        # Nếu dùng Mac, hãy thay bằng: terminal_cmd = ["osascript", "-e", f'tell app "Terminal" to do script "{" ".join(cmd)}"']
        return subprocess.Popen(terminal_cmd)

def run_client(client_id):
    """
    Chạy một client với ID cụ thể
    """
    cmd = [sys.executable, "client.py", "--client_id", str(client_id)]
    
    # Mở một terminal mới và chạy client
    if os.name == 'nt':  # Windows
        return subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:  # Linux/Mac
        terminal_cmd = ["gnome-terminal", "--"] + cmd  # Cho Ubuntu/Debian
        # Nếu dùng Mac, hãy thay bằng: terminal_cmd = ["osascript", "-e", f'tell app "Terminal" to do script "{" ".join(cmd)}"']
        return subprocess.Popen(terminal_cmd)

def main():
    # Số lượng clients muốn chạy
    num_clients = 2
    
    # Chạy server
    print("Khởi động server...")
    server_proc = run_server()
    
    # Đợi server khởi động
    print("Đợi server khởi động (5 giây)...")
    time.sleep(5)
    
    # Chạy các clients
    client_procs = []
    for i in range(1, num_clients + 1):
        print(f"Khởi động client {i}...")
        proc = run_client(i)
        client_procs.append(proc)
        time.sleep(1)  # Đợi 1 giây giữa các lần khởi động
    
    print(f"Đã khởi động server và {num_clients} clients.")
    print("Nhấn Ctrl+C để kết thúc tất cả các processes.")
    
    try:
        # Đợi cho đến khi người dùng nhấn Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Đang kết thúc các processes...")
        
        # Kết thúc các clients trước
        for proc in client_procs:
            try:
                proc.terminate()
            except:
                pass
        
        # Sau đó kết thúc server
        try:
            server_proc.terminate()
        except:
            pass
        
        print("Đã kết thúc tất cả các processes.")
        
        # Nếu đang chạy trên Windows, kill tất cả các process Python
        if os.name == 'nt':
            try:
                os.system("taskkill /f /im python.exe")
            except:
                pass

if __name__ == "__main__":
    main() 