import socket
import threading

class SocketClient:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False

    def connect(self):
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"已连接到服务器 {self.host}:{self.port}")

            # 启动接收消息的线程
            receive_thread = threading.Thread(target=self.receive_messages)
            receive_thread.daemon = True
            receive_thread.start()

            return True

        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def receive_messages(self):
        """接收服务器消息"""
        try:
            while self.connected:
                data = self.socket.recv(1024).decode('utf-8')
                if not data:
                    print("与服务器的连接已断开")
                    self.connected = False
                    break
                # print(f"服务器回复: {data}")
                print(f"服务器已收到消息 !!!")
        except Exception as e:
            if self.connected:  # 只在连接状态下显示错误
                print(f"接收消息时出错: {e}")

    def send_message(self, message):
        """发送消息到服务器"""
        try:
            if self.connected:
                self.socket.send(message.encode('utf-8'))
                return True
            else:
                print("未连接到服务器")
                return False
        except Exception as e:
            print(f"发送消息失败: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """断开连接"""
        self.connected = False
        if self.socket:
            self.socket.close()
        print("已断开与服务器的连接")

def main():
    client = SocketClient()

    if not client.connect():
        return

    try:
        print("输入消息发送到服务器，输入 'quit' 退出:")
        while True:
            message = input()
            if message.lower() == 'quit':
                break
            if not client.send_message(message):
                break
    except KeyboardInterrupt:
        print("\n正在断开连接...")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()