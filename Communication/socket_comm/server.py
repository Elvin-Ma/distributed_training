import socket
import threading

class SocketServer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False

    def start_server(self):
        """启动服务器"""
        try:
            # 创建 socket 对象
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 设置端口复用，避免"Address already in use"错误
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # 绑定地址和端口
            self.socket.bind((self.host, self.port))
            # 开始监听，最大连接数为5
            self.socket.listen(5)

            print(f"服务器已启动，监听 {self.host}:{self.port}")
            self.running = True

            while self.running:
                # 接受客户端连接
                client_socket, client_address = self.socket.accept()
                print(f"接收到来自 {client_address} 的连接")

                # 为每个客户端创建新线程
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, client_address)
                )
                client_thread.daemon = True
                client_thread.start()

        except Exception as e:
            print(f"服务器错误: {e}")
        finally:
            self.stop_server()

    def handle_client(self, client_socket, client_address):
        """处理客户端连接"""
        try:
            # 发送欢迎消息
            welcome_msg = "欢迎连接到服务器！发送 'quit' 退出连接。\n"
            client_socket.send(welcome_msg.encode('utf-8'))

            while True:
                # 接收客户端数据
                data = client_socket.recv(1024).decode('utf-8')

                if not data or data.strip().lower() == 'quit':
                    print(f"客户端 {client_address} 断开连接")
                    break

                print(f"收到来自 {client_address} 的消息: {data}")

                # 处理消息并回复
                response = f"服务器已收到你的消息: {data}"
                client_socket.send(response.encode('utf-8'))

        except Exception as e:
            print(f"处理客户端 {client_address} 时出错: {e}")
        finally:
            client_socket.close()

    def stop_server(self):
        """停止服务器"""
        self.running = False
        if self.socket:
            self.socket.close()
        print("服务器已停止")

if __name__ == "__main__":
    server = SocketServer()
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n正在关闭服务器...")
        server.stop_server()