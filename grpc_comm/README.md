# 1 环境安装

```sh
pip install grpcio grpcio-tools protobuf
```

# 2 编译proto文件
```proto
syntax = "proto3";

package example;

// 定义服务接口
service HelloService {
  // 单向请求响应模式
  rpc SendMessage (HelloRequest) returns (HelloResponse) {}
}

// 请求消息类型
message HelloRequest {
  string name = 1;
}

// 响应消息类型
message HelloResponse {
  string message = 1;
}
```

# 3 生成python文件
```bash
# 生成python代码
python -m grpc_tools.protoc  -I. --python_out=. --grpc_python_out=. hello.proto

# 生成两个文件
# - `hello_pb2.py` ：数据序列化类
# - `hello_pb2_grpc.py` ：gRPC服务接口类

# 也可程序Java/Go/C++ 等类型代码
```

# 4 编写服务端代码
```python
import grpc
import hello_pb2
import hello_pb2_grpc
from concurrent import futures

class HelloServiceServicer(hello_pb2_grpc.HelloServiceServicer):
    def SendMessage(self, request, context):
        response = hello_pb2.HelloResponse()
        response.message  = f"Hello, {request.name}!"
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    hello_pb2_grpc.add_HelloServiceServicer_to_server(
        HelloServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

# 5 编写客户端代码
```python
import grpc
import hello_pb2
import hello_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = hello_pb2_grpc.HelloServiceStub(channel)

    request = hello_pb2.HelloRequest(name="World")
    response = stub.SendMessage(request)

    print("Received:", response.message)

if __name__ == '__main__':
    run()
```

# 6 comm 原理

```python
# Client 端 和 Server 端调用流程
# Client                     Server
#    │                           │
#    ├─────── SendMessage ──────►
#    │       (HelloRequest)      │
#    │                           │
#    ◄─────── Response ──────────┤
#    │       (HelloResponse)     │
```

# 7 代码运行

## 4.5 运行服务端和客户端
```bash
python server.py

# other terminal
python client.py

# 输出：Received: Hello, World!
```