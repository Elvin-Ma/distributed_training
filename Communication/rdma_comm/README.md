# 1 RDMA Communication
- [github](https://github.com/linux-rdma/rdma-core)

- [rdma-core doc](https://github.com/linux-rdma/rdma-core/tree/master/Documentation)

- [rdma-core tests](https://github.com/linux-rdma/rdma-core/blob/master/tests/base_rdmacm.py)

# pyverbs whl 编译

- [参考连接](https://github.com/Elvin-Ma/project_build_tools/tree/main/rdma-core_build-wheel)

# 2 example code
```sh
cd python_rdma_test

# help doc
python rdma_client.py -h

# run server
python rdma_server.py -d mlx5_2 -o write -n 5 # ibv_devices 查询设备

# run client
python rdma_client.py -d mlx5_2 -o read -n 5 -s 127.0.0.1 -p 12345 # ibv_devices 查询设备
```

# 3 RDMA 类别 TCP
| RDMA 操作 | 类比 TCP |
| ---- | ---- |
| CMID(...) | socket() |
| cmid.listen() | listen() |
| cmid.get_request() | accept() 返回新的 socket |
| client_cmid.accept() | 完成连接确认 |
| cmid.close() | 关闭监听 socket |
| cmid = client_cmid | 使用 accept 返回的新 socket 通信 |

# 4 post_send 和 rdma_write

| 对比项 | post_send() | rdma_write() |
| ---- | ---- | ---- |
| 类型 | 普通发送操作（Send） | RDMA 写操作（Remote Direct Memory Access） |
| 是否需要对方主动接收 | ✅ 需要对方调用 post_recv() 准备缓冲区 | ❌ 不需要，直接写入远程内存 |
| 数据流向 | 本地 → 远程（需对方接收） | 本地 → 远程（直接写入远程内存） |
| 是否消耗接收队列资源 | ✅ 是（需要预先注册接收缓冲区） | ❌ 否 |
| 是否阻塞 | ❌ 否（异步操作） | ❌ 否（异步操作） |
| 底层原理 | 基于 Send/Recv 队列对（QP） | 基于 RDMA 写请求 |
| 典型用途 | 握手、同步、控制信息交换 | 高性能数据传输、零拷贝内存操作 |

# 3 MR(Memory Region)

## 3.1 主动申请并注册一块可被 RDMA 访问的内存区域**

```python
self._mr = MR(self._pd, data_size_bytes, access_flags, 0)
```

- 完全由用户控制：你可以指定大小、访问权限（本地/远程读写）、绑定到特定的 Protection Domain (PD)。
- 常用于 RDMA 操作：(可接收，可主动发起)
    - **接收**远程主机通过 rdma_write() 写入的数据。
    - 注册完成后**主动发起** rdma_read() 读取远程数据。
- 必须设置合适的访问标志位，例如：
    - IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE
- 资源管理责任在用户手上，记得 close 或避免频繁申请/释放造成性能损耗

## 3.2 通过 CMID 类提供的接口来获取一个 用于 Send/Recv 的内存缓冲区
```python
recv_mr = self.cmid.reg_msgs(size + RESERVED_LEN)
```

- 内部封装了 MR 和 Buffer 的创建过程，简化了流程。
- 用于 Connection Manager 建立的连接上进行标准 Send/Recv 消息传递。
- **不适用于 RDMA 操作，因为默认没有设置 IBV_ACCESS_REMOTE_WRITE 等权限**。
- 缓冲区较小且生命周期短，适合传输元数据、握手信息等。
- 不需要自己管理 PD 或 QP，CMID 已经帮你完成了底层配置。

## 3.3 比较

|类型 |类比| 说明|
|----|----|----|
|直接建立 MR|可反复使用的保险柜|安全、稳定、支持远程存取，适合大量数据传输|
|CMID 导出 MR|一次性纸箱|用完即弃，适合传个信、打个招呼|