# IB 设备 和 IPoIB 协议

**相关概念**

- IPoIB 是 IP over InfiniBand 的缩写，即 “基于 InfiniBand 的 IP 协议”，是一种在 InfiniBand（IB）高性能网络上运行 IP 协议的技术。


| 特性       | mlx5_1 (IB 设备)       | ibs110 (IPoIB 接口)     |
| ---------- | ---------------------- | ----------------------- |
| 层级       | 硬件/驱动层            | 网络协议层              |
| 关注点     | 物理连接状态、速率、LID | IP 地址、子网掩码、MTU  |
| 配置命令   | ibstatus, ibdev2lid     | ifconfig, ip addr       |
| 依赖关系   | 必须先有物理连接       | 依赖 IB 设备状态        |



**IB 设备状态**

```shell
# mlx5_1: Mellanox ConnectX-5 系列网卡设备名称
# port 1: 设备的第一个端口
Infiniband device 'mlx5_1' port 1 status:
        # 全局标识符，类似以太网的MAC地址
        # fe80::/10 前缀表示这是一个链路本地地址
        default gid:     fe80:0000:0000:0000:08c0:eb03:00f7:c3d6
        # 本地标识符，在InfiniBand子网内唯一的16位标识
        base lid:        0x9
        # 子网管理器的LID，当前管理这个子网的控制器
        sm lid:          0xb
        # 端口逻辑状态：已激活并可传输数据, 状态代码4表示端口完全正常运行
        state:           4: ACTIVE
        # 物理连接状态：链路已连接，状态代码5表示链路已连接
        phys state:      5: LinkUp
        # 当前4x链路总速率：100 Gbps
        # 4X：使用4条链路通道
        # EDR：Enhanced Data Rate，InfiniBand标准之一
        # 除了EDR 还有：FDR、QDR、HDR、SDR等等
        rate:            100 Gb/sec (4X EDR)
        # 确认使用的是原生InfiniBand协议
        link_layer:      InfiniBand
```

**IPoIB 接口状态**

```shell
# ibs110: IPoIB 接口名称
# flags=4163: 接口标志的十六进制值
# <UP,BROADCAST,RUNNING,MULTICAST>: 接口状态标志
# up: 接口已启动，broadcast: 接口可广播，running: 接口已运行，multicast: 接口支持组播
# mtu 2044: 最大传输单元为2044字节
ibs110: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 2044
        # inet: IPv4地址族, netmask : 子网掩码, broadcast: 广播地址
        inet 10.10.100.7  netmask 255.255.248.0  broadcast 10.10.103.255
        # inet6: IPv6地址族, prefixlen: 前缀长度为64位, scopeid: 域ID 作用域为链路本地
        inet6 fe80::ac0:eb03:f7:c3d6  prefixlen 64  scopeid 0x20<link>
        # unspec: 未指定的地址族（可能是InfiniBand的特殊格式）
        # txqueuelen 256: 传输队列长度为256个数据包
        unspec 00-00-0A-7F-FE-80-00-00-00-00-00-00-00-00-00-00  txqueuelen 256  (UNSPEC)
        # RX packets 5: 接收了5个数据包
        # RX bytes 500: 接收总字节数为500字节的数据
        RX packets 5  bytes 500 (500.0 B)
        #  接收错误数为，丢弃的数据包数，溢出错误， 帧错误
        RX errors 0  dropped 0  overruns 0  frame 0
        # TX packets 503: 发送了503个数据包，
        # bytes 30276: 发送总字节数为30276字节（约30.2KB）
        TX packets 503  bytes 30276 (30.2 KB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```



