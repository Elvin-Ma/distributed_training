# PCI 设备 和 PCIe
GPU 和 NVSwitch 都是 PCIe 设备，它们都连接到 PCIe 总线，并分配了 PCI 配置空间。

因为它们都连接到PCIe总线，并且被系统PCI枚举器发现，分配了PCI配置空间。操作系统通过PCI配置空间来识别设备类型、厂商ID、设备ID等，并加载相应的驱动程序


**PCIe是系统集成接口**

- 所有设备都需要通过PCIe与CPU/内存通信
- 包括GPU、NVSwitch、网卡、存储控制器等
- PCIe提供了设备发现、资源配置、中断处理等基础功能

**NVLink是专用互连**
- NVLink是GPU间的专用高速互连
- 物理上独立于PCIe
- 但管理上仍然通过PCIe配置空间控制

```shell
CPU
  ↕ PCIe
NVSwitch (PCIe设备)
  ↕ NVLink (专用互连)
GPU1 (PCIe设备)    GPU2 (PCIe设备)    GPU3 (PCIe设备)
```
> 注释:
> 所有外设都通过PCIe与CPU通信: CPU → PCIe → [GPU, 网卡, 存储控制器, 声卡, USB控制器...]；
> NVLink只处理数据交换，不处理设备枚举、资源配置等, 不能替代PCIe；
> PCIe兼容PCI的配置空间和枚举方式。操作系统和驱动程序通常使用PCI的编程模型来访问PCIe设备。因此，在软件中，我们通常统称为PCI设备，而实际上指的是PCIe设备.

# PCIe and PCIe switch
## PCIe - 点对点连接

现代PCIe采用点对点拓扑，每个设备都独享与根复合体（CPU内的PCIe控制器）的通道。

但CPU提供的PCIe通道数量是有限的（例如，**64条**）。如果你想连接4个GPU（每个需要x16通道），再加上SSD、网卡等，通道数根本不够用。

![alt text](images/image.png)

![alt text](images/image-2.png)

**与RDMA的连接关系** <br>

![alt text](images/image-3.png)

## PCIe Switch - 扩展与互连
PCIe Switch 就是为了解决上述问题而生的。它就像一个网络交换机, 其工作方式为：

- Switch通过一条x16的高速链路(**上行端口**)连接到CPU。

- Switch提供多个端口（**下行端口**），可以连接多个GPU和其他设备。

- 当GPU 0 想要与CPU通信时，Switch会将数据从连接GPU 0的端口路由到连接CPU的上行端口。

- 关键作用：**Switch还允许连接在它下面的设备之间直接通信（Peer-to-Peer, P2P）**。例如，`GPU 0 可以直接发送数据给GPU 1`，数据流经Switch，但`不需要占用上行带宽`去打扰CPU。


![alt text](images/image-1.png)

**不同的最佳通信路径** <br>

![alt text](images/image-5.png)

- 在PCIe Switch的P2P中，DMA设备是GPU自身的DMA引擎，工作在PCIe协议上，通过PCIe Switch进行寻址和路由:
  - 协议：整个过程完全遵循PCIe协议.
  - 地址：GPU A需要知道GPU B的PCIe总线地址（Bus Address）。这个地址空间是由CPU和PCIe Root Complex管理和分配的.
  - 路径：数据流经的物理链路是PCIe链路，逻辑路径通过PCIe Switch
  - 优化技术：GPU Direct P2P 就是这种DMA模式的优化实现，它避免了通过系统内存中转，实现了GPU间的直接DMA.
  - 在保持全带宽（x16）的情况下，一个高端PCIe Switch芯片通常最多能连接 4 到 5 个 GPU.
  - 一个PCIe Switch芯片在实际产品中，最常见的是连接4个GPU，以保证每个GPU拥有全额的x16带宽.
  - 不在同一个PCIe Switch下的两个GPU，它们之间的通信必须通过一个更上层的“交通枢纽”——通常是CPU内部的PCIe Root Complex——来中转.


- 在NVSwitch系统中，执行数据搬运的仍然是GPU内部的硬件单元，但它工作在NVLink协议上，通过NVSwitch进行寻址和路由，并且实现了统一内存访问:
  - 协议：整个过程使用NVLink协议，这是一种比PCIe延迟更低、带宽更高的互联协议.
  - 地址：GPU共享一个统一的虚拟地址空间。GPU A可以直接使用GPU B的虚拟地址进行访问，无需关心底层的物理地址转换，这极大地简化了编程模型并降低了延迟。
  - 路径：数据流经的物理链路是NVLink链路，逻辑路径通过NVSwitch。这是一个与PCIe网络完全并行的独立网络。
  - 通信模型：这更像是一个非一致存储访问（NUMA） 架构下的远程内存访问，而不是传统意义上的“DMA复制”。对于程序员来说，它表现为一个**共享的内存空间**.

## CPU 与 PCIe switch 与 PCIe Root Complex
一个物理CPU内部通常集成一个 PCIe Root Complex，双路服务器示意图如下：<br>

![alt text](images/deepseek_mermaid_20251009_b9e940.png)


**PCIe Root Complex:** 这是CPU`内部`的一个部件，它是整个PCIe总线系统的“根”和“大脑”。它负责发起和终止PCIe通信，管理下游的所有PCIe设备。你可以把它看作是最高级别的、集成在CPU内部的“PCIe Switch”。

**外置PCIe Switch芯片**： 这是一种`独立的芯片`，用于扩展Root Complex提供的PCIe通道，将一个x16的插槽扩展成多个x8或x4的插槽.

# NVLink and NVSwitch

**如何通过NVLink Switch 实现全互联**

- NVSwitch 芯片
  - H100 系统使用第三代 NVSwitch 芯片，每个芯片拥有 64 个 NVLink 端口.
  - 每个 H100 GPU 有 18 个 NVLink 端口, 8 个 GPU总共需要 8 × 18 = 144 个 NVLink 端口
  - 单个 NVSwitch 只有 64 个端口，远远不够, 必须使用多个NVSwitch 芯片来构建一个交换网络
  - DGX H100 中，使用了 4 个 NVSwitch 芯片来连接 8 个 H100 GPU
  - 节点内的每个 H100 GPU 都通过其全部的 18 个 NVLink 端口连接到 NVSwitch **芯片组**（通常由多个 NVSwitch 芯片构成一个交换平面.
  - NVSwitch 作为一个集中的交换矩阵，使得任何两个 GPU 之间都可以进行高速、直接的通信，而无需经过其他 GPU 或 PCIe 总线.

- 典型的 8-GPU DGX H100 系统中，每个 GPU 到交换机的带宽高达 900 GB/s（双向）。任何一对 GPU 之间的通信都能达到这个极高的带宽.

- NVSwitch 芯片组连接图
  - 某些 NVSwitch：每个连接 5 个 NVLink 端口
  - 其他 NVSwitch: 每个连接 4 个 NVLink 端口
  - 总计: 5 + 5 + 4 + 4 = 18 个 NVLink 端口(全部利用)

![alt text](images/image-4.png)


# IB 设备 和 IPoIB 协议

## IB 是什么？
InfiniBand 不是单一的物理设备，而是一整套网络技术标准体系：

```sh
InfiniBand 生态系统
    ├── 物理设备 (硬件)
    │    ├── HCA (主机通道适配器-NIC网卡) - 如 ConnectX-7
    │    ├── 交换机 - 如 Mellanox Quantum系列
    │    ├── 线缆 - 主动光缆、铜缆等
    │    └── 转接器 - QSFP、QSFP-DD等
    │
    ├── 通信协议 (软件/固件)
    │    ├── 传输协议
    │    ├── 路由协议
    │    ├── 管理协议
    │    └──  Verbs API
    │
    └── 管理工具
         ├── 子网管理器
         ├── 诊断工具
         └── 监控系统

# 以太网 vs IB

以太网 (标准)           vs    InfiniBand (标准)
    ├── 网卡                  ├── HCA
    ├── 交换机                ├── InfiniBand交换机
    ├── 网线                  ├── InfiniBand线缆
    └── TCP/IP协议            └── InfiniBand协议栈
```

InfiniBand主机通道适配器（HCA，即网卡）、InfiniBand交换机、InfiniBand线缆（如铜缆、光缆）

- NVIDIA的NVSwitch是一个物理芯片，它被焊接在像HGX这样的服务器主板上；
- HCA(主机通道适配器 host channel adaptor) ： 对应NIC, 为和RDMA技术优化的**超级网卡**;
- HCA卡, 在电气和逻辑上，正是连接到 PCIe Switch（或CPU内部的PCIe Root Complex）的一个下游端口;
- 将IB HCA卡插入服务器的PCIe插槽时，也就相当于连接到 PCIe Switch.
- GPUDirect RDMA允许网络适配器（如InfiniBand HCA）直接访问GPU显存，而无需通过主机内存拷贝。这减少了数据传输的延迟和CPU开销。
- GDA 和 GPUDirect RDMA :

**RDMA 硬件通信链路**
![alt text](images/image-6.png)

*传统的以太网中，我们有 NIC, 在 InfiniBand 网络中，对应的硬件就是 HCA(主机通道适配器)， 专门为高性能计算和RDMA技术优化的“超级网卡”*

**IB 交换机冗余设计，类似于NVSwitch**

![alt text](images/image-7.png)

一个HCA 网卡可以连接到不同PCIe Switch 设备.

![alt text](deepseek_mermaid_20251011_aa835c.png)

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


## IB 线缆

| 代际  | 总带宽(单向) | 带宽GB | 单链路速率 | 物理链路数 | 连接器类型  |
|-------|------------ |--------|------------|------------|-------------|
| NDR   | 400 Gbps    | 50GB   | 100 Gbps   | 4x         | QSFP-DD     |
| HDR   | 200 Gbps    | 25GB   | 50 Gbps    | 4x         | QSFP56      |
| EDR   | 100 Gbps    | 12.5GB | 25 Gbps    | 4x         | QSFP28      |
| FDR   | 56 Gbps     | 7GB    | 14 Gbps    | 4x         | QSFP+       |

