# Pyverbs

Pyverbs provides a Python API over rdma-core, the Linux userspace C API for
the RDMA stack.

## Goals

1. Provide easier access to RDMA: RDMA has a steep learning curve as is and
   the C interface requires the user to initialize multiple structs before
   having usable objects. Pyverbs attempts to remove much of this overhead and
   provide a smoother user experience.
2. Improve our code by providing a test suite for rdma-core. This means that
   new features will be tested before merge, and it also means that users and
   distros will have tests for new and existing features, as well as the means
   to create them quickly.
3. Stay up-to-date with rdma-core - cover new features during development and
   provide a test / unit-test alongside the feature.

## Limitations

Python handles memory for users. As a result, memory is allocated by Pyverbs
when needed (e.g. user buffer for memory region). The memory will be accessible
to the users, but not allocated or freed by them.

## Usage Examples
Note that all examples use a hard-coded device name ('mlx5_0').
##### Open an IB device

Import the device module and open a device by name:

```python
import pyverbs.device as d
ctx = d.Context(name='mlx5_0')
```

'ctx' is Pyverbs' equivalent to rdma-core's ibv_context. At this point, the IB
device is already open and ready to use.

##### Query a device
```python
import pyverbs.device as d
ctx = d.Context(name='mlx5_0')
attr = ctx.query_device()
print(attr)
FW version            : 16.24.0185
Node guid             : 9803:9b03:0000:e4c6
Sys image GUID        : 9803:9b03:0000:e4c6
Max MR size           : 0xffffffffffffffff
Page size cap         : 0xfffffffffffff000
Vendor ID             : 0x2c9
Vendor part ID        : 4119
HW version            : 0
Max QP                : 262144
Max QP WR             : 32768
Device cap flags      : 3983678518
Max SGE               : 30
Max SGE RD            : 30
MAX CQ                : 16777216
Max CQE               : 4194303
Max MR                : 16777216
Max PD                : 16777216
Max QP RD atom        : 16
Max EE RD atom        : 0
Max res RD atom       : 4194304
Max QP init RD atom   : 16
Max EE init RD atom   : 0
Atomic caps           : 1
Max EE                : 0
Max RDD               : 0
Max MW                : 16777216
Max raw IPv6 QPs      : 0
Max raw ethy QP       : 0
Max mcast group       : 2097152
Max mcast QP attach   : 240
Max AH                : 2147483647
Max FMR               : 0
Max map per FMR       : 2147483647
Max SRQ               : 8388608
Max SRQ WR            : 32767
Max SRQ SGE           : 31
Max PKeys             : 128
local CA ack delay    : 16
Phys port count       : 1
```

'attr' is Pyverbs' equivalent to ibv_device_attr. Pyverbs will provide it to
the user upon completion of the call to ibv_query_device.

##### Query GID

```python
import pyverbs.device as d
ctx = d.Context(name='mlx5_0')
gid = ctx.query_gid(port_num=1, index=3)
print(gid)
0000:0000:0000:0000:0000:ffff:0b87:3c08
```

'gid' is Pyverbs' equivalent to ibv_gid, provided to the user by Pyverbs.

##### Query port
The following code snippet provides an example of pyverbs' equivalent of
querying a port. Context's query_port() command wraps ibv_query_port().
The example below queries the first port of the device.
```python
import pyverbs.device as d
ctx=d.Context(name='mlx5_0')
port_attr = ctx.query_port(1)
print(port_attr)
Port state              : Active (4)
Max MTU                 : 4096 (5)
Active MTU              : 1024 (3)
SM lid                  : 0
Port lid                : 0
lmc                     : 0x0
Link layer              : Ethernet
Max message size        : 0x40000000
Port cap flags          : IBV_PORT_CM_SUP IBV_PORT_IP_BASED_GIDS
Port cap flags 2        :
max VL num              : 0
Bad Pkey counter        : 0
Qkey violations counter : 0
Gid table len           : 256
Pkey table len          : 1
SM sl                   : 0
Subnet timeout          : 0
Init type reply         : 0
Active width            : 4X (2)
Ative speed             : 25.0 Gbps (32)
Phys state              : Link up (5)
Flags                   : 1
```

##### Extended query device
The example below shows how to open a device using pyverbs and query the
extended device's attributes.
Context's query_device_ex() command wraps ibv_query_device_ex().
```python
import pyverbs.device as d

ctx = d.Context(name='mlx5_0')
attr = ctx.query_device_ex()
attr.max_dm_size
131072
attr.rss_caps.max_rwq_indirection_table_size
2048
```

# 2 Create RDMA objects
## 2.1 PD
PD (Protection Domain) 是 保护域 的概念，它是 pyverbs 中的核心资源管理单元。下面我将详细解释 PD 的作用、工作原理和实际使用方法：<br>

- 核心作用: 提供内存访问保护和资源隔离;
- 类比理解：类似操作系统的进程地址空间，隔离不同应用的内存访问；
- 技术定义：一个逻辑容器，包含一组关联的RDMA资源，如QP、MR等, 确保只有同PD内的资源才能直接交互；

**为什么需要PD**
|场景 |	无 PD|	有 PD|
|--|--|--|
|内存安全|	任意 QP 可访问所有内存|	仅同 PD 的 QP 可访问 MR|
|多租户隔离|	不同用户资源可能冲突|	用户独占 PD 安全隔离|
|资源管理|	全局资源难回收|	PD 释放自动销毁所有子资源|
|权限控制|	单一权限模型|	每个 PD 独立权限策略|


![alt text](image.png)

- 硬件实现：HCA 通过 PD Key 验证资源所属关系

- 访问规则：

    - 同 PD 内：QP 可直接访问 MR

    - 跨 PD 访问：需特殊授权（如 Memory Windows）

下面的示例展示了如何打开设备并使用其上下文进行创建PD。

```python
import pyverbs.device as d
from pyverbs.pd import PD

with d.Context(name='mlx5_0') as ctx:
    pd = PD(ctx)
```

---

## 2.2 MR

在 RDMA 编程中，MR（Memory Region） 是核心概念之一，它代表了已注册到 RDMA 设备的内存区域.


| MR 类型       | 创建方式                                   | 特点                  | 适用场景               |
|---------------|--------------------------------------------|-----------------------|------------------------|
| 标准 MR       | `MR(pd, buffer, size, access)`             | 用户提供内存          | 通用场景               |
| 匿名 MR       | `MR(pd, length=size, access=access)`       | 驱动分配内存          | 性能敏感型应用         |
| 设备 MR       | `MR(pd, ..., pd_flags=IBV_PD_MEM_DEVICE)`  | 设备内存              | GPU RDMA               |
| iWARP MR      | `MR(pd, ..., comp_mask=IBV_MR_COMP_MASK_INDIRECT)` | 间接内存 | iWARP 设备             |


下面的示例展示了如何使用pyverbs创建MR。与C类似，在创建之前必须打开一个设备，并且必须分配一个PD。

```python
import pyverbs.device as d
from pyverbs.pd import PD
from pyverbs.mr import MR
import pyverbs.enums as e

with d.Context(name='mlx5_0') as ctx:
    with PD(ctx) as pd:
        mr_len = 1000
        flags = e.IBV_ACCESS_LOCAL_WRITE
        mr = MR(pd, mr_len, flags)
```

---

## 2.3 Memory window

MW（Memory Window）是一种高级内存管理机制，用于更细粒度地控制远程访问权限。MW允许动态地授予或撤销对已注册内存区域（MR）的特定部分的访问权限，而无需重新注册整个MR. 这在需要频繁更改访问权限或共享内存子集的**多租户**环境中特别有用。<br>

以下示例展示了创建类型 1 内存窗口（memory window，简称 MW）的等效操作。
该操作包括打开设备以及分配必要的页目录（Page Directory，简称 PD）。
用户应在**解除内存窗口绑定或关闭内存窗口之后，才能注销该内存窗口所绑定的内存区域（Memory Region，简称 MR）** <br>

```python
import pyverbs.device as d
from pyverbs.pd import PD
from pyverbs.mr import MW
import pyverbs.enums as e

with d.Context(name='mlx5_0') as ctx:
    with PD(ctx) as pd:
        mw = MW(pd, e.IBV_MW_TYPE_1)
```
---

## 2.4 Device memory

在 PyVerbs（一个用于 RDMA 编程的 Python 库）中，DM 是 Device Manager 的缩写，用于管理 **RDMA 设备（如 InfiniBand 或 RoCE 网卡）**。它提供了发现、选择和操作 RDMA 设备的核心功能。<br>


以下代码片段展示了如何使用设备的内存来分配一个 DM（直接内存对象，Direct Memory object）。<br>

```python
import random

from pyverbs.device import DM, AllocDmAttr
import pyverbs.device as d

with d.Context(name='mlx5_2') as ctx:
    attr = ctx.query_device_ex()
    if attr.max_dm_size != 0:
        dm_len = random.randint(4, attr.max_dm_size)
        dm_attrs = AllocDmAttr(dm_len)
        dm = DM(ctx, dm_attrs)
```

---

## 2.5 DMMR

DMMR（Device Memory Memory Region） 是一种特殊的内存区域类型，它允许直接使用设备本地内存（如 GPU 显存）进行 RDMA 操作，而无需通过主机内存进行数据复制。这是实现 GPU Direct RDMA (GDR) 的关键技术，能显著减少数据传输延迟。<br>

以下示例展示了如何打开一个 DMMR（设备内存内存区域，Device Memory MR），即使用**设备自身的内存**而非用户分配的缓冲区。<br>

```python
import random

from pyverbs.device import DM, AllocDmAttr
from pyverbs.mr import DMMR
import pyverbs.device as d
from pyverbs.pd import PD
import pyverbs.enums as e

with d.Context(name='mlx5_0') as ctx:
    attr = ctx.query_device_ex()
    if attr.max_dm_size != 0:
        dm_len = random.randint(4, attr.max_dm_size)
        dm_attrs = AllocDmAttr(dm_len)
        dm_mr_len = random.randint(4, dm_len)
        with DM(ctx, dm_attrs) as dm:
            with PD(ctx) as pd:
                dm_mr = DMMR(pd, dm_mr_len, e.IBV_ACCESS_ZERO_BASED, dm=dm,
                             offset=0)
```

## 2.6 CQ

在 PyVerbs 中，**CQ (Completion Queue，完成队列) 是 RDMA 编程的核心组件之一**，用于`异步通知工作请求 (Work Request, WR) 的完成状态`。当应用程序提交的 RDMA 操作（如发送、接收、RDMA 读写等）被硬件处理完毕后，CQ `会生成完成队列元素 (CQE) 来通知应用程序`。

**核心作用** <br>

> 1. 异步事件通知: 当工作请求被网卡处理完成时，自动生成完成通知; <br>
> 2. 状态报告: 报告操作成功/失败状态及错误信息; <br>
> 3. 批处理支持: 一个 CQ 可关联多个队列对 (QP); <br>
> 4. 性能优化: 减少应用程序轮询开销; <br>

以下代码片段展示了如何使用 pyverbs 创建**完成队列**（Completion Queue，简称 CQ）。pyverbs 同时支持普通完成队列（CQ）和扩展完成队列（Extended CQ，简称 CQEX）。就像在 C 语言中一样，完成队列可以在有或没有完成通道（completion channel）的情况下创建，代码片段展示了这两种情况。CQ 的第三个参数是 cq_context，即用户自定义的上下文。在我们的代码片段中，我们使用了 None。

**CQ** <br>

```python
import random

from pyverbs.cq import CompChannel, CQ
import pyverbs.device as d

with d.Context(name='mlx5_0') as ctx:
    num_cqes = random.randint(0, 200) # Just arbitrary values. Max value can be
                                      # found in device attributes
    comp_vector = 0 # An arbitrary value. comp_vector is limited by the
                    # context's num_comp_vectors
    if random.choice([True, False]):
        with CompChannel(ctx) as cc:
            cq = CQ(ctx, num_cqes, None, cc, comp_vector)
    else:
        cq = CQ(ctx, num_cqes, None, None, comp_vector)
    print(cq)

CQ
Handle                : 0
CQEs                  : 63
```

**CQEX** <br>

```python
import random

from pyverbs.cq import CqInitAttrEx, CQEX
import pyverbs.device as d
import pyverbs.enums as e

with d.Context(name='mlx5_0') as ctx:
    num_cqe = random.randint(0, 200)
    wc_flags = e.IBV_WC_EX_WITH_CVLAN
    comp_mask = 0 # Not using flags in this example
    # completion channel is not used in this example
    attrs = CqInitAttrEx(cqe=num_cqe, wc_flags=wc_flags, comp_mask=comp_mask,
                         flags=0)
    print(attrs)
    cq_ex = CQEX(ctx, attrs)
    print(cq_ex)
    Number of CQEs        : 10
WC flags              : IBV_WC_EX_WITH_CVLAN
comp mask             : 0
flags                 : 0

Extended CQ:
Handle                : 0
CQEs                  : 15
```

## 2.7 Addressing related objects

以下代码演示了创建全局路由（GlobalRoute）、地址句柄属性（AHAttr）和地址句柄（AH）对象的过程。
该示例创建了一个全局的地址句柄（AH），因此它无需修改即可在 RoCE（RDMA over Converged Ethernet，融合以太网的 RDMA）上运行。

```python

from pyverbs.addr import GlobalRoute, AHAttr, AH
import pyverbs.device as d
from pyverbs.pd import PD

with d.Context(name='mlx5_0') as ctx:
    port_number = 1
    gid_index = 0  # GID index 0 always exists and valid
    gid = ctx.query_gid(port_number, gid_index)
    gr = GlobalRoute(dgid=gid, sgid_index=gid_index)
    ah_attr = AHAttr(gr=gr, is_global=1, port_num=port_number)
    print(ah_attr)
    with PD(ctx) as pd:
        ah = AH(pd, attr=ah_attr)
DGID                  : fe80:0000:0000:0000:9a03:9bff:fe00:e4bf
flow label            : 0
sgid index            : 0
hop limit             : 1
traffic class         : 0
```

## 2.8 QP

在 RDMA 编程中，QP（Queue Pair，队列对） 是最核心的通信抽象，它是所有 RDMA 操作（发送、接收、RDMA读写、原子操作等）的载体。在 PyVerbs 中，QP 提供了高效、低延迟的数据传输能力。<br>

### 2.8.1 基本概念

- QP 队列对 结构

![alt text](image-1.png)

> 发送队列 (SQ)：存放要执行的操作（发送、RDMA写等） <br>
> 接收队列 (RQ)：存放接收缓冲区的工作请求 <br>
> 每个队列有独立的 CQ：用于通知操作完成 <br>


### 2.8.2 代码

以下代码片段将演示如何创建一个队列对（QP）以及执行一个简单的 post_send 操作。如需更复杂的示例，请参阅 **pyverbs 的 examples 部分。**

```python
from pyverbs.qp import QPCap, QPInitAttr, QPAttr, QP
from pyverbs.addr import GlobalRoute
from pyverbs.addr import AH, AHAttr
import pyverbs.device as d
import pyverbs.enums as e
from pyverbs.pd import PD
from pyverbs.cq import CQ
import pyverbs.wr as pwr


ctx = d.Context(name='mlx5_0')
pd = PD(ctx)
cq = CQ(ctx, 100, None, None, 0)
cap = QPCap(100, 10, 1, 1, 0)
qia = QPInitAttr(cap=cap, qp_type = e.IBV_QPT_UD, scq=cq, rcq=cq)
# A UD QP will be in RTS if a QPAttr object is provided
udqp = QP(pd, qia, QPAttr())
port_num = 1
gid_index = 3 # Hard-coded for RoCE v2 interface
gid = ctx.query_gid(port_num, gid_index)
gr = GlobalRoute(dgid=gid, sgid_index=gid_index)
ah_attr = AHAttr(gr=gr, is_global=1, port_num=port_num)
ah=AH(pd, ah_attr)
wr = pwr.SendWR()
wr.set_wr_ud(ah, 0x1101, 0) # in real life, use real values
udqp.post_send(wr)
```
## 2.9 Extended QP
扩展的队列对（QP）为用户暴露了一组**新**的 QP 发送操作接口——提供了对新发送操作码、供应商特定发送操作码甚至**供应商特定 QP 类型**的可扩展性支持。
现在，pyverbs 提供了创建此类 QP 所需的接口。
需要注意的是，在使用扩展 QP 的新发送提交（post send）机制时，comp_mask 中的 IBV_QP_INIT_ATTR_SEND_OPS_FLAGS 是必填项。

```python
from pyverbs.qp import QPCap, QPInitAttrEx, QPAttr, QPEx
import pyverbs.device as d
import pyverbs.enums as e
from pyverbs.pd import PD
from pyverbs.cq import CQ


ctx = d.Context(name='mlx5_0')
pd = PD(ctx)
cq = CQ(ctx, 100)
cap = QPCap(100, 10, 1, 1, 0)
qia = QPInitAttrEx(qp_type=e.IBV_QPT_UD, scq=cq, rcq=cq, cap=cap, pd=pd,
                   comp_mask=e.IBV_QP_INIT_ATTR_SEND_OPS_FLAGS| \
                   e.IBV_QP_INIT_ATTR_PD)
qp = QPEx(ctx, qia)
```

## 2.10 XRCD

在 PyVerbs 中，XRCD（Extended Reliable Connection Domain） 是一种**高级资源管理对象**，主要用于支持 XRC（Extended Reliable Connection）协议，这种协议在 HPC（高性能计算）和大规模集群环境中特别重要，用于优化大规模 RDMA 通信的资源利用率。

**传统 RC（可靠连接）QP 需要每个进程对之间建立独立连接, 而XRC 协议多个进程共享同一个物理连接;**

![alt text](image-2.png)

下面的代码演示了XRCD对象的创建。

```python
from pyverbs.xrcd import XRCD, XRCDInitAttr
import pyverbs.device as d
import pyverbs.enums as e
import stat
import os


ctx = d.Context(name='ibp0s8f0')
xrcd_fd = os.open('/tmp/xrcd', os.O_RDONLY | os.O_CREAT,
                  stat.S_IRUSR | stat.S_IRGRP)
init = XRCDInitAttr(e.IBV_XRCD_INIT_ATTR_FD | e.IBV_XRCD_INIT_ATTR_OFLAGS,
                    os.O_CREAT, xrcd_fd)
xrcd = XRCD(ctx, init)
```

## 2.11 SRQ (共享接受队列)

在Pyverbs库中，共享接收队列（Shared Receive Queue, SRQ）是RDMA编程中用于优化资源管理的重要机制。以下是其核心介绍及使用要点：<br>

SRQ是多个队列对（QP）共享的接收队列，允许不同QP复用同一接收队列资源，避免为每个QP单独维护RQ.

以下代码片段将演示如何创建一个 XRC 共享接收队列（SRQ）对象。如需更复杂的示例，请参阅 pyverbs/tests/test_odp 目录下的相关代码。<br>

```python
from pyverbs.xrcd import XRCD, XRCDInitAttr
from pyverbs.srq import SRQ, SrqInitAttrEx
import pyverbs.device as d
import pyverbs.enums as e
from pyverbs.cq import CQ
from pyverbs.pd import PD
import stat
import os


ctx = d.Context(name='ibp0s8f0')
pd = PD(ctx)
cq = CQ(ctx, 100, None, None, 0)
xrcd_fd = os.open('/tmp/xrcd', os.O_RDONLY | os.O_CREAT,
                  stat.S_IRUSR | stat.S_IRGRP)
init = XRCDInitAttr(e.IBV_XRCD_INIT_ATTR_FD | e.IBV_XRCD_INIT_ATTR_OFLAGS,
                    os.O_CREAT, xrcd_fd)
xrcd = XRCD(ctx, init)

srq_attr = SrqInitAttrEx(max_wr=10)
srq_attr.srq_type = e.IBV_SRQT_XRC
srq_attr.pd = pd
srq_attr.xrcd = xrcd
srq_attr.cq = cq
srq_attr.comp_mask = e.IBV_SRQ_INIT_ATTR_TYPE | e.IBV_SRQ_INIT_ATTR_PD | \
                     e.IBV_SRQ_INIT_ATTR_CQ | e.IBV_SRQ_INIT_ATTR_XRCD
srq = SRQ(ctx, srq_attr)
```

---


## 2.12 Open an mlx5 provider
提供者（provider）本质上是一个**具备驱动特定额外功能的上下文（Context）对象**。因此，它**继承**自上下文（Context）类。在传统的流程中，上下文（Context）会遍历 InfiniBand（IB）设备，并打开与用户给定名称（通过 name= 参数指定）相匹配的设备。当同时给出了提供者属性（attr= 参数）时，上下文（Context）会将相关的 ib_device 分配给其设备成员变量，这样提供者就能够以特定方式打开该设备，如下文所演示的那样：<br>

```python
import pyverbs.providers.mlx5.mlx5dv as m
from pyverbs.pd import PD
attr = m.Mlx5DVContextAttr()  # Default values are fine
ctx = m.Mlx5Context(attr=attr, name='rocep0s8f0')
# The provider context can be used as a regular Context, e.g.:
pd = PD(ctx)  # Success
```

## 2.13 Query an mlx5 provider

在打开一个 mlx5 提供者之后，用户可以使用针对该设备的特定查询来获取非传统（non-legacy）属性。以下代码片段演示了如何进行这一操作。<br>

```python
import pyverbs.providers.mlx5.mlx5dv as m
ctx = m.Mlx5Context(attr=m.Mlx5DVContextAttr(), name='ibp0s8f0')
mlx5_attrs = ctx.query_mlx5_device()
print(mlx5_attrs)
Version             : 0
Flags               : CQE v1, Support CQE 128B compression, Support CQE 128B padding, Support packet based credit mode (in RC QP)
comp mask           : CQE compression, SW parsing, Striding RQ, Tunnel offloads, Dynamic BF regs, Clock info update, Flow action flags
CQE compression caps:
  max num             : 64
  supported formats   : with hash, with RX checksum CSUM, with stride index
SW parsing caps:
  SW parsing offloads :
  supported QP types  :
Striding RQ caps:
  min single stride log num of bytes: 6
  max single stride log num of bytes: 13
  min single wqe log num of strides: 9
  max single wqe log num of strides: 16
  supported QP types  : Raw Packet
Tunnel offloads caps:
Max dynamic BF registers: 1024
Max clock info update [nsec]: 1099511
Flow action flags   : 0
```

## 2.14  Create an mlx5 QP
使用 Mlx5Context 对象时，用户可以创建一个传统的队列对（QP）（创建过程与传统方式相同），或者创建一个 mlx5 特定的队列对（QP）。mlx5 队列对（QP）继承自普通队列对（QP），但它的构造函数接收一个名为 dv_init_attr 的关键字参数。如果用户提供了这个参数，队列对（QP）将使用 mlx5dv_create_qp 函数来创建，而不是使用 ibv_create_qp_ex 函数。以下代码片段演示了如何创建一个动态连接（DC，Dynamically Connected）的队列对（QP）和一个使用 mlx5 特定功能的原始数据包队列对（Raw Packet QP），这些功能在传统接口中是不可用的。目前，pyverbs 仅支持创建动态连接接口（DCI，Dynamically Connected Interface）。动态连接传输（DCT，Dynamically Connected Transport）的支持将在后续的拉取请求（PR，Pull Request）中添加。
```python
from pyverbs.providers.mlx5.mlx5dv import Mlx5Context, Mlx5DVContextAttr
from pyverbs.providers.mlx5.mlx5dv import Mlx5DVQPInitAttr, Mlx5QP
import pyverbs.providers.mlx5.mlx5_enums as me
from pyverbs.qp import QPInitAttrEx, QPCap
import pyverbs.enums as e
from pyverbs.cq import CQ
from pyverbs.pd import PD

with Mlx5Context(name='rocep0s8f0', attr=Mlx5DVContextAttr()) as ctx:
    with PD(ctx) as pd:
        with CQ(ctx, 100) as cq:
            cap = QPCap(100, 0, 1, 0)
            # Create a DC QP of type DCI
            qia = QPInitAttrEx(cap=cap, pd=pd, scq=cq, qp_type=e.IBV_QPT_DRIVER,
                               comp_mask=e.IBV_QP_INIT_ATTR_PD, rcq=cq)
            attr = Mlx5DVQPInitAttr(comp_mask=me.MLX5DV_QP_INIT_ATTR_MASK_DC)
            attr.dc_type = me.MLX5DV_DCTYPE_DCI

            dci = Mlx5QP(ctx, qia, dv_init_attr=attr)

            # Create a Raw Packet QP using mlx5-specific capabilities
            qia.qp_type = e.IBV_QPT_RAW_PACKET
            attr.comp_mask = me.MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS
            attr.create_flags = me.MLX5DV_QP_CREATE_ALLOW_SCATTER_TO_CQE |\
                                me.MLX5DV_QP_CREATE_TIR_ALLOW_SELF_LOOPBACK_UC |\
                                me.MLX5DV_QP_CREATE_TUNNEL_OFFLOADS
            qp = Mlx5QP(ctx, qia, dv_init_attr=attr)
```

## 2.15 Create an mlx5 CQ

Mlx5Context 还允许用户创建一个 mlx5 特定的完成队列（CQ）。Mlx5CQ 继承自扩展完成队列（CQEX），但它的构造函数接收 3 个参数，而不是 2 个。第 3 个参数是一个名为 dv_init_attr 的关键字参数。如果用户提供了这个参数，完成队列（CQ）将使用 mlx5dv_create_cq 函数来创建。
以下代码片段展示了这一简单的创建过程。

```python
from pyverbs.providers.mlx5.mlx5dv import Mlx5Context, Mlx5DVContextAttr
from pyverbs.providers.mlx5.mlx5dv import Mlx5DVCQInitAttr, Mlx5CQ
import pyverbs.providers.mlx5.mlx5_enums as me
from pyverbs.cq import CqInitAttrEx

with Mlx5Context(name='rocep0s8f0', attr=Mlx5DVContextAttr()) as ctx:
    cqia = CqInitAttrEx()
    mlx5_cqia = Mlx5DVCQInitAttr(comp_mask=me.MLX5DV_CQ_INIT_ATTR_MASK_COMPRESSED_CQE,
                                 cqe_comp_res_format=me.MLX5DV_CQE_RES_FORMAT_CSUM)
    cq = Mlx5CQ(ctx, cqia, dv_init_attr=mlx5_cqia)
```

## 2.16 CMID
以下代码片段将演示如何创建一个 CMID 对象，该对象代表 rdma_cm_id C 结构体，并在两个对等实体之间建立连接。目前，仅支持同步控制路径（rdma_create_ep）。如需更复杂的示例，请参阅 tests/test_rdmacm 目录下的代码。

```python
from pyverbs.qp import QPInitAttr, QPCap
from pyverbs.cmid import CMID, AddrInfo
import pyverbs.cm_enums as ce


cap = QPCap(max_recv_wr=1)
qp_init_attr = QPInitAttr(cap=cap)
addr = '11.137.14.124'
port = '7471'

# Passive side

sai = AddrInfo(src=addr, src_service=port, port_space=ce.RDMA_PS_TCP, flags=ce.RAI_PASSIVE)
sid = CMID(creator=sai, qp_init_attr=qp_init_attr)
sid.listen()  # listen for incoming connection requests
new_id = sid.get_request()  # check if there are any connection requests
new_id.accept()  # new_id is connected to remote peer and ready to communicate

# Active side

cai = AddrInfo(src=addr, dst=addr, dst_service=port, port_space=ce.RDMA_PS_TCP)
cid = CMID(creator=cai, qp_init_attr=qp_init_attr)
cid.connect()  # send connection request to passive addr
```

## 2.17 ParentDomain
The following code demonstrates the creation of Parent Domain object.
In this example, a simple Python allocator is defined. It uses MemAlloc class to
allocate aligned memory using a C style aligned_alloc.
```python
from pyverbs.pd import PD, ParentDomainInitAttr, ParentDomain, \
    ParentDomainContext
from pyverbs.device import Context
import pyverbs.mem_alloc as mem


def alloc_p_func(pd, context, size, alignment, resource_type):
    p = mem.posix_memalign(size, alignment)
    return p


def free_p_func(pd, context, ptr, resource_type):
    mem.free(ptr)


ctx = Context(name='rocep0s8f0')
pd = PD(ctx)
pd_ctx = ParentDomainContext(pd, alloc_p_func, free_p_func)
pd_attr = ParentDomainInitAttr(pd=pd, pd_context=pd_ctx)
parent_domain = ParentDomain(ctx, attr=pd_attr)
```

##### MLX5 VAR
The following code snippet demonstrates how to allocate an mlx5dv_var then using
it for memory address mapping, then freeing the VAR.
```python
from pyverbs.providers.mlx5.mlx5dv import Mlx5VAR
from pyverbs.device import Context
import mmap

ctx = Context(name='rocep0s8f0')
var = Mlx5VAR(ctx)
var_map = mmap.mmap(fileno=ctx.cmd_fd, length=var.length, offset=var.mmap_off)
# There is no munmap method in mmap Python module, but by closing the mmap
# instance the memory is unmapped.
var_map.close()
var.close()
```

##### MLX5 PP
Packet Pacing (PP) entry can be used for some device commands over the DEVX
interface. It allows a rate-limited flow configuration on SQs.
The following code snippet demonstrates how to allocate an mlx5dv_pp with rate
limit value of 5, then frees the entry.
```python
from pyverbs.providers.mlx5.mlx5dv import Mlx5Context, Mlx5DVContextAttr, Mlx5PP
import pyverbs.providers.mlx5.mlx5_enums as e

# The device must be opened as DEVX context
mlx5dv_attr = Mlx5DVContextAttr(e.MLX5DV_CONTEXT_FLAGS_DEVX)
ctx = Mlx5Context(attr=mlx5dv_attr, name='rocep0s8f0')
rate_limit_inbox = (5).to_bytes(length=4, byteorder='big', signed=True)
pp = Mlx5PP(ctx, rate_limit_inbox)
pp.close()
```

##### MLX5 UAR
User Access Region (UAR) is part of PCI address space that is mapped for direct
access to the HCA from the CPU.
The UAR is needed for some device commands over the DevX interface.
The following code snippet demonstrates how to allocate and free an
mlx5dv_devx_uar.
```python
from pyverbs.providers.mlx5.mlx5dv import Mlx5UAR
from pyverbs.device import Context

ctx = Context(name='rocep0s8f0')
uar = Mlx5UAR(ctx)
uar.close()
```

##### Import device, PD and MR
Importing a device, PD and MR enables processes to share their context and then
share PDs and MRs that is associated with.
A process creates a device and then uses some of the Linux systems calls to dup
its 'cmd_fd' member which lets other process to obtain ownership.
Once other process obtains the 'cmd_fd' it can import the device, then PD(s) and
MR(s) to share these objects.
Like in C, Pyverbs users are responsible for unimporting the imported objects
(which will also close the Pyverbs instance in our case) after they finish using
them, and they have to sync between the different processes in order to
coordinate the closure of the objects.
Unlike in C, closing the underlying objects is currently supported only via the
"original" object (meaning only by the process that creates them) and not via
the imported object. This limitation is made because currently there's no
reference or relation between different Pyverbs objects in different processes.
But it's doable and might be added in the future.
Here is a demonstration of importing a device, PD and MR in one process.
```python
from pyverbs.device import Context
from pyverbs.pd import PD
from pyverbs.mr import MR
import pyverbs.enums as e
import os

ctx = Context(name='ibp0s8f0')
pd = PD(ctx)
mr = MR(pd, 100, e.IBV_ACCESS_LOCAL_WRITE)
cmd_fd_dup = os.dup(ctx.cmd_fd)
imported_ctx = Context(cmd_fd=cmd_fd_dup)
imported_pd = PD(imported_ctx, handle=pd.handle)
imported_mr = MR(imported_pd, handle=mr.handle)
# MRs can be created as usual on the imported PD
secondary_mr = MR(imported_pd, 100, e.IBV_ACCESS_REMOTE_READ)
# Must manually unimport the imported objects (which close the object and frees
# other resources that use them) before closing the "original" objects.
# This prevents unexpected behaviours caused by the GC.
imported_mr.unimport()
imported_pd.unimport()
```


##### Flow Steering
Flow steering rules define packet matching done by the hardware.
A spec describes packet matching on a specific layer (L2, L3 etc.).
A flow is a collection of specs.
A user QP can attach to flows in order to receive specific packets.

###### Flow and FlowAttr

```python
from pyverbs.qp import QPCap, QPInitAttr, QPAttr, QP
from pyverbs.flow import FlowAttr, Flow
from pyverbs.spec import EthSpec
import pyverbs.device as d
import pyverbs.enums as e
from pyverbs.pd import PD
from pyverbs.cq import CQ


ctx = d.Context(name='rocep0s8f0')
pd = PD(ctx)
cq = CQ(ctx, 100, None, None, 0)
cap = QPCap(100, 10, 1, 1, 0)
qia = QPInitAttr(cap=cap, qp_type = e.IBV_QPT_UD, scq=cq, rcq=cq)
qp = QP(pd, qia, QPAttr())

# Create Eth spec
eth_spec = EthSpec(ether_type=0x800, dst_mac="01:50:56:19:20:a7")
eth_spec.src_mac = "24:8a:07:a5:28:c8"
eth_spec.src_mac_mask = "ff:ff:ff:ff:ff:ff"

# Create Flow
flow_attr = FlowAttr(num_of_specs=1)
flow_attr.specs.append(eth_spec)
flow = Flow(qp, flow_attr)
```

###### Specs
Each spec holds a specific network layer parameters for matching. To enforce
the match, the user sets a mask for each parameter. If the bit is set in the
mask, the corresponding bit in the value should be matched.
Packets coming from the wire are matched against the flow specification. If a
match is found, the associated flow actions are executed on the packet. In
ingress flows, the QP parameter is treated as another action of scattering the
packet to the respected QP.


###### Notes
* When creating specs mask will be set to FF's to all the given values (unless
provided by the user). When editing a spec mask should be specified explicitly.
* If a field is not provided its value and mask will be set to zeros.
* Hardware only supports full / empty masks.
* Ethernet, IPv4, TCP/UDP, IPv6 and ESP specs can be inner (IBV_FLOW_SPEC_INNER),
but set to outer by default.


###### Ethernet spec
Example of creating and editing Ethernet spec
```python
from pyverbs.spec import EthSpec
eth_spec = EthSpec(src_mac="ab:cd:ef:ab:cd:ef", vlan_tag=0x123, is_inner=1)
eth_spec.dst_mac = "de:de:de:00:de:de"
eth_spec.dst_mac_mask = "ff:ff:ff:ff:ff:ff"
eth_spec.ether_type = 0x321
eth_spec.ether_type_mask = 0xffff
# Resulting spec
print(f'{eth_spec}')
```
Below is the output when printing the spec.

    Spec type       : IBV_FLOW_SPEC_INNER IBV_FLOW_SPEC_ETH
    Size            : 40
    Src mac         : ab:cd:ef:ab:cd:ef    mask: ff:ff:ff:ff:ff:ff
    Dst mac         : de:de:de:00:de:de    mask: ff:ff:ff:ff:ff:ff
    Ether type      : 8451                 mask: 65535
    Vlan tag        : 8961                 mask: 65535


##### MLX5 DevX Objects
A DevX object represents some underlay firmware object, the input command to
create it is some raw data given by the user application which should match the
device specification.
Upon successful creation, the output buffer includes the raw data from the device
according to its specification and is stored in the Mlx5DevxObj instance. This
data can be used as part of related firmware commands to this object.
In addition to creation, the user can query/modify and destroy the object.

Although weakrefs and DevX objects closure are added and handled by
Pyverbs, the users must manually close these objects when finished, and
should not let them be handled by the GC, or by closing the Mlx5Context directly,
since there's no guarantee that the DevX objects are closed in the correct order,
because Mlx5DevxObj is a general class that can be any of the device's available
objects.
But Pyverbs does guarantee to close DevX UARs and UMEMs in order, and after
closing the other DevX objects.

The following code snippet shows how to allocate and destroy a PD object over DevX.
```python
from pyverbs.providers.mlx5.mlx5dv import Mlx5Context, Mlx5DVContextAttr, Mlx5DevxObj
import pyverbs.providers.mlx5.mlx5_enums as dve
import struct

attr = Mlx5DVContextAttr(dve.MLX5DV_CONTEXT_FLAGS_DEVX)
ctx = Mlx5Context(attr, 'rocep8s0f0')
MLX5_CMD_OP_ALLOC_PD = 0x800
MLX5_CMD_OP_ALLOC_PD_OUTLEN = 0x10
cmd_in = struct.pack('!H14s', MLX5_CMD_OP_ALLOC_PD, bytes(0))
pd = Mlx5DevxObj(ctx, cmd_in, MLX5_CMD_OP_ALLOC_PD_OUTLEN)
pd.close()
```
