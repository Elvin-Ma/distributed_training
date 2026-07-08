# 1 SymmetricMemory 背景介绍

## 1.1 SymmetricMemory 基类

```c++
class TORCH_API SymmetricMemory : public torch::CustomClassHolder {
 public:
  ~SymmetricMemory() override = default;

  virtual std::vector<void*> get_buffer_ptrs() = 0;
  virtual std::vector<void*> get_signal_pad_ptrs() = 0;

  // get_buffer_ptrs_dev() and get_signal_pad_ptrs_dev() each return a pointer
  // to a device array of size world_size, containing buffer pointers and
  // signal pad pointers, respectively.
  virtual void** get_buffer_ptrs_dev() = 0;
  virtual void** get_signal_pad_ptrs_dev() = 0;
  virtual size_t get_buffer_size() = 0;
  size_t get_signal_pad_size();

  virtual size_t get_offset() = 0;

  virtual bool has_multicast_support() = 0;
  virtual void* get_multicast_ptr() = 0;

  at::Tensor get_buffer(
      int rank,
      c10::IntArrayRef sizes,
      c10::ScalarType dtype,
      int64_t storage_offset);

  at::Tensor get_signal_pad(
      int rank,
      c10::IntArrayRef sizes,
      std::optional<c10::ScalarType> dtype = std::nullopt,
      int64_t storage_offset = 0);

  at::Tensor get_remote_tensor(
      int peer,
      c10::IntArrayRef sizes,
      c10::ScalarType dtype);

  virtual void barrier(int channel, size_t timeout_ms) = 0;
  virtual void put_signal(int dst_rank, int channel, size_t timeout_ms) = 0;
  virtual void wait_signal(int src_rank, int channel, size_t timeout_ms) = 0;

  virtual int get_rank() = 0;
  virtual int get_world_size() = 0;
  virtual c10::Device get_device() = 0;

  virtual const std::vector<int>& get_rank_to_global_rank() {
    TORCH_CHECK(false, "NYI");
  }

  virtual int* get_rank_to_global_rank_dev() {
    TORCH_CHECK(false, "NYI");
  }

  // Returns true if *all* peers within the group are accessible via direct
  // memory load and store.
  virtual bool world_within_direct_access() {
    TORCH_CHECK(false, "NYI");
  }
};
```

## 1.2 SymmetricMemoryAllocator
```c++
class SymmetricMemoryAllocator : public c10::intrusive_ptr_target {
 public:
  ~SymmetricMemoryAllocator() override = default;

  virtual void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) = 0;

  virtual void free(void* ptr) = 0;
  virtual size_t get_alloc_size(void* ptr) = 0;
  virtual c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) = 0;
  virtual bool has_multicast_support(int device_idx) = 0;
  virtual c10::DeviceType supported_device_type() = 0;
  virtual std::string name() = 0;
};
```

## 1.3  三种不同的 backend 实现方式:

```c++
class CUDASymmetricMemory : public SymmetricMemory;
class NVSHMEMSymmetricMemory : public SymmetricMemory;
class NCCLSymmetricMemory : public SymmetricMemory
```

```sh
SymmetricMemory 抽象接口
        |
        +-- CUDASymmetricMemory      // CUDA VMM / IPC / multicast 实现
        |
        +-- NVSHMEMSymmetricMemory   // NVSHMEM 实现
        |
        +-- NCCLSymmetricMemory      // NCCL 相关实现
```

- 使用时可以指定使用哪个 backend:

```py
import torch.distributed._symmetric_memory as symm_mem

symm_mem.set_backend("NVSHMEM")
# or
symm_mem.set_backend("NCCL")
# or
symm_mem.set_backend("CUDA")
```

- 也可以通过环境变量指定 backend:

```sh
TORCH_SYMMMEM=CUDA
TORCH_SYMMMEM=NVSHMEM
TORCH_SYMMMEM=NCCL
```

## 1.4 python 接口

pytorch/torch/distributed/_symmetric_memory/__init__.py 内暴露了很多symmetric memory python 接口:
```sh
def enable_symm_mem_for_group(group_name: c10d.GroupName) -> None:

def empty(*size: Any, dtype: _dtype | None = None, device: _device | None = None) -> torch.Tensor:

def rendezvous(tensor: torch.Tensor, group: c10d.GroupName | ProcessGroup) -> _SymmetricMemory:

def set_backend(name: Literal["NVSHMEM", "CUDA", "NCCL"]) -> None:

def set_signal_pad_size(size: int) -> None:

# An internal map from device to the symmetric memory pool for that device.
_symm_mem_pools: dict[_device, torch.cuda.MemPool] = {}

def get_mem_pool(device: _device) -> torch.cuda.MemPool:
```

- class _SymmetricMemory:

```py
# pytorch/torch/distributed/_symmetric_memory/__init__.py
class _SymmetricMemory:
    @staticmethod
    def set_group_info(
        group_name: str,
        rank: int,
        world_size: int,
        store: Store,
    ) -> None: ...
    @staticmethod
    def empty_strided_p2p(
        size: torch.types._size,
        stride: torch.types._size,
        dtype: torch.dtype,
        device: torch.device,
        group_name: str | None = None,
        alloc_id: int | None = None,
    ) -> torch.Tensor: ...
    @staticmethod
    def has_multicast_support(
        device_type: DeviceType,
        device_idx: int,
    ) -> bool: ...
    # Set Symmetric Memory allocation backend.
    @staticmethod
    def set_backend(name: str) -> None: ...
    @staticmethod
    def get_backend(device: torch.device) -> Optional[str]: ...
    @staticmethod
    def get_mempool_allocator(device: torch.device) -> Any: ...
    signal_pad_size: int
    @property
    def rank(self) -> int: ...
    @property
    def world_size(self) -> int: ...
    @staticmethod
    def rendezvous(
        tensor: torch.Tensor, group_name: str | None = None
    ) -> _SymmetricMemory: ...
    def get_buffer(
        self,
        rank: int,
        sizes: torch.types._size,
        dtype: torch.dtype,
        storage_offset: int | None = 0,
    ) -> torch.Tensor: ...
    def get_signal_pad(
        self,
        rank: int,
        sizes: torch.types._size = [],
        dtype: torch.dtype | None = None,
        storage_offset: int | None = 0,
    ) -> torch.Tensor: ...
    def barrier(self, channel: int = 0, timeout_ms: int = 0) -> None: ...
    def put_signal(
        self,
        dst_rank: int,
        channel: int = 0,
        timeout_ms: int = 0,
    ) -> None: ...
    def wait_signal(
        self,
        src_rank: int,
        channel: int = 0,
        timeout_ms: int = 0,
    ) -> None: ...
    def get_remote_tensor(
        self,
        peer: int,
        sizes: torch.types._size,
        dtype: torch.dtype,
    ) -> torch.Tensor: ...
    @staticmethod
    def memset32(
        tensor: torch.Tensor, offset: int, val: int, count: int = 1
    ) -> torch.Tensor: ...
    @staticmethod
    def stream_write_value32(
        tensor: torch.Tensor, offset: int, val: int
    ) -> torch.Tensor: ...
    @property
    def buffer_ptrs(self) -> list[int]: ...
    @property
    def buffer_ptrs_dev(self) -> int: ...
    @property
    def signal_pad_ptrs(self) -> list[int]: ...
    @property
    def signal_pad_ptrs_dev(self) -> int: ...
    @property
    def multicast_ptr(self) -> int: ...
    @property
    def buffer_size(self) -> int: ...
    @property
    def device(self) -> torch.device: ...
```

- pybind

```c++
  using SymmetricMemory = ::c10d::symmetric_memory::SymmetricMemory;
  py::class_<SymmetricMemory, c10::intrusive_ptr<SymmetricMemory>>(
      module, "_SymmetricMemory")
      .def_static("set_group_info", &::c10d::symmetric_memory::set_group_info)
      .def_static(
          "empty_strided_p2p",
          ::c10d::symmetric_memory::empty_strided_p2p,
          py::arg("size"),
          py::arg("stride"),
          py::arg("dtype"),
          py::arg("device"),
          py::arg("group_name") = py::none(),
          py::arg("alloc_id") = py::none())
      .def_static(
          "rendezvous",
          &::c10d::symmetric_memory::rendezvous,
          py::arg("tensor"),
          py::arg("group_name") = py::none())
      .def_static(
          "has_multicast_support",
          &::c10d::symmetric_memory::has_multicast_support)
      .def_static("set_backend", &::c10d::symmetric_memory::set_backend)
      .def_static("get_backend", &::c10d::symmetric_memory::get_backend)
      .def_property_static(
          "signal_pad_size",
          [](py::object /* self */) {
            return ::c10d::symmetric_memory::get_signal_pad_size();
          },
          [](py::object /* self */, size_t size) {
            ::c10d::symmetric_memory::set_signal_pad_size(size);
          })
      .def_static(
          "get_mempool_allocator",
          &::c10d::symmetric_memory::get_mempool_allocator)
      .def_property_readonly("rank", &SymmetricMemory::get_rank)
      .def_property_readonly("world_size", &SymmetricMemory::get_world_size)
      .def_property_readonly(
          "buffer_ptrs",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            std::vector<uintptr_t> ret;
            for (auto ptr : symm_mem->get_buffer_ptrs()) {
              ret.push_back(reinterpret_cast<uintptr_t>(ptr));
            }
            return ret;
          })
      .def_property_readonly(
          "buffer_ptrs_dev",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            return reinterpret_cast<uintptr_t>(symm_mem->get_buffer_ptrs_dev());
          })
      .def_property_readonly(
          "signal_pad_ptrs",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            std::vector<uintptr_t> ret;
            for (auto ptr : symm_mem->get_signal_pad_ptrs()) {
              ret.push_back(reinterpret_cast<uintptr_t>(ptr));
            }
            return ret;
          })
      .def_property_readonly(
          "signal_pad_ptrs_dev",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            return reinterpret_cast<uintptr_t>(
                symm_mem->get_signal_pad_ptrs_dev());
          })
      .def_property_readonly(
          "multicast_ptr",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            return reinterpret_cast<uintptr_t>(symm_mem->get_multicast_ptr());
          })
      .def_property_readonly("buffer_size", &SymmetricMemory::get_buffer_size)
      .def_property_readonly("offset", &SymmetricMemory::get_offset)
      .def_property_readonly("device", &SymmetricMemory::get_device)
      // Convert the pybind `_SymmetricMemory` object into the TorchBind custom
      // class object (`__torch__.torch.classes.c10d.SymmetricMemory`) so it can
      // be passed to dispatcher ops that expect the TorchBind type.
      .def(
          "boxed",
          [](c10::intrusive_ptr<SymmetricMemory> self) {
            return torch::jit::toPyObject(c10::IValue(std::move(self)));
          })
      .def(
          "get_buffer",
          &SymmetricMemory::get_buffer,
          py::arg("rank"),
          py::arg("sizes"),
          py::arg("dtype"),
          py::arg("storage_offset") = 0)
      .def(
          "get_signal_pad",
          &SymmetricMemory::get_signal_pad,
          py::arg("rank"),
          py::arg("sizes") = py::list(),
          py::arg("dtype") = py::none(),
          py::arg("storage_offset") = 0)
      .def(
          "barrier",
          &SymmetricMemory::barrier,
          py::arg("channel") = 0,
          py::arg("timeout_ms") = 0)
      .def(
          "put_signal",
          &SymmetricMemory::put_signal,
          py::arg("dst_rank"),
          py::arg("channel") = 0,
          py::arg("timeout_ms") = 0)
      .def(
          "wait_signal",
          &SymmetricMemory::wait_signal,
          py::arg("src_rank"),
          py::arg("channel") = 0,
          py::arg("timeout_ms") = 0)
      .def(
          "get_remote_tensor",
          &SymmetricMemory::get_remote_tensor,
          py::arg("peer"),
          py::arg("sizes"),
          py::arg("dtype"))
      // Util functions that are often used together with symmetric memory but
      // not necessarily directly on symmetric memory.
      .def_static(
          "stream_write_value32",
          [](at::Tensor& input, int64_t offset, int64_t val) {
            // The range of `val` is checked inside the op
            auto op =
                c10::Dispatcher::singleton()
                    .findSchemaOrThrow("symm_mem::stream_write_value32_", "")
                    .typed<at::Tensor(at::Tensor&, int64_t, int64_t)>();
            return op.call(input, offset, val);
          },
          py::arg("input"),
          py::arg("offset"),
          py::arg("val"))
      .def_static(
          "memset32",
          [](at::Tensor& input, int64_t offset, int64_t val, int64_t count) {
            // The range of `val` is checked inside the op
            auto op = c10::Dispatcher::singleton()
                          .findSchemaOrThrow("symm_mem::memset32_", "")
                          .typed<at::Tensor(
                              at::Tensor&, int64_t, int64_t, int64_t)>();
            return op.call(input, offset, val, count);
          },
          py::arg("input"),
          py::arg("offset"),
          py::arg("val"),
          py::arg("count") = 1);
```

## 1.5 SymmetricMemory 实现的通信和通算并行算子
```c++

```


## 1.6 SymmetricMemory Tensor 生命周期管理

1. local symmetric tensor 创建

- 通过_symmetric_memory.empty 创建 local tensor, c++ 侧对应函数 SymmetricMemory::empty_strided_p2p;
- 通过传入deleter 函数, 当ref_count == 0 时自动删除.


```c++
at::Tensor empty_strided_p2p(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::ScalarType dtype,
    c10::Device device,
    const std::optional<std::string>& group_name,
    std::optional<uint64_t> alloc_id) {
  if (alloc_id.has_value()) {
    return empty_strided_p2p_persistent(
        size, stride, dtype, device, group_name, *alloc_id);
  }
  const size_t numel = std::accumulate(
      size.begin(),
      size.end(),
      static_cast<size_t>(1),
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      std::multiplies<size_t>());
  const size_t element_size = c10::elementSize(dtype);
  const size_t alloc_size = numel * element_size;

  auto allocator = get_allocator(device.type());
  void* dev_ptr = allocator->alloc(alloc_size, device.index(), group_name);

  auto options = at::TensorOptions().dtype(dtype).device(device);
  return at::from_blob(
      dev_ptr,
      size,
      stride,
      [allocator = std::move(allocator)](void* ptr) { allocator->free(ptr); },
      options);
}
```

2. peer symmetric tensor 获取

- 通过 _symmetric_memory.get_buffer 获取 peer symmetric tensor, c++ 侧是 SymmetricMemory.get_buffer;
- peer symmetric tensor 在当前进程不会被删除，没有deleter 函数.

```c++
static at::Tensor get_buffer_at_byte_offset(
    SymmetricMemory* handle,
    int peer,
    c10::IntArrayRef sizes,
    c10::ScalarType dtype,
    size_t offset_bytes) {
  TORCH_CHECK(
      peer >= 0 && peer < handle->get_world_size(),
      "Invalid peer rank: ",
      peer);
  auto peer_ptr = handle->get_buffer_ptrs()[peer];
  TORCH_CHECK(
      peer_ptr != nullptr,
      "Cannot get buffer across nodes, my rank: ",
      handle->get_rank(),
      ", peer: ",
      peer);
  const size_t tensor_bytes = nbytes_of(sizes, dtype);
  const auto req_size = offset_bytes + tensor_bytes;
  const auto buffer_size = handle->get_buffer_size();
  TORCH_CHECK(
      req_size <= buffer_size,
      "SymmetricMemory::get_buffer: the requested size (",
      req_size,
      " bytes) exceeds the allocated size (",
      buffer_size,
      " bytes)");
  auto data_ptr = reinterpret_cast<uint8_t*>(peer_ptr) + offset_bytes;
  auto device = handle->get_device();
  auto options = at::TensorOptions().dtype(dtype).device(device);
  return at::for_blob(data_ptr, sizes)
      .options(options)
      .target_device(device)
      .make_tensor();
}

// Implementation of SymmetricMemory APIs common to all backends

at::Tensor SymmetricMemory::get_buffer(
    int rank,
    c10::IntArrayRef sizes,
    c10::ScalarType dtype,
    int64_t storage_offset) {
  // storage_offset is in element, convert to byte
  const auto offset_bytes = storage_offset * c10::elementSize(dtype);
  return get_buffer_at_byte_offset(this, rank, sizes, dtype, offset_bytes);
}

/**
 * 直接获取对应rendezvous 的 tensor 对应的peer tensor:
 *   hdl_y = symm_mem.rendezvous(y, group=group_name)
 *   y_remote = hdl_y.get_remote_tensor(peer, y.size(), y.dtype)
 *   y_remote.copy_(x)
 *
 **/
at::Tensor SymmetricMemory::get_remote_tensor(
    int peer,
    c10::IntArrayRef sizes,
    c10::ScalarType dtype) {
  return get_buffer_at_byte_offset(this, peer, sizes, dtype, get_offset());
}
```

- get_buffer 和 get_remote_tensor 函数的区别

| API | offset 来源 | offset 单位 | 典型用途 |
|---|---|---|---|
| `get_buffer(rank, sizes, dtype, storage_offset=0)` | 调用者传入 | 元素 | 手动访问某个 rank 的 symmetric buffer 任意位置 |
| `get_remote_tensor(peer, sizes, dtype)` | handle 的 `get_offset()` | byte，内部已有 | 访问 peer 上与当前 rendezvous tensor 对应的远端 tensor |

## 1.7 SymmetricMemory 不同 backend 区别

SymmetricMemory 是否可跨机

PyTorch SymmetricMemory 抽象可以覆盖跨机 group，但默认 CUDA backend 通常主要服务同机 GPU 直接共享；真正跨机更应该看 NVSHMEM backend 或 fabric-handle 环境是否可用。


# 2 CUDASymmetricMemory

CUDASymmetricMemory 本身没有用 NVSHMEM。它是 PyTorch symmetric memory 的 CUDA VMM / CUDA IPC / CUDA multicast 后端，主要用的是 CUDA Driver API 这一套：

```sh
cuMemCreate
cuMemAddressReserve
cuMemMap
cuMemSetAccess
cuMemExportToShareableHandle
cuMemImportFromShareableHandle
cuMemRelease
```

- 一个block 里显存分段

```sh
block base ptr
|
|-- user buffer:      [0, buffer_size)
|
|-- padding/alignment
|
|-- signal pad:       [signal_pad_offset, block_size)
```

## 2.1  AllocationRef

AllocationRef 的核心作用是：用 RAII 管住一段 CUDA symmetric memory 的底层资源生命周期。

AllocationRef 不是“用户可见的 symmetric memory handle”，而是更底层的资源包装器，负责保存并最终释放。

Ref 的含义很重要：不是普通裸指针，而是一个引用计数对象。多个地方可以共同持有它，只要还有一个 intrusive_ptr<AllocationRef> 活着，底层 CUDA allocation 就不会释放。

```c++
// Resource wrapper that owns a (vaddr, allocation handle) pair. Upon
// destruction, it unmaps the vaddr and releases the allocation handle.
// 资源管理 wrapper，用来管理一段 CUDA VMM 映射：
// 析构时会 unmap 虚拟地址，并 release allocation handle
// 地址映射原理: 每个进程/rank 都只真正分配了自己的 GPU memory，rendezvous 后，
//              它会把别的 rank 的 GPU memory 也映射进自己的地址空间里
// CUDASymmetricMemory: 不像 allocator ExpandableSegment 那样，把多个 rank 的 memory 放进同一段连续 VA 里
struct AllocationRef : public c10::intrusive_ptr_target {
  // 这段 allocation 映射到当前进程后的虚拟地址，也就是可以被 GPU 访问的地址
  void* ptr;
  /**
     using HandleType = CUmemGenericAllocationHandle
     是 CUDA Driver API 里的一种物理 GPU 内存 allocation 句柄；
     它不是一个可以直接读写的指针，而是 cuMemCreate() 创建出来的“内存对象 handle”；
     要真正访问这块内存，还需要把它 map 到某段 GPU 虚拟地址上(虚拟地址才可直接访问)；
     cuMemCreate(&handle, size, &prop, 0);  ---> cuMemAddressReserve(&addr, size, ...);
     cuMemMap(addr, size, 0, handle, 0) --> cuMemSetAccess(addr, size, &desc, 1).
  **/
  HandleType handle;
  // 这段映射/分配的总大小，单位是字节 : user buffer + signal pad = signal_pad_offset + get_signal_pad_size()
  size_t block_size;
  int device_idx;    // 这段 allocation 所属或映射所在的 CUDA device index
  bool is_multicast; // 标记这个 AllocationRef 管理的是不是 CUDA multicast object/mapping

  AllocationRef(
      void* ptr,
      HandleType handle,
      size_t block_size,
      int device_idx,
      bool is_multicast = false);

  ~AllocationRef();
};
```

## 2.2  CUDAPeerAllocInfo

CUDAPeerAllocInfo 保存的是某一个 base allocation 在整个 group 里 rendezvous 之后得到的**共享信息**.

```c++
class CUDAPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  CUDAPeerAllocInfo(
      std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs,
      std::vector<void*> buffers,
      std::vector<void*> signal_pads,
      HandleType mc_handle,
      void* mc_addr,
      size_t buffer_size,
      int local_device_idx,
      int rank,
      int world_size);

 private:
  std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs_;
  /**
      1. CUDASymmetricMemory 这里不是像 allocator 的 ExpandableSegment 那样，
                             把多个 rank 的 memory 放进同一段连续 VA 里.
      2. rendezvous 后，当前 rank 把所有 peer 的 block 分别 import + map,
                        然后把每个 peer 的地址放到 buffers_[rank] 指针表里.
      3. map：
           - cuMemImportFromShareableHandle(&handles[r], ...)；
           - map_block(&buffers[r], handles[r], block->block_size, block->device_idx)：
               - cuMemAddressReserve(...)
               - cuMemMap(...)
      4. 单个 rank 自己的 block 内部是连续的: [ user buffer ][ padding/alignment ][ signal pad ]
   **/
  std::vector<void*> buffers_;
  /**
      1. signal_pad 是 symmetric memory 里专门留出来做跨 rank/GPU 同步的一小块 GPU 内存;
      2. 它是一个“信号槽数组”，每个 rank 都有自己的 signal pad，其他 rank 可以通过 P2P 直接写它，用来表达：
         我已经到达 barrier 了, 我已经把数据写好了, 你现在可以继续读/算了
      3. 在 CUDA backend 里，signal 的基本单位是 uint32_t，因为同步逻辑用 atomic CAS:
         0 -> 1  // put_signal，发送信号
         1 -> 0  // wait_signal，消费信号并清零
      4. 一次成功同步后，signal pad 又回到 0，可以复用
      5. signal_pad 的布局:
         channel 0: [from rank0][from rank1][from rank2]...[from rank W-1]
         channel 1: [from rank0][from rank1][from rank2]...[from rank W-1]
      6. 每一个rank 都有一个 signal pad， 一个signal_pad 有多个channel， 每个channel的size 是world_size
         意味着每个channel 给所有peer rank 留一个 signal 槽.
      7. default_signal_pad_size = symm_max_nblocks * max_cuda_p2p_domain_size * sizeof(uint32_t)
         symm_max_nblocks = 32 ：
           - signal pad 默认预留 32 组同步槽;
           - 对 block-level sync 来说，是最多 32 个 CUDA block（CTA: Cooperative Thread Array）;
           - 对 channel-based sync 来说，是默认最多 32 个 channel;
         max_cuda_p2p_domain_size = 72 : 默认域内GPU 个数
      8. buffer ptr/ base ptr position:
         - signal_pads 开始位置： signal_pad_offset = at::round_up(size, 16UL)
         - signal_pads_size = set signal_pad_size 或 default_signal_pad_size
  **/
  std::vector<void*> signal_pads_;
  // mc: multicast 的意思 ：mc_handle_ 是共享的 multicast object 的 handle
  // multicast object 的 handle, 和 AllocationRef::handle 在 multicast 场景下值可能相同
  HandleType mc_handle_;
  /**
      1. multicast object 被映射到本进程虚拟地址空间后的基地址
         此时：一个 CUDASymmetricMemory 是 base allocation 里的某个切片;
      2. mc_addr_ 是整块 multicast mapping 的起点,
         offset_ 才定位到当前 tensor/handle 对应的位置;
      3. 几种地址对比:
          - 本 rank buffer 地址: block->alloc_ref->ptr 或 buffers_[rank];
          - peer buffer 地址:  buffers_[other_rank];
          - mc_addr_:每个进程本地 map mc_handle 后得到的本地虚拟地址，不保证跨进程数值一致
      4. 不同 rank 的mc_addr_ 不一定一样， 每个 rank 自己用自己的 mc_addr_ + offset 访问，
         就能走到同一个 multicast object 的对应位置；
  **/
  void* mc_addr_;
  size_t buffer_size_;
  int local_device_idx_;
  int rank_;
  int world_size_;
  /**
   * 1. buffers_dev_ 是 GPU 上的一段 raw memory，里面也存着一串 void*。但 raw memory 没 有
   *    std::vector 这个容器类型，只能用 C 风格指针表示“指向数组首元素的指针”;
   * 2. buffers_dev_ device: [ buffers_[0], buffers_[1], buffers_[2], ... ];
   * 3. CPU 代码可以直接访问 std::vector<void*> buffers_;
   * 4. CUDA kernel 不能直接读 host 上的 std::vector。kernel 需要一个 device memory 里的 void**;
   * 5. void**  = 指向“一组 buffer 地址”的数组首地址, buffers_.data() 的类型也是 void**;
   * 6. cudaMemcpy(buffers_dev_, buffers_.data(), arr_size, cudaMemcpyHostToDevice);
  **/
  void** buffers_dev_;
  void** signal_pads_dev_;

  friend class CUDASymmetricMemory;
};
```

## 2.3 CUDASymmetricMemory

```c++
class CUDASymmetricMemory : public SymmetricMemory {
 public:
  // This is mostly a shallow copy that shares the pointer to
  // `CUDAPeerAllocInfo` which corresponds to the base Block. The
  // CUDASymmetricMemory handle is specified by the offset to the base ptr.
  CUDASymmetricMemory(
      const c10::intrusive_ptr<CUDAPeerAllocInfo>& pai,
      size_t offset);

  ~CUDASymmetricMemory() override {};

  // 返回 host 侧的 std::vector<void*>，长度通常是 world_size_。第 i 项是 rank i 的 base buffer 地址
  std::vector<void*> get_buffer_ptrs() override;
  // 返回 host 侧的 signal pad 指针表。signal pad 是用于同步的 P2P 可访问内存区域
  std::vector<void*> get_signal_pad_ptrs() override;
  // 返回 device 侧的 void** 数组，数组里存放各 rank 的 buffer 指针, 自定义CUDA kernel 可以直接拿这个device array 做远端访问
  void** get_buffer_ptrs_dev() override;
  // 返回 device 侧的 signal pad 指针数组。barrier、put_signal、wait_signal 内部 kernel 就是用这个
  void** get_signal_pad_ptrs_dev() override;
  // 返回每个 symmetric buffer 的大小，来自 pai_->buffer_size_
  // 这是 base allocation 的可用 buffer 区大小，不包含 signal pad
  size_t get_buffer_size() override;
  // 返回当前 handle 的字节偏移。
  // 通用实现里 get_remote_tensor(peer, sizes, dtype) 会用这个 offset 从 peer base buffer 上构造 tensor。
  size_t get_offset() override;

  // 判断是否有 CUDA multicast 地址：pai_->mc_addr_ != nullptr
  bool has_multicast_support() override;
  void* get_multicast_ptr() override;

  // 在当前 CUDA stream 上 launch 一个 kernel，让当前 rank 和所有其他 rank 通过 signal pad 互相发/收 signal，形成 group barrier.
  // channel 用于把不同 stream 或不同用途的同步隔离开，避免 signal 串扰
  void barrier(int channel, size_t timeout_ms) override;
  // 向 dst_rank 的 signal pad 写一个 signal　：signal_pads[dst_rank] + world_size * channel + rank
  void put_signal(int dst_rank, int channel, size_t timeout_ms) override;
  // 等待 src_rank 发来的 signal :　signal_pads[rank] + world_size * channel + src_rank
  void wait_signal(int src_rank, int channel, size_t timeout_ms) override;

  int get_rank() override;
  int get_world_size() override;
  c10::Device get_device() override;
  // 判断是否可以 DMA 直接访问memory
  // 如果某个 peer pointer 是 nullptr，仍会拒绝构造远端 tensor；
  // 所以这个返回值表达的是 backend/rendezvous 语义上的 direct access 能力
  bool world_within_direct_access() override;

 private:
  int local_device_idx_;                      // 当前 rank 使用的本地 CUDA device index
  int rank_;                                  // 当前进程/设备在 symmetric memory group 里的 rank, 同步时用它决定 signal 写到哪个槽位
  int world_size_;                            // 当前 group 的 rank 数量
  c10::intrusive_ptr<CUDAPeerAllocInfo> pai_; // 共享的 CUDAPeerAllocInfo
  size_t offset_{0};                          // 当前 handle 相对 base buffer 的字节偏移,
};
```

## 2.4 Block

**Block 不是用户数据本身，而是 allocator 维护的“这块 symmetric memory allocation 的账本”。**

Block 是 CUDASymmetricMemoryAllocator 给“一次底层 symmetric CUDA allocation”保存的元数据。可以把它理解成：用户拿到的是 ptr，allocator 内部用 Block 记住这块内存怎么释放、真实大小、用户 buffer 和 signal pad 怎么切分，以及它已经在哪些 process group 上做过 rendezvous。


```c++
// Metadata associated with each allocation performed by
// `CUDASymmetricMemoryAllocator`.
struct Block : public c10::intrusive_ptr_target {
  c10::intrusive_ptr<AllocationRef> alloc_ref;   // 底层 CUDA allocation 的拥有者引用Ref
  int device_idx;                                // allocation 所在GPU
  size_t block_size;                             // CUDA 实际分配/mapping 的总大小
  size_t buffer_size;                            // 用户请求的可用 buffer 大小
  size_t signal_pad_offset;                      // signal pad 在 allocation 内的位置
  std::optional<std::string> default_group_name; // 默认 rendezvous group
  // 按 group name 缓存的 rendezvous 的结果
  std::map<std::string, c10::intrusive_ptr<CUDAPeerAllocInfo>> symm_mems;

  Block(
      c10::intrusive_ptr<AllocationRef> alloc_ref,
      int device_idx,
      size_t block_size,
      size_t buffer_size,
      size_t signal_pad_offset,
      const std::optional<std::string>& group_name);
};
```

> 注意: rendezvous 是比较贵的，因为要通过 store/IPCs 交换 handle、import peer allocations、map peer memory。所以第一次对某个 group_name 做 rendezvous 时创建 CUDAPeerAllocInfo，之后同一个 Block + group_name 直接复用缓存。

## 2.5 CUDASymmetricMemoryAllocator

```c++
class CUDASymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override;

  void free(void* ptr) override;
  size_t get_alloc_size(void* ptr) override;
  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override;
  bool has_multicast_support(int device_idx) override;
  c10::DeviceType supported_device_type() override;
  std::string name() override;

 private:
  c10::intrusive_ptr<Block> find_block(void* ptr);
  c10::intrusive_ptr<Block> find_block_covering(void* ptr, size_t& offset);

  std::shared_mutex mutex_;
  std::unordered_map<void*, c10::intrusive_ptr<Block>> ptr_to_block_;
  c10::cuda::CUDACachingAllocator::Expandable_Segments_Handle_Type
      handle_type_ = c10::cuda::CUDACachingAllocator::
          Expandable_Segments_Handle_Type::UNSPECIFIED;
};
```

## 2.6 barrier and signal sync

```sh
# signal pads channel 0
rank 0: [0, 0, 0, 0]
rank 1: [0, 0, 0, 0]
rank 2: [0, 0, 0, 0]
rank 3: [0, 0, 0, 0]

# rank 0 put signal:
rank 0: [1, 0, 0, 0]
rank 1: [1, 0, 0, 0]
rank 2: [1, 0, 0, 0]
rank 3: [1, 0, 0, 0]
# rank 1 put signal:
rank 0: [0, 1, 0, 0]
rank 1: [0, 1, 0, 0]
rank 2: [0, 1, 0, 0]
rank 3: [0, 1, 0, 0]
# rank 2 put signal:
rank 0: [0, 0, 1, 0]
rank 1: [0, 0, 1, 0]
rank 2: [0, 0, 1, 0]
rank 3: [0, 0, 1, 0]
# rank 3 put signal:
rank 0: [1, 0, 0, 1]
rank 1: [1, 0, 0, 1]
rank 2: [1, 0, 0, 1]
rank 3: [1, 0, 0, 1]

# ===================================
# rank 0 wait signal:
rank 0: [1, 1, 1, 1]
rank 1: [0, 0, 0, 0]
rank 2: [0, 0, 0, 0]
rank 3: [0, 0, 0, 0]

# rank 1 wait signal:
rank 0: [0, 0, 0, 0]
rank 1: [1, 1, 1, 1]
rank 2: [0, 0, 0, 0]
rank 3: [0, 0, 0, 0]

# rank 2 wait signal:
rank 0: [0, 0, 0, 0]
rank 1: [0, 0, 0, 0]
rank 2: [1, 1, 1, 1]
rank 3: [0, 0, 0, 0]

# rank 3 wait signal:
rank 0: [0, 0, 0, 0]
rank 1: [0, 0, 0, 0]
rank 2: [0, 0, 0, 0]
rank 3: [1, 1, 1, 1]
```

### 2.6.1 barrier_kernal

- collective 级别的同步，所有rank 上的同步.
- barrier kernel 用一个block 内的 world_size 个线程 来完成 world_size 个 信号同步.

```c++
static __global__ void barrier_kernel(
    uint32_t** signal_pads,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms) {
  if (threadIdx.x < world_size) {
    auto target_rank = threadIdx.x;
    if (target_rank == rank) {
      return;
    }
    auto put_success = try_put_signal<std::memory_order_release>(
        signal_pads[target_rank] + world_size * channel + rank, timeout_ms);
    if (!put_success) {
      printf(
          "[FATAL] CUDASymmetricMemory::barrier: rank %d failed to send signal "
          "to rank %d on channel %d after %lu microseconds\n",
          rank,
          target_rank,
          channel,
          timeout_ms);
      trap();
    }
    auto wait_success = try_wait_signal<std::memory_order_acquire>(
        signal_pads[rank] + world_size * channel + target_rank, timeout_ms);
    if (!wait_success) {
      printf(
          "[FATAL] CUDASymmetricMemory::barrier: rank %d failed to receive signal "
          "from rank %d on channel %d after %lu microseconds\n",
          rank,
          target_rank,

}          channel,
          timeout_ms);
      trap();
    }
  }

void CUDASymmetricMemory::barrier(int channel, size_t timeout_ms) {
  check_channel(channel, world_size_);
  c10::cuda::CUDAGuard guard(local_device_idx_);
  barrier_kernel<<<
      1,
      max(at::cuda::warp_size(), world_size_),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t**>(pai_->signal_pads_dev_),
      channel,
      rank_,
      world_size_,
      timeout_ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

### 2.6.2 put_signal_kernal

- **点对点的同步**

CUDA kernel，用来从当前 rank 给另一个 rank 的 signal pad 写一个“信号”.

核心逻辑是：<br>

- **signal_pads[dst_rank] + world_size * channel + rank**
  - signal_pads[dst_rank]：目标 rank 的 signal pad 起始地址
  - channel：逻辑通道，允许同一组 rank 使用不同槽位做不同同步
  - world_size * channel：跳到该 channel 的起点
  - + rank：当前 rank 在这个 channel 中对应的发送槽位

```c++
static __global__ void put_signal_kernel(
    uint32_t** signal_pads,
    int dst_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms) {
  if (threadIdx.x == 0) {
    bool success = try_put_signal<std::memory_order_release>(
        signal_pads[dst_rank] + world_size * channel + rank, timeout_ms);
    if (!success) {
      printf(
          "[FATAL] CUDASymmetricMemory::put_signal: rank %d failed to send signal "
          "to rank %d on channel %d after %lu microseconds\n",
          rank,
          dst_rank,
          channel,
          timeout_ms);
      trap();
    }
  }
}

// 不断尝试把该地址上的 uint32_t 从 0 CAS 成 1;
// 如果地址当前不是 0，说明上一个 signal 还没被对方消费掉，就会继续等；
// 超过 timeout_ms 或 原来就是1 则返回 false.
template <std::memory_order Sem>
__device__ __forceinline__ bool try_put_signal(
    uint32_t* addr,
    size_t timeout_ms) {
  size_t deadline = global_timer_ns() + timeout_ms * ns_per_ms;
  while (cas<Sem>(addr, 0, 1) != 0) {
    if (timeout_ms != 0 && global_timer_ns() > deadline) {
      return false;
    }
  }
  return true;
}
```

### 2.6.3 wait_signal_kernel

- 点对点的同步的wait

```c++
static __global__ void wait_signal_kernel(
    uint32_t** signal_pads,
    int src_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms) {
  if (threadIdx.x == 0) {
    bool success = try_wait_signal<std::memory_order_acquire>(
        signal_pads[rank] + world_size * channel + src_rank, timeout_ms);
    if (!success) {
      printf(
          "[FATAL] CUDASymmetricMemory::wait_signal rank %d failed to receive signal "
          "from rank %d on channel %d after %lu microseconds\n",
          rank,
          src_rank,
          channel,
          timeout_ms);
#if !defined(USE_ROCM)
      __trap();
#else
      assert(0);
#endif
    }
  }
  // __threadfence_block()	仅本 block 内
  // __threadfence()	仅本设备(单个 GPU)内
  // __threadfence_system()	跨设备的整个系统(对端 GPU + 主机)
  __threadfence_system();
}


template <std::memory_order Sem>
__device__ __forceinline__ bool try_wait_signal(
    uint32_t* addr,
    size_t timeout_ms) {
  size_t deadline = global_timer_ns() + timeout_ms * ns_per_ms;
  while (cas<Sem>(addr, 1, 0) != 1) {
    if (timeout_ms != 0 && global_timer_ns() > deadline) {
      return false;
    }
  }
  return true;
}

void CUDASymmetricMemory::wait_signal(
    int src_rank,
    int channel,
    size_t timeout_ms) {
  check_channel(channel, world_size_);
  c10::cuda::CUDAGuard guard(local_device_idx_);
  wait_signal_kernel<<<
      1,
      at::cuda::warp_size(),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t**>(pai_->signal_pads_dev_),
      src_rank,
      channel,
      rank_,
      world_size_,
      timeout_ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

# 3 跨进程共享信息

## 3.1 POSIX_FD 机内信息共享

### 3.1.1 跨进程数据传输的基础数据结构
socket 通过 msghdr 和 cmsghdr 来完成机内跨进程数据传输.

这里的跨进程传输的是GPU分配的物理Handle, fd 指向的根本不是文件,而是一块物理显存对象。把它叫「文件描述符」只是复用了 Unix 这套统一的、可被内核管理和传递的句柄机制。fd 只是个「把手」,真正的资源在内核对象里.

```sh
struct msghdr msg
├── msg_name
│   └── 目标/来源地址，比如 sockaddr_un
│
├── msg_iov
│   └── 普通数据 payload
│       例如发一个 byte、tag、同步消息
│
├── msg_iovlen
│
├── msg_control
│   └── 控制数据缓冲区 control_buf
│       ┌─────────────────────────────┐
│       │ struct cmsghdr              │  <-- cmsg = CMSG_FIRSTHDR(&msg)
│       │ ├── cmsg_len                │
│       │ ├── cmsg_level = SOL_SOCKET │
│       │ └── cmsg_type = SCM_RIGHTS  │
│       ├─────────────────────────────┤
│       │ CMSG_DATA(cmsg)             │
│       │ └── int fd                  │  <-- 真正放 fd 的地方
│       └─────────────────────────────┘
│
├── msg_controllen
│   └── msg_control 这块 buffer 的大小
│
└── msg_flags
```

- 通过 socket path 来配对

```sh
每个进程创建一个 socket
  socket path = /tmp/symm_mem-<pid>

send_fd(dst_pid, fd)
  发送到 /tmp/symm_mem-<dst_pid>

recv_fd()
  在自己的 /tmp/symm_mem-<my_pid> 上收下一条消息
```

### 3.1.2 struct msghdr 和 struct cmsghdr
- sendmsg 和 recvmsg 真正传输的就是struct msghdr;
- 传输的是GPU 物理线程的句柄handle，被设置为fd 放到 msg_control-> __cmsg_data 中;
- sendmsg 和 recvmsg 通过 socket的地址(由各进程号生成) 识别彼此；
- 需要提前用store 的 allgather 达到所有进程的 pid 信息.

```c++
# define CMSG_DATA(cmsg) ((cmsg)->__cmsg_data)

// msg_control 强转成 cmsghdr
#define CMSG_FIRSTHDR(mhdr) \
  ((size_t) (mhdr)->msg_controllen >= sizeof (struct cmsghdr)		      \
   ? (struct cmsghdr *) (mhdr)->msg_control : (struct cmsghdr *) 0)

/* Structure describing messages sent by
   `sendmsg' and received by `recvmsg'.  */
struct msghdr
  {
    void *msg_name;		/* Address to send to/receive from.  */
    socklen_t msg_namelen;	/* Length of address data.  */

    struct iovec *msg_iov;	/* Vector of data to send/receive into.  */
    size_t msg_iovlen;		/* Number of elements in the vector.  */

    void *msg_control;		/* Ancillary data (eg BSD filedesc passing). */
    size_t msg_controllen;	/* Ancillary data buffer length.
				   !! The type should be socklen_t but the
				   definition of the kernel is incompatible
				   with this.  */

    int msg_flags;		/* Flags on received message.  */
  };

/* Structure used for storage of ancillary data object information.  */
struct cmsghdr
  {
    size_t cmsg_len;		/* Length of data in cmsg_data plus length
				   of cmsghdr structure.
				   !! The type should be socklen_t but the
				   definition of the kernel is incompatible
				   with this.  */
    int cmsg_level;		/* Originating protocol.  */
    int cmsg_type;		/* Protocol specific type.  */
    __extension__ unsigned char __cmsg_data[] /* Ancillary data.  */
#endif
  };
```

### 3.1.3 rendezvous 流程

```c++
// step1 : make_peer_alloc_info
c10::intrusive_ptr<SymmetricMemory> CUDASymmetricMemoryAllocator::rendezvous(
    void* ptr,
    const std::optional<std::string>& group_name) {
  size_t offset = 0;
  // 查找对应的 block，中间 ptr
  auto block = find_block_covering(ptr, offset);

  // If found, this block has been rendezvous by the given group
  auto it = block->symm_mems.find(group_name_);
  if (it == block->symm_mems.end()) {
    auto pai = make_peer_alloc_info<false>(ptr, block, group_name_);
    // Cache it with the group name
    it = block->symm_mems.emplace(group_name_, pai).first;
  }

  // Create symm mem handle for this tensor, specified by its offset
  auto pai = it->second;
  return c10::make_intrusive<CUDASymmetricMemory>(pai, offset);
}

template <bool use_fabric_handle>
c10::intrusive_ptr<CUDAPeerAllocInfo> make_peer_alloc_info(
    void* ptr,
    c10::intrusive_ptr<Block> block,
    const std::string& group_name) {
  using BlockHandleType = int;
  BlockHandleType block_handle;

  auto group = resolve_process_group(group_name);
  auto rank = group->getRank();
  auto world_size = group->getSize();
  auto store = group->getStore();

  using IpcChannelType = IpcChannel;
  IpcChannelType ipc_channel;

  auto driver_api = c10::cuda::DriverAPI::get();
  // using the CUDA Driver API to export a GPU memory block as a
  // POSIX file descriptor (FD), so it can be shared across processes via IPC.
  // 由 handle --> 生成 FD
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemExportToShareableHandle_(
      &block_handle,
      block->alloc_ref->handle,
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
      0));

  auto local_req = RendezvousRequest{
      .device_idx = block->device_idx,
      .pid = getpid(),
      .block_size = block->block_size,
      .buffer_size = block->buffer_size,
      .signal_pad_offset = block->signal_pad_offset,
      .has_multicast_support = device_has_multicast_support(block->device_idx)};

  // Populate hostname field for host identification
  gethostname(local_req.hostname, sizeof(local_req.hostname));
  // 先allgather requests 信息
  auto reqs = storeExchange.all_gather(store, rank, world_size, local_req);
  validate_rendezvous_requests(reqs, world_size);

  std::vector<int> pids(world_size);
  for (int r = 0; r < world_size; ++r) {
    pids[r] = reqs[r].pid;
  }

  std::vector<BlockHandleType> imported_handles;
  imported_handles = ipc_channel.all_gather_fds(rank, pids, block_handle);

  std::vector<HandleType> handles(world_size);
  std::vector<void*> buffers(world_size, nullptr);
  std::vector<void*> signal_pads(world_size, nullptr);

  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      handles[r] = block->alloc_ref->handle;
      buffers[r] = ptr;
      signal_pads[r] = (void*)((uintptr_t)ptr + block->signal_pad_offset);
      continue;
    }

    C10_CUDA_DRIVER_CHECK(driver_api->cuMemImportFromShareableHandle_(
        &handles[r],
        (void*)(uintptr_t)imported_handles[r],
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

    map_block(&buffers[r], handles[r], block->block_size, block->device_idx);
    signal_pads[r] = (void*)((uintptr_t)buffers[r] + block->signal_pad_offset);
    if constexpr (!use_fabric_handle) {
      close(imported_handles[r]);
    }
  }

  storeExchange.barrier(store, rank, world_size);
  close(block_handle);

  HandleType mc_handle{};
  void* mc_addr = nullptr;
  bool group_has_multicast_support = check_group_multicast_support(reqs);

  std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs;
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      if (mc_addr != nullptr) {
        alloc_refs.push_back(c10::make_intrusive<AllocationRef>(
            mc_addr, mc_handle, block->block_size, block->device_idx, true));
      }

      alloc_refs.emplace_back(block->alloc_ref);
      continue;
    }
    alloc_refs.push_back(c10::make_intrusive<AllocationRef>(
        buffers[r], handles[r], block->block_size, block->device_idx));
  }

  auto pai = c10::make_intrusive<CUDAPeerAllocInfo>(
      std::move(alloc_refs),
      std::move(buffers),
      std::move(signal_pads),
      mc_handle,
      mc_addr,
      block->buffer_size,
      block->device_idx,
      rank,
      world_size);

  return pai;
}

// 收集不同进程的 fd --> 跨进程 handle 会变化，但实际物理显存不变
std::vector<int> IpcChannel::all_gather_fds(
    int rank,
    const std::vector<int>& pids,
    int fd) {
  int world_size = static_cast<int>(pids.size());
  std::vector<int> fds(pids.size());
  fds[rank] = fd;

  int dst_rank = (rank + 1) % world_size;
  for (int step = 1; step < world_size; ++step) {
    int src_rank = (rank + world_size - step) % world_size;
    send_fd(pids[dst_rank], fd);
    // cpu blocking
    fd = recv_fd();
    fds[src_rank] = fd;
  }
  return fds;
}


void IpcChannel::send_fd(int dst_pid, int fd) {
  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  auto socket_name = get_socket_name(dst_pid);
  std::copy(socket_name.begin(), socket_name.end(), addr.sun_path);

  struct iovec io = {.iov_base = (void*)"fd", .iov_len = 2};
  char cbuf[CMSG_SPACE(sizeof(int))];
  memset(cbuf, 0, sizeof(cbuf));

  // Create message header
  struct msghdr msg{
      .msg_name = (void*)&addr,
      .msg_namelen = sizeof(struct sockaddr_un),
      .msg_iov = &io,
      .msg_iovlen = 1,
      .msg_control = cbuf,
      .msg_controllen = sizeof(cbuf)};

  auto cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;

  if (fd != -1) {
    std::copy(
        reinterpret_cast<const char*>(&fd),
        reinterpret_cast<const char*>(&fd) + sizeof(fd),
        reinterpret_cast<char*>(CMSG_DATA(cmsg)));
  } else {
    msg.msg_controllen = 0;
  }

  // Finally send the message
  TORCH_CHECK(
      sendmsg(socket_, &msg, 0) > 0,
      "Failed to send fd: ",
      c10::utils::str_error(errno));
}

int IpcChannel::recv_fd() {
  char buf[2];
  memset(&buf, 0, sizeof(buf));
  struct iovec io = {.iov_base = (void*)buf, .iov_len = sizeof(buf)};

  char cbuf[CMSG_SPACE(sizeof(int))];
  memset(cbuf, 0, sizeof(cbuf));

  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  std::copy(socket_name_.begin(), socket_name_.end(), addr.sun_path);

  // Prepare message header
  struct msghdr msg = {
      .msg_name = (void*)&addr,
      .msg_namelen = sizeof(struct sockaddr_un),
      .msg_iov = &io,
      .msg_iovlen = 1,
      .msg_control = cbuf,
      .msg_controllen = sizeof(cbuf)};

  // Receive message on socket_
  TORCH_CHECK(
      recvmsg(socket_, &msg, 0) > 0,
      "Failed to receive fd: ",
      c10::utils::str_error(errno));

  if (msg.msg_controllen == 0) {
    return -1;
  }

  return *reinterpret_cast<int*>(CMSG_DATA(cmsg));
}
```
