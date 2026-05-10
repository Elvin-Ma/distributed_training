# pytorch allocator\

# SegmentInfo

SegmentInfo.address 是 GPU 虚拟地址（device virtual address，位于 CUDA 的统一虚拟地址空间 UVA 中），不是物理地址;

一次底层显存申请得到的连续虚拟地址段(segment)" ---> 容易被口头叫"物理段";

```c++
// Struct containing info of a memory segment (i.e. one contiguous cudaMalloc).
struct SegmentInfo {
  c10::DeviceIndex device = 0;
  size_t address = 0;
  size_t total_size = 0;
  size_t requested_size = 0; // unrounded, actually requested size
  size_t allocated_size = 0;
  size_t active_size = 0;
  cudaStream_t stream = nullptr;
  bool is_large = false;
  bool is_expandable = false;
  MempoolId_t owner_private_pool_id = {0, 0};
  std::vector<BlockInfo> blocks;
  std::shared_ptr<GatheredContext> context_when_allocated;
}
```


# ncclcomm->registerSegment

```sh
一次 cudaMalloc → 一个 Segment（连续的大块物理内存）
    └── Block 1（已分配给某个 tensor）
    └── Block 2（已分配给另一个 tensor）
    └── Block 3（空闲，在 cache 中）
```


